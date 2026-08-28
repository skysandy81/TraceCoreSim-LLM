"""Core decoder-only language model simulator.

实现一个小型 decoder-only Transformer，包括 embedding、
GQA attention、RoPE、RMSNorm、SwiGLU、residual、final projection 和 KV cache。
This module implements a small decoder-only Transformer with embedding,
GQA attention, RoPE, RMSNorm, SwiGLU, residuals, final projection, and KV cache.
"""

import numpy as np

from .config import (
    d_ff,
    d_model,
    head_dim,
    kv_dim,
    kv_head_dim,
    lora_rank,
    lora_scaling,
    lr_dpo,
    lr_ppo,
    lr_pretrain,
    n_heads,
    n_kv_heads,
    n_layers,
    use_rope,
    weight_decay,
)
from .data import token_loss_mask, vocab_size
from .ops import dropout, get_causal_mask, rms_norm, rope_embedding, softmax, swiglu
from .optim import AdamW


class LoRALinear:
    """Minimal LoRA adapter wrapper around a frozen base weight.

    LoRA的思想是冻结原始权重W，只训练低秩矩阵A/B；这里保留结构演示，但默认训练流程没有启用adapter-only tuning。
    LoRA freezes the base weight W and trains low-rank matrices A/B.
    This class demonstrates the structure, but default training does not enable adapter-only tuning.
    """

    def __init__(self, origin_weight, rank=lora_rank):
        """Initialize low-rank factors.

        B 初始化为 0，使初始 LoRA 输出为 0，不改变 base layer。
        B starts at zero so the initial LoRA path does not change the base layer.
        """
        self.W = origin_weight
        in_dim, out_dim = origin_weight.shape
        self.A = np.random.randn(in_dim, rank) * 0.01
        self.B = np.zeros((rank, out_dim))

    def forward(self, x):
        """Compute base linear output plus scaled LoRA delta.

        输出 = xW + scale * xAB。
        Output = xW + scale * xAB.
        """
        base = x @ self.W
        lora_out = (x @ self.A) @ self.B * lora_scaling
        return base + lora_out

    def get_params(self):
        """Return trainable adapter parameters.

        真实 LoRA 微调通常只把 A/B 交给优化器。
        Real LoRA fine-tuning usually passes only A/B to the optimizer.
        """
        return {"A": self.A, "B": self.B}


class GQAAttention:
    """Grouped-query self-attention with RoPE and KV cache support.

    Q有`n_heads`个头，K/V只有`n_kv_heads`个头；推理时K/V可以缓存，后续token只需追加新的K/V。
    Q has `n_heads`, while K/V have `n_kv_heads`. During inference, K/V can be cached and extended one token at a time.
    """

    def __init__(self):
        """Initialize attention projections.

        Wk/Wv 的输出维度是 `n_kv_heads * head_dim`，这是 GQA 和普通MHA的关键区别。
        Wk/Wv output `n_kv_heads * head_dim`, the key difference between
        GQA and standard MHA.
        """
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, kv_dim) * 0.1
        self.Wv = np.random.randn(d_model, kv_dim) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1
        self.drop_attn = 0.1

    def forward(self, x, mask=None, training=True, return_attn=False, past_kv=None):
        """Run one causal self-attention block.

        输入 x 形状为 `[seq, d_model]`。返回 attention 输出和当前层
        K/V cache；如果 `return_attn=True`，也返回注意力权重。
        Input x has shape `[seq, d_model]`. Returns attention output and
        this layer's K/V cache; optionally returns attention weights.
        """
        seq_len = x.shape[0]

        # 线性投影得到 Q/K/V。真实模型中这些通常是一个或几个大矩阵乘法。
        # Linear projections produce Q/K/V, usually implemented as large matmuls.
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        start_pos = 0 if past_kv is None else past_kv[0].shape[1]

        # 把 `[seq, channels]` 拆成 `[heads, seq, head_dim]`，便于逐头注意力。
        # Reshape `[seq, channels]` into `[heads, seq, head_dim]` for per-head attention.
        Q = Q.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
        K = K.reshape(seq_len, n_kv_heads, kv_head_dim).swapaxes(0, 1)
        V = V.reshape(seq_len, n_kv_heads, kv_head_dim).swapaxes(0, 1)

        # RoPE 必须在拼接 cache 前对新 token 的 Q/K 施加正确绝对位置。
        # Apply RoPE to new-token Q/K at their absolute positions before cache append.
        if use_rope:
            for h in range(n_heads):
                Q[h] = rope_embedding(Q[h], start_pos=start_pos)
            for h in range(n_kv_heads):
                K[h] = rope_embedding(K[h], start_pos=start_pos)

        if past_kv is not None:
            # 增量推理时复用旧 K/V，只追加当前 step 的 K/V。
            # Incremental decoding reuses old K/V and appends only the current step.
            past_K, past_V = past_kv
            K_cache = np.concatenate([past_K, K], axis=1)
            V_cache = np.concatenate([past_V, V], axis=1)
        else:
            K_cache = K
            V_cache = V

        # GQA 通过重复 K/V 头来匹配 Q 头数，模拟共享 K/V 的推理节省。
        # GQA repeats K/V heads to match Q heads, modeling shared-K/V inference savings.
        repeat_times = n_heads // n_kv_heads
        K_attn = np.repeat(K_cache, repeat_times, axis=0)
        V_attn = np.repeat(V_cache, repeat_times, axis=0)

        # 缩放点积注意力。除以 sqrt(head_dim) 可稳定 softmax 分布。
        # Scaled dot-product attention; division by sqrt(head_dim) stabilizes softmax.
        attn_score = Q @ K_attn.swapaxes(-1, -2) / np.sqrt(head_dim)
        if mask is None:
            mask = get_causal_mask(seq_len, K_attn.shape[1], start_pos)
        attn_score += np.expand_dims(mask, axis=0)
        attn_weight = softmax(attn_score)
        attn_weight = dropout(attn_weight, self.drop_attn, training)

        # 注意力权重对 V 加权求和，然后拼接所有头并投影回 d_model。
        # Attention weights mix V, then heads are concatenated and projected to d_model.
        attn_out = attn_weight @ V_attn
        attn_out = attn_out.swapaxes(0, 1).reshape(seq_len, d_model)
        final_out = attn_out @ self.Wo
        final_out = dropout(final_out, 0.1, training)

        if return_attn:
            return final_out, attn_weight, (K_cache, V_cache)
        return final_out, (K_cache, V_cache)


class DecoderLayer:
    """One pre-norm Transformer decoder layer.

    结构为 RMSNorm -> causal self-attention -> residual -> RMSNorm -> SwiGLU FFN -> residual，接近LLaMA/GPT类decoder block。
    Structure is RMSNorm -> causal self-attention -> residual -> RMSNorm -> SwiGLU FFN -> residual, similar to LLaMA/GPT decoder blocks.
    """

    def __init__(self):
        """Initialize attention, normalization weights, and FFN weights."""
        self.attn = GQAAttention()
        self.norm1_weight = np.ones(d_model)
        self.norm2_weight = np.ones(d_model)
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W3 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b3 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x, mask, training=True, past_kv=None):
        """Run one decoder block and return updated hidden states plus cache.

        `past_kv` 只影响 attention；FFN 对每个 token 独立计算。
        `past_kv` only affects attention; the FFN is token-wise.
        """
        # Pre-norm 让残差路径更稳定，是现代 LLM 的常见设计。
        # Pre-norm stabilizes the residual path and is common in modern LLMs.
        x_norm1 = rms_norm(x, self.norm1_weight)
        attn_res, present_kv = self.attn.forward(x_norm1, mask, training, past_kv=past_kv)
        x = x + attn_res

        # FFN 提升每个 token 的非线性表达能力，不在 token 间通信。
        # The FFN increases per-token nonlinear capacity without cross-token communication.
        x_norm2 = rms_norm(x, self.norm2_weight)
        ff_hidden = swiglu(x_norm2, self.W1, self.b1, self.W3, self.b3)
        ff_hidden = dropout(ff_hidden, 0.1, training)
        ff_res = ff_hidden @ self.W2 + self.b2
        ff_res = dropout(ff_res, 0.1, training)
        x = x + ff_res
        return x, present_kv


class CoreSimLLM:
    """decoder-only language model used

    该类展示真实LLM的推理形状，但训练反传是简化版
    This class mirrors the inference shape of a real LLM, while backpropagation is simplified.
    """

    def __init__(self):
        """Initialize embeddings, decoder blocks, final norm/projection, and optimizers."""
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.decoder_blocks = [DecoderLayer() for _ in range(n_layers)]
        self.final_norm_weight = np.ones(d_model)
        self.final_proj = np.random.randn(d_model, vocab_size) * 0.1
        self.drop_emb = 0.1
        self.opt_pretrain_sft = AdamW(lr_pretrain, weight_decay)
        self.opt_ppo = AdamW(lr_ppo, weight_decay)
        self.opt_dpo = AdamW(lr_dpo, weight_decay)

    def forward(self, token_ids, training=True, return_attn=False, past_kv=None):
        """Run a full model forward pass.

        输出`probs`是每个位置预测下一个token的概率；`x`是最终hidden states；`present_kv` 可用于后续增量生成。
        `probs` are next-token probabilities per position, `x` contains
        final hidden states, and `present_kv` can be reused for incremental decoding.
        """
        token_ids = np.asarray(token_ids, dtype=int)
        seq_len = len(token_ids)

        # token id 先查 embedding 表，得到连续向量表示。
        # Token ids index the embedding table to obtain dense vectors.
        token_vec = self.token_embedding[token_ids]
        x = dropout(token_vec, self.drop_emb, training)

        attn_mats = []
        present_kv = []
        for layer_idx, block in enumerate(self.decoder_blocks):
            # 每层都有自己的 KV cache；past_len 决定当前 token 的绝对位置。
            # Each layer owns its KV cache; past_len determines the absolute position.
            layer_past = None if past_kv is None else past_kv[layer_idx]
            past_len = 0 if layer_past is None else layer_past[0].shape[1]
            mask = get_causal_mask(seq_len, past_len + seq_len, past_len)
            if return_attn:
                # 显式展开该分支，方便返回 attention matrix 用于可视化。
                # This branch is expanded so attention matrices can be returned for visualization.
                blk_out, attn_mat, layer_present = block.attn.forward(
                    rms_norm(x, block.norm1_weight),
                    mask,
                    training,
                    return_attn=True,
                    past_kv=layer_past,
                )
                attn_mats.append(attn_mat)
                x = x + blk_out
                x_norm2 = rms_norm(x, block.norm2_weight)
                ff_hidden = swiglu(x_norm2, block.W1, block.b1, block.W3, block.b3)
                ff_hidden = dropout(ff_hidden, 0.1, training)
                ff_res = ff_hidden @ block.W2 + block.b2
                ff_res = dropout(ff_res, 0.1, training)
                x = x + ff_res
            else:
                x, layer_present = block.forward(x, mask, training, past_kv=layer_past)
            present_kv.append(layer_present)

        # 最终 RMSNorm + output projection 产生 vocab logits，再 softmax 为概率。
        # Final RMSNorm plus output projection produces vocab logits, then softmax probabilities.
        x = rms_norm(x, self.final_norm_weight)
        logits = x @ self.final_proj
        probs = softmax(logits)
        if return_attn:
            return probs, x, attn_mats, present_kv
        return probs, x, present_kv

    def compute_ce_loss(self, probs, target_ids, loss_mask=None):
        """Compute masked cross-entropy loss.

        pretrain 会 mask 掉 pad；SFT 还会 mask 掉 prompt，只训练回答。
        Pretraining masks padding; SFT also masks prompt tokens and trains answers only.
        """
        target_ids = np.asarray(target_ids, dtype=int)
        mask = token_loss_mask(target_ids, loss_mask)
        rows = np.arange(len(target_ids))
        nll = -np.log(probs[rows, target_ids] + 1e-8) * mask
        return np.sum(nll) / max(1.0, np.sum(mask))

    def compute_ppl(self, probs, target_ids, loss_mask=None):
        """Compute perplexity from cross-entropy.

        PPL = exp(CE)，越低表示 next-token 预测越好。
        PPL = exp(CE); lower means better next-token prediction.
        """
        ce = self.compute_ce_loss(probs, target_ids, loss_mask)
        return np.exp(ce)

    def update_pretrain_sft(self, token_ids, target_ids, probs, x, lr, loss_mask=None):
        """Apply the simplified pretrain/SFT update.

        真实训练会通过 autograd 更新所有 Transformer 权重；这里为了可读性只手写更新 final projection 和 token embedding。
        Real training uses autograd to update all Transformer weights.
        For readability, this hand-written update changes only final projection and token embeddings.
        """
        token_ids = np.asarray(token_ids, dtype=int)
        target_ids = np.asarray(target_ids, dtype=int)
        mask = token_loss_mask(target_ids, loss_mask)
        if np.sum(mask) == 0:
            return

        grad = probs.copy()
        # softmax + CE 的 logits 梯度为 probs - onehot(target)。
        # For softmax + CE, the logits gradient is probs - onehot(target).
        grad[np.arange(len(target_ids)), target_ids] -= 1
        grad *= mask[:, None]
        grad /= max(1.0, np.sum(mask))

        self.opt_pretrain_sft.lr = lr
        # 输出层梯度来自 hidden states 与 logits 梯度的外积累加。
        # Output-layer gradient is hidden states multiplied by logits gradients.
        d_fp = np.dot(x.T, grad)
        d_emb = np.dot(grad, self.final_proj.T)
        self.final_proj = self.opt_pretrain_sft.update("fp", self.final_proj, d_fp)

        emb_grad = np.zeros_like(self.token_embedding)
        # 同一个 token 多次出现时用 add.at 正确累加梯度。
        # add.at correctly accumulates gradients for repeated token ids.
        np.add.at(emb_grad, token_ids, d_emb)
        self.token_embedding = self.opt_pretrain_sft.update("emb", self.token_embedding, emb_grad)

    def count_parameters(self):
        """Count all simulator parameters.

        用于展示模型规模；不包含优化器状态。
        Reports model size; optimizer state is not counted.
        """
        total = self.token_embedding.size + self.final_norm_weight.size + self.final_proj.size
        for block in self.decoder_blocks:
            params = [
                block.norm1_weight,
                block.norm2_weight,
                block.attn.Wq,
                block.attn.Wk,
                block.attn.Wv,
                block.attn.Wo,
                block.W1,
                block.W3,
                block.b1,
                block.b3,
                block.W2,
                block.b2,
            ]
            total += sum(p.size for p in params)
        return total

    def state_dict(self):
        """Return a serializable copy of model weights.

        仿照真实框架中的 `state_dict`，用于快照、比较和 checkpoint。
        Mirrors framework-style `state_dict` for snapshots, comparison, and checkpointing.
        """
        state = {
            "token_embedding": np.copy(self.token_embedding),
            "final_norm_weight": np.copy(self.final_norm_weight),
            "final_proj": np.copy(self.final_proj),
        }
        for i, block in enumerate(self.decoder_blocks):
            prefix = f"blocks.{i}."
            state[prefix + "norm1_weight"] = np.copy(block.norm1_weight)
            state[prefix + "norm2_weight"] = np.copy(block.norm2_weight)
            state[prefix + "attn.Wq"] = np.copy(block.attn.Wq)
            state[prefix + "attn.Wk"] = np.copy(block.attn.Wk)
            state[prefix + "attn.Wv"] = np.copy(block.attn.Wv)
            state[prefix + "attn.Wo"] = np.copy(block.attn.Wo)
            state[prefix + "W1"] = np.copy(block.W1)
            state[prefix + "W3"] = np.copy(block.W3)
            state[prefix + "b1"] = np.copy(block.b1)
            state[prefix + "b3"] = np.copy(block.b3)
            state[prefix + "W2"] = np.copy(block.W2)
            state[prefix + "b2"] = np.copy(block.b2)
        return state

    def load_state_dict(self, state):
        """Load weights from a state dictionary.

        只恢复权重，不恢复优化器动量；阶段快照对比只需要权重。
        Restores weights only, not optimizer moments. Stage comparison only needs weights.
        """
        self.token_embedding = np.copy(state["token_embedding"])
        self.final_norm_weight = np.copy(state["final_norm_weight"])
        self.final_proj = np.copy(state["final_proj"])
        for i, block in enumerate(self.decoder_blocks):
            prefix = f"blocks.{i}."
            block.norm1_weight = np.copy(state[prefix + "norm1_weight"])
            block.norm2_weight = np.copy(state[prefix + "norm2_weight"])
            block.attn.Wq = np.copy(state[prefix + "attn.Wq"])
            block.attn.Wk = np.copy(state[prefix + "attn.Wk"])
            block.attn.Wv = np.copy(state[prefix + "attn.Wv"])
            block.attn.Wo = np.copy(state[prefix + "attn.Wo"])
            block.W1 = np.copy(state[prefix + "W1"])
            block.W3 = np.copy(state[prefix + "W3"])
            block.b1 = np.copy(state[prefix + "b1"])
            block.b3 = np.copy(state[prefix + "b3"])
            block.W2 = np.copy(state[prefix + "W2"])
            block.b2 = np.copy(state[prefix + "b2"])


def copy_policy(target, source):
    """Copy model weights from source policy to target policy.

    PPO 需要 old policy 和 new policy；DPO 需要 ref model。
    PPO needs an old and new policy; DPO needs a reference model.
    """
    target.load_state_dict(source.state_dict())


def save_checkpoint(model, path="coresim_checkpoint.npz"):
    """Save model weights as a compressed NumPy checkpoint.

    这是轻量 checkpoint，用于教学；真实工程会保存更多元数据。
    This is a lightweight teaching checkpoint; production systems save more metadata.
    """
    np.savez_compressed(path, **model.state_dict())


def load_checkpoint(path):
    """Load a model from a compressed NumPy checkpoint.

    返回一个新的 `CoreSimLLM` 实例，并填入 checkpoint 权重。
    Returns a new `CoreSimLLM` instance populated with checkpoint weights.
    """
    model = CoreSimLLM()
    with np.load(path) as data:
        model.load_state_dict(data)
    return model
