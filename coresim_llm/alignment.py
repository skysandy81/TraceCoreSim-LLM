"""Alignment algorithms: simplified PPO and DPO.

RLHF/PPO和DPO的核心数据流：旧策略、新策略、奖励模型、参考模型、response log-prob、KL惩罚和偏好margin。
The core RLHF/PPO and DPO data flow: old/new policy, reward model, reference model,
response log-probs, KL penalty, and preference margin.
"""

import numpy as np

from .config import (
    EPS,
    PPO_CLIP_HIGH,
    PPO_CLIP_LOW,
    dpo_beta,
    epoch_dpo,
    kl_coeff,
    lr_dpo,
    lr_ppo,
    seq_max_len,
)
from .data import (
    make_chat_example,
    pair_rm_data,
    sequence_logprob,
    token_loss_mask,
)
from .generation import generate_answer
from .model import CoreSimLLM, copy_policy
from .ops import softmax

# old_model：旧策略，用来计算旧概率 π_old。
# new_model：当前要更新的新策略 π_new。
# rm：reward model，提供 value baseline。
# input_ids：prompt + response 的 token 输入。
# target_ids：每个位置要预测的下一个 token。
# loss_mask：只让 response token 参与 PPO loss。
# reward：当前生成回答得到的奖励分数。
def ppo_update(old_model, new_model, rm, input_ids, target_ids, loss_mask, reward):
    """Apply one simplified PPO-style policy update.
    真实 PPO 会采样 rollout、估计 advantage、分 mini-batch 多轮优化。这里保留最核心的 ratio clipping + KL penalty，用于展示阶段效果。
    Real PPO samples rollouts, estimates advantages, and optimizes over mini-batches. This keeps the core ratio clipping plus KL penalty for stage demos.
    """
    input_ids = np.asarray(input_ids, dtype=int)
    target_ids = np.asarray(target_ids, dtype=int)
    mask = token_loss_mask(target_ids, loss_mask)
    old_prob, _, _ = old_model.forward(input_ids, training=False)
    new_prob, new_x, _ = new_model.forward(input_ids, training=False)
    seq_len = len(input_ids)

    # 只在 response token 上计算新旧策略概率比。
    # Policy ratio is computed only on response tokens.
    logp_old = sequence_logprob(old_prob, target_ids, mask, normalize=True)
    logp_new = sequence_logprob(new_prob, target_ids, mask, normalize=True)
    ratio = np.exp(np.clip(logp_new - logp_old, -20, 20))
    v_curr = rm.get_value(input_ids)
    advantage = reward - v_curr

    # PPO clip 防止新策略相对旧策略一步变化过大。
    # PPO clipping prevents the new policy from moving too far from the old policy.
    clip1 = ratio * advantage
    clip2 = np.clip(ratio, PPO_CLIP_LOW, PPO_CLIP_HIGH) * advantage
    ppo_loss = -np.minimum(clip1, clip2)

    rows = np.arange(seq_len)
    onehot = np.zeros_like(new_prob)
    onehot[rows, target_ids] = 1
    clipped = (advantage >= 0 and ratio > PPO_CLIP_HIGH) or (
        advantage < 0 and ratio < PPO_CLIP_LOW
    )
    policy_scale = 0.0 if clipped else -advantage * ratio
    denom = max(1.0, np.sum(mask))
    # 手写 softmax policy gradient；只作为流程近似，不是完整 autograd。
    # Hand-written softmax policy gradient; a workflow approximation, not full autograd.
    grad = policy_scale * (onehot - new_prob) * mask[:, None] / denom

    # KL 项让 new policy 不要偏离 old policy 太多。
    # The KL term discourages the new policy from drifting too far from the old policy.
    kl_token = np.sum(old_prob * np.log((old_prob + EPS) / (new_prob + EPS)), axis=-1)
    kl_mask = mask if np.sum(mask) > 0 else np.ones_like(mask)
    kl = np.sum(kl_token * kl_mask) / max(1.0, np.sum(kl_mask))
    total_loss = ppo_loss + kl_coeff * kl
    grad += kl_coeff * (new_prob - old_prob) * kl_mask[:, None] / max(1.0, np.sum(kl_mask))

    # 与主训练一致，仅更新 output projection 和 token embedding。
    # As in the main training loop, only output projection and token embeddings are updated.
    d_fp = np.dot(new_x.T, grad)
    d_emb = np.dot(grad, new_model.final_proj.T)
    emb_grad = np.zeros_like(new_model.token_embedding)
    np.add.at(emb_grad, input_ids, d_emb)
    new_model.opt_ppo.lr = lr_ppo
    new_model.final_proj = new_model.opt_ppo.update("ppo_fp", new_model.final_proj, d_fp)
    new_model.token_embedding = new_model.opt_ppo.update("ppo_emb", new_model.token_embedding, emb_grad)
    return total_loss, kl, reward


def dpo_loss(ref_model, train_model, pos_example, neg_example, beta=dpo_beta):
    """Compute the Direct Preference Optimization loss.

    DPO 比较 chosen/rejected response 相对 reference model 的 log-ratio。如果 chosen 的相对概率高于 rejected，loss 下降。
    DPO compares chosen/rejected response log-ratios against a reference model. The loss decreases when chosen is preferred over rejected.
    """
    pos_inp, pos_tgt, pos_mask = pos_example
    neg_inp, neg_tgt, neg_mask = neg_example

    # 用 normalize=True 把不同长度 response 的 log-prob 转成平均值。
    # normalize=True converts sequence log-probs into average token log-probs.
    pos_prob, pos_x, _ = train_model.forward(pos_inp, training=False)
    pos_ref_prob, _, _ = ref_model.forward(pos_inp, training=False)
    log_pos = sequence_logprob(pos_prob, pos_tgt, pos_mask, normalize=True)
    log_pos_ref = sequence_logprob(pos_ref_prob, pos_tgt, pos_mask, normalize=True)

    neg_prob, neg_x, _ = train_model.forward(neg_inp, training=False)
    neg_ref_prob, _, _ = ref_model.forward(neg_inp, training=False)
    log_neg = sequence_logprob(neg_prob, neg_tgt, neg_mask, normalize=True)
    log_neg_ref = sequence_logprob(neg_ref_prob, neg_tgt, neg_mask, normalize=True)

    log_ratio_pos = log_pos - log_pos_ref
    log_ratio_neg = log_neg - log_neg_ref
    loss = -np.log(softmax(np.array([beta * log_ratio_pos, beta * log_ratio_neg]))[0])
    return loss, log_ratio_pos, log_ratio_neg, pos_prob, neg_prob, pos_x, neg_x


def train_dpo(new_model):
    """Run the 小型 DPO alignment stage.

    先复制当前策略作为 frozen reference model，然后对偏好样本做 DPO 更新。
    First clone the current policy as a frozen reference model, then update on preference pairs with DPO.
    """
    ref_model = CoreSimLLM()
    copy_policy(ref_model, new_model)
    dpo_rec = []
    for epoch in range(epoch_dpo):
        total_loss = 0.0
        for pair in pair_rm_data:
            # chosen/rejected 共享同一个 prompt，只在 answer token 上计算 loss。
            # Chosen/rejected share the same prompt; loss is computed on answer tokens only.
            pos_example = make_chat_example(pair["prompt"], pair["good"], max_len=seq_max_len)
            neg_example = make_chat_example(pair["prompt"], pair["bad"], max_len=seq_max_len)
            pos_inp, pos_tgt, pos_mask = pos_example
            neg_inp, neg_tgt, neg_mask = neg_example
            loss, log_ratio_pos, log_ratio_neg, pos_prob, neg_prob, pos_x, neg_x = dpo_loss(
                ref_model, new_model, pos_example, neg_example
            )
            total_loss += loss

            # margin 越大表示模型越偏向 chosen；梯度推动 margin 继续增大。
            # Larger margin means the model prefers chosen; the gradient pushes it higher.
            margin = log_ratio_pos - log_ratio_neg
            grad_margin = -dpo_beta / (1 + np.exp(dpo_beta * margin))

            pos_onehot = np.zeros_like(pos_prob)
            neg_onehot = np.zeros_like(neg_prob)
            pos_onehot[np.arange(len(pos_tgt)), pos_tgt] = 1
            neg_onehot[np.arange(len(neg_tgt)), neg_tgt] = 1
            pos_valid = token_loss_mask(pos_tgt, pos_mask)
            neg_valid = token_loss_mask(neg_tgt, neg_mask)
            # chosen 梯度提高目标 token 概率，rejected 梯度降低其概率。
            # Chosen gradients raise target-token probability; rejected gradients lower it.
            pos_grad = (
                grad_margin
                * (pos_onehot - pos_prob)
                * pos_valid[:, None]
                / max(1.0, np.sum(pos_valid))
            )
            neg_grad = (
                -grad_margin
                * (neg_onehot - neg_prob)
                * neg_valid[:, None]
                / max(1.0, np.sum(neg_valid))
            )

            # 仍然只更新输出层和 embedding，保持演示简单可读。
            # Still updates only output projection and embeddings for readability.
            d_fp = np.dot(pos_x.T, pos_grad) + np.dot(neg_x.T, neg_grad)
            d_emb_pos = np.dot(pos_grad, new_model.final_proj.T)
            d_emb_neg = np.dot(neg_grad, new_model.final_proj.T)
            emb_grad = np.zeros_like(new_model.token_embedding)
            np.add.at(emb_grad, pos_inp, d_emb_pos)
            np.add.at(emb_grad, neg_inp, d_emb_neg)

            new_model.opt_dpo.lr = lr_dpo
            new_model.final_proj = new_model.opt_dpo.update("dpo_fp", new_model.final_proj, d_fp)
            new_model.token_embedding = new_model.opt_dpo.update("dpo_emb", new_model.token_embedding, emb_grad)

        avg_loss = total_loss / len(pair_rm_data)
        dpo_rec.append(avg_loss)
        if epoch % 60 == 0:
            test_out = generate_answer(new_model, "Is coding fun?", 16)
            print(f"DPO epoch {epoch:3d} loss: {avg_loss:.3f} answer: {test_out}")
    return dpo_rec
