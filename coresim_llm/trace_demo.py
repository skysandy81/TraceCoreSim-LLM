"""
单独构造d_model=8的确定性前向传播，用真实Transformer的关键步骤展示每一步张量数值
Independently building a deterministic d_model=8 forward pass and
shows the tensor values for real Transformer-style steps
"""

import numpy as np
import json
from pathlib import Path
from .ops import get_causal_mask, rms_norm, rope_embedding, silu

# trace demo: TRACE_D_MODEL = 8; training simulation = 32
TRACE_D_MODEL = 8
TRACE_N_HEADS = 2
TRACE_N_KV_HEADS = 1
TRACE_HEAD_DIM = TRACE_D_MODEL // TRACE_N_HEADS
TRACE_D_FF = 8

TRACE_TOKENS = ["<bos>", "I", "like"]
TRACE_VOCAB = ["<bos>", "I", "like", "coding", ".", "fun"]
TRACE_PARAM_STYLE = "handcrafted small values"


def _build_params():
    """Building all parameters needed by one tiny decoder block.
    参数命名与主模型保持一致
    Parameter names mirror the main model
    """
    # 对应到 TRACE_VOCAB 中的每一个word，为便于演示，此处为设定值，实际为随机值
    token_embedding = np.array(
        [
            [0.8, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.0, 0.0, 0.1, 0.0, 0.1],
            [0.0, 0.2, 0.9, 0.1, 0.0, 0.0, 0.2, 0.1],
            [0.0, 0.1, 0.2, 0.9, 0.1, 0.0, 0.0, 0.2],
            [0.1, 0.0, 0.0, 0.2, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.1, 0.8, 0.2, 0.0],
        ],
        dtype=float,
    )
    #单位矩阵是指主对角线上的元素为1，其余元素为0的方阵。维度[d_model, n_heads * head_dim]
    Wq = np.eye(TRACE_D_MODEL) * 0.6
    # 维度[d_model, n_kv_heads * head_dim]
    Wk = np.array(
        [
            [0.6, 0.0, 0.1, 0.0],
            [0.0, 0.6, 0.0, 0.1],
            [0.1, 0.0, 0.6, 0.0],
            [0.0, 0.1, 0.0, 0.6],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ],
        dtype=float,
    )
    # 维度[d_model, n_kv_heads * head_dim]
    Wv = np.array(
        [
            [0.5, 0.0, 0.0, 0.1],
            [0.0, 0.5, 0.1, 0.0],
            [0.1, 0.0, 0.5, 0.0],
            [0.0, 0.1, 0.0, 0.5],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ],
        dtype=float,
    )
    #维度[d_model, vocab_size]
    Wout = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.5, 0.0, 0.2],
            [0.0, 0.0, 0.1, 0.7, 0.2, 0.3],
            [0.2, 0.0, 0.0, 0.0, 0.6, 0.0],
            [0.0, 0.1, 0.0, 0.0, 0.1, 0.7],
            [0.0, 0.0, 0.2, 0.4, 0.0, 0.2],
            [0.0, 0.0, 0.1, 0.3, 0.0, 0.1],
        ],
        dtype=float,
    )
    return {
        "token_embedding": token_embedding,
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv,
        "Wo": np.eye(TRACE_D_MODEL) * 0.35,
        "W1": np.eye(TRACE_D_MODEL, TRACE_D_FF) * 0.55,
        "W3": np.eye(TRACE_D_MODEL, TRACE_D_FF) * 0.45,
        "W2": np.eye(TRACE_D_FF, TRACE_D_MODEL) * 0.30,
        "Wout": Wout,
        "b1": np.zeros(TRACE_D_FF),
        "b3": np.zeros(TRACE_D_FF),
        "b2": np.zeros(TRACE_D_MODEL),
    }


def _round(value, decimals=3):
    """Round numeric arrays before serialization.
    HTML 和文本演示只需要可读数值，四舍五入不会改变计算流程说明。
    The demo needs readable values.
    """
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:  # 如果这个numpy数组是 字符串数组、字节串数组、object对象数组
        return arr.tolist()
    return np.round(arr.astype(float), decimals).tolist()


def _value(name, value, note=None, decimals=3):
    """
    每个 value block 都包含名称、shape、数值和可选说明，供 CLI 和 HTML 共用。
    Each value block contains name, shape, value, and optional note, and is shared by the CLI and HTML renderers.
    """
    arr = np.asarray(value)
    payload = {
        "name": name,
        "shape": list(arr.shape),
        "value": _round(arr, decimals=decimals),
    }
    if note:
        payload["note"] = note
    return payload


def _text_value(name, value, note=None):
    """
    tokens、top candidates 这类数据不是普通矩阵，单独包装便于渲染。
    Tokens and top candidates are not plain matrices, so they get a separate wrapper for rendering.
    """
    payload = {"name": name, "shape": [], "value": value}
    if note:
        payload["note"] = note
    return payload


def _calc(title, formula, values, detail=None):
    """
    一个大步骤可能包含多个小计算。这个结构让 CLI 和 HTML 可以统一渲染这些小计算。
    A large step can contain several small calculations. This structure lets the CLI and HTML render those calculations consistently.
    """
    payload = {
        "title": title,
        "formula": formula,
        "values": values,
    }
    if detail:
        payload["detail"] = detail
    return payload


def _step(title, formula, values, detail, calculation=None):
    """
    formula 给出数学形式，values给出实际数值，detail解释它对应真实 LLM 的哪一步。
    formula gives the math, values give actual numbers, and detail explains which real-LLM operation this step represents.
    """
    return {
        "title": title,
        "formula": formula,
        "values": values,
        "detail": detail,
        "calculation": calculation or [],
    }


def _softmax_trace(x, input_name="x"):
    """
    softmax 的稳定实现为：y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))。
    减去max不改变概率，只避免exp溢出；attention score和logits都用这一步。
    Stable softmax is: y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)).
    Subtracting max preserves probabilities while preventing overflow. Both attention scores and logits use this operation.
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x, axis=-1, keepdims=True)
    shifted = x - max_x
    exp_shifted = np.exp(shifted)
    exp_sum = np.sum(exp_shifted, axis=-1, keepdims=True)
    output = exp_shifted / exp_sum
    values = [
        _value(input_name, x, "Softmax is applied row-wise along the last axis."),
        _value("max(x)", max_x, "One max value is kept for each row/vector."),
        _value("x - max(x)", shifted, "This improves numerical stability."),
        _value("exp(x - max(x))", exp_shifted),
        _value("sum(exp(...))", exp_sum),
        _value("softmax output", output, decimals=4),
    ]
    return output, values


def build_trace():
    """
    函数返回纯Python/JSON友好的结构，不依赖UI。这样同一份trace可以同时用于`python main.py --trace`的文本输出和`--trace-html`的可视化页面。
    The function returns a plain Python/JSON-friendly structure with no UI dependency.
    The same trace powers both `python main.py --trace` text output and the `--trace-html` visualization.
    """
    params = _build_params()
    token_to_id = {tok: idx for idx, tok in enumerate(TRACE_VOCAB)}
    token_ids = np.array([token_to_id[tok] for tok in TRACE_TOKENS], dtype=int)
    seq_len = len(token_ids)

    # GQA 中一个 K/V head 会被重复给多个 Q heads 使用。
    # In GQA, one K/V head is repeated to serve multiple Q heads.
    repeat_times = TRACE_N_HEADS // TRACE_N_KV_HEADS

    steps = []
    steps.append(
        _step(
            "Tokenize",
            "tokens -> token_ids",
            [
                _text_value("tokens", TRACE_TOKENS),
                _value("token_ids", token_ids, "These integer ids index the embedding table."),
            ],
            "The demo uses a compact token vocabulary so every following tensor stays readable.",
            calculation=[
                _calc(
                    "Vocabulary lookup",
                    "token_id = vocab_index[token]",
                    [
                        _text_value(
                            "token id mapping",
                            [f"{tok} -> {idx}" for tok, idx in token_to_id.items()],
                        ),
                        _text_value(
                            "lookup path",
                            [f"{tok} -> {token_to_id[tok]}" for tok in TRACE_TOKENS],
                        ),
                    ],
                    "Tokenization converts text symbols into integer indices before any neural math starts.",
                )
            ],
        )
    )

    # embedding lookup 是离散 token id 到连续向量空间的入口。
    # Embedding lookup maps discrete token ids into continuous vectors.
    X = params["token_embedding"][token_ids]
    steps.append(
        _step(
            "Embedding lookup",
            "X = E[token_ids]",
            [
                _value("embedding table E", params["token_embedding"]),
                _value("input embeddings X", X, "Shape is sequence length by d_model."),
            ],
            "Each token id selects one 8-dimensional row from the embedding table.",
            calculation=[
                _calc(
                    "Gather rows from E",
                    "X[t] = E[token_ids[t]]",
                    [
                        _text_value(
                            "selected rows",
                            [f"X[{i}] = E[{tok_id}] for token {TRACE_TOKENS[i]}" for i, tok_id in enumerate(token_ids)],
                        ),
                        _value("X", X, "This matrix is the first hidden-state matrix used by the decoder block."),
                    ],
                )
            ],
        )
    )

    # pre-norm decoder block 会先归一化再进入 attention。
    # A pre-norm decoder block normalizes before attention.
    x_norm1 = rms_norm(X)
    x_square1 = X**2
    mean_square1 = np.mean(x_square1, axis=-1, keepdims=True)
    denom1 = np.sqrt(mean_square1 + 1e-5)
    steps.append(
        _step(
            "Pre-attention RMSNorm",
            "x_norm = X / sqrt(mean(X^2) + eps)",
            [
                _value("input X", X),
                _value("RMS denominator", denom1),
                _value("x_norm", x_norm1),
            ],
            "RMSNorm rescales each token vector while preserving its direction.",
            calculation=[
                _calc(
                    "Square every feature",
                    "X^2",
                    [_value("X^2", x_square1)],
                ),
                _calc(
                    "Average squared features per token",
                    "mean_square = mean(X^2, axis=-1)",
                    [_value("mean_square", mean_square1)],
                ),
                _calc(
                    "Divide by the RMS denominator",
                    "x_norm = X / sqrt(mean_square + 1e-5)",
                    [
                        _value("sqrt(mean_square + 1e-5)", denom1),
                        _value("x_norm", x_norm1),
                    ],
                ),
            ],
        )
    )

    # Q/K/V 是 attention 的三组线性投影；Q 决定“我要找什么”，K 决定“我是什么”， V 决定“我携带什么信息”。
    # Q/K/V are the three attention projections: Q asks what to look for,
    # K describes what each token offers, and V carries the information to mix.
    Q_raw = x_norm1 @ params["Wq"]
    K_raw = x_norm1 @ params["Wk"]
    V_raw = x_norm1 @ params["Wv"]
    q00_terms = x_norm1[0] * params["Wq"][:, 0]
    Q = Q_raw.reshape(seq_len, TRACE_N_HEADS, TRACE_HEAD_DIM).swapaxes(0, 1)
    K = K_raw.reshape(seq_len, TRACE_N_KV_HEADS, TRACE_HEAD_DIM).swapaxes(0, 1)
    V = V_raw.reshape(seq_len, TRACE_N_KV_HEADS, TRACE_HEAD_DIM).swapaxes(0, 1)
    steps.append(
        _step(
            "Q/K/V projections",
            "Q = x_norm Wq, K = x_norm Wk, V = x_norm Wv",
            [
                _value("Q raw", Q_raw),
                _value("K raw", K_raw),
                _value("V raw", V_raw),
                _value("Q reshaped", Q, "Two query heads, each with four dimensions."),
                _value("K reshaped", K, "One shared KV head for grouped-query attention."),
                _value("V reshaped", V),
            ],
            "GQA keeps two query heads but only one K/V head, then repeats K/V for attention.",
            calculation=[
                _calc(
                    "Projection inputs and weights",
                    "Q_raw = x_norm Wq; K_raw = x_norm Wk; V_raw = x_norm Wv",
                    [
                        _value("input x_norm", x_norm1),
                        _value("Wq", params["Wq"]),
                        _value("Wk", params["Wk"]),
                        _value("Wv", params["Wv"]),
                    ],
                ),
                _calc(
                    "One dot-product example",
                    "Q_raw[0,0] = sum_i x_norm[0,i] * Wq[i,0]",
                    [
                        _value("elementwise products", q00_terms),
                        _value("Q_raw[0,0]", np.array([np.sum(q00_terms)])),
                    ],
                    "The same dot-product pattern is repeated for every row and output column.",
                ),
                _calc(
                    "Reshape into attention heads",
                    "[seq, d_model] -> [n_heads, seq, head_dim]",
                    [
                        _value("Q_raw", Q_raw),
                        _value("Q reshaped", Q),
                        _value("K reshaped", K),
                        _value("V reshaped", V),
                    ],
                ),
            ],
        )
    )

    # RoPE 把位置信息注入 Q/K，因此 attention score 同时依赖内容和位置。
    # RoPE injects position into Q/K, so attention scores depend on both content and position.
    Q_rot = Q.copy()
    K_rot = K.copy()
    theta = 10000 ** (-(np.arange(0, TRACE_HEAD_DIM // 2) / (TRACE_HEAD_DIM // 2)))
    positions = np.arange(seq_len)[:, None]
    freqs = positions * theta
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    for h in range(TRACE_N_HEADS):
        Q_rot[h] = rope_embedding(Q_rot[h])
    for h in range(TRACE_N_KV_HEADS):
        K_rot[h] = rope_embedding(K_rot[h])
    steps.append(
        _step(
            "RoPE position rotation",
            "RoPE rotates every even/odd feature pair by token position",
            [
                _value("Q after RoPE", Q_rot),
                _value("K after RoPE", K_rot),
            ],
            "The first token is barely rotated; later tokens receive position-dependent rotations.",
            calculation=[
                _calc(
                    "Build RoPE angles",
                    "freq[pos, pair] = pos * 10000^(-pair/(head_dim/2))",
                    [
                        _value("positions", np.arange(seq_len)),
                        _value("theta", theta),
                        _value("freqs", freqs),
                        _value("cos(freqs)", cos),
                        _value("sin(freqs)", sin),
                    ],
                ),
                _calc(
                    "Rotate even/odd feature pairs",
                    "even' = even*cos - odd*sin; odd' = even*sin + odd*cos",
                    [
                        _value("Q before RoPE", Q),
                        _value("Q after RoPE", Q_rot),
                        _value("K before RoPE", K),
                        _value("K after RoPE", K_rot),
                    ],
                ),
            ],
        )
    )

    # 因果 mask 阻止当前位置看到未来 token，是自回归生成的核心约束。
    # The causal mask prevents each position from seeing future tokens, which is the key constraint for autoregressive generation.
    K_attn = np.repeat(K_rot, repeat_times, axis=0)
    V_attn = np.repeat(V, repeat_times, axis=0)
    causal_mask = get_causal_mask(seq_len)
    scores_no_mask = Q_rot @ K_attn.swapaxes(-1, -2) / np.sqrt(TRACE_HEAD_DIM)
    scores = scores_no_mask + np.expand_dims(causal_mask, axis=0)
    weights, attention_softmax_values = _softmax_trace(scores, "masked attention scores")
    steps.append(
        _step(
            "Causal attention",
            "softmax((Q K^T / sqrt(head_dim)) + causal_mask)",
            [
                _value("causal mask", causal_mask, "Large negative values block future tokens."),
                _value("scores before mask", scores_no_mask),
                _value("scores after mask", scores),
                _value("attention weights", weights),
            ],
            "Each token can attend to itself and previous tokens, but not future tokens.",
            calculation=[
                _calc(
                    "Repeat shared K/V heads for GQA",
                    "K_attn = repeat(K, n_heads / n_kv_heads); V_attn = repeat(V, ...)",
                    [
                        _value("K after RoPE", K_rot),
                        _value("K_attn", K_attn),
                        _value("V_attn", V_attn),
                    ],
                    "This keeps two query heads but shares one K/V head, like grouped-query attention.",
                ),
                _calc(
                    "Scaled dot-product scores",
                    "scores_before_mask = Q_rot K_attn^T / sqrt(head_dim)",
                    [
                        _value("Q_rot", Q_rot),
                        _value("K_attn^T", K_attn.swapaxes(-1, -2)),
                        _value("sqrt(head_dim)", np.array([np.sqrt(TRACE_HEAD_DIM)])),
                        _value("scores before mask", scores_no_mask),
                    ],
                ),
                _calc(
                    "Add the causal mask",
                    "masked_scores = scores_before_mask + causal_mask",
                    [
                        _value("causal mask", causal_mask, "Future columns contain a very large negative number."),
                        _value("masked attention scores", scores),
                    ],
                ),
                _calc(
                    "Stable softmax, row by row",
                    "softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))",
                    attention_softmax_values,
                    "Rows with masked future tokens get exp(very negative) = 0, so future attention becomes 0.",
                ),
            ],
        )
    )

    # attention 输出会通过 residual 加回输入，让原始 token 表示继续保留。
    # Attention output is added through a residual connection so the original token representation remains available.
    context = weights @ V_attn
    attn_concat = context.swapaxes(0, 1).reshape(seq_len, TRACE_D_MODEL)
    attn_out = attn_concat @ params["Wo"]
    x_after_attn = X + attn_out
    steps.append(
        _step(
            "Attention output and residual",
            "attention_out = concat(weights V) Wo; x = X + attention_out",
            [
                _value("context per head", context),
                _value("concatenated context", attn_concat),
                _value("attention output", attn_out),
                _value("residual after attention", x_after_attn),
            ],
            "The residual path carries the original token signal forward.",
            calculation=[
                _calc(
                    "Mix value vectors with attention weights",
                    "context = attention_weights V_attn",
                    [
                        _value("attention weights", weights),
                        _value("V_attn", V_attn),
                        _value("context per head", context),
                    ],
                ),
                _calc(
                    "Concatenate heads and project back to d_model",
                    "attention_out = concat(context_heads) Wo",
                    [
                        _value("concatenated context", attn_concat),
                        _value("Wo", params["Wo"]),
                        _value("attention output", attn_out),
                    ],
                ),
                _calc(
                    "Residual addition",
                    "x_after_attn = X + attention_out",
                    [
                        _value("original X", X),
                        _value("attention output", attn_out),
                        _value("x_after_attn", x_after_attn),
                    ],
                    "The plus sign is element-wise and keeps the original embedding signal available.",
                ),
            ],
        )
    )

    # SwiGLU FFN 是现代 LLM 常见的门控前馈层，逐 token 独立处理特征。
    # SwiGLU FFN is a common gated feed-forward layer in modern LLMs and processes each token independently.
    x_norm2 = rms_norm(x_after_attn)
    x_square2 = x_after_attn**2
    mean_square2 = np.mean(x_square2, axis=-1, keepdims=True)
    denom2 = np.sqrt(mean_square2 + 1e-5)
    gate_pre = x_norm2 @ params["W1"] + params["b1"]
    up = x_norm2 @ params["W3"] + params["b3"]
    sigmoid_gate = 1 / (1 + np.exp(-gate_pre))
    gate = silu(gate_pre)
    hidden = gate * up
    ffn_out = hidden @ params["W2"] + params["b2"]
    x_after_ffn = x_after_attn + ffn_out
    steps.append(
        _step(
            "SwiGLU feed-forward block",
            "ffn = (SiLU(x_norm W1) * (x_norm W3)) W2; x = x + ffn",
            [
                _value("x_norm2", x_norm2),
                _value("gate pre-activation", gate_pre),
                _value("SiLU gate", gate),
                _value("up projection", up),
                _value("gated hidden", hidden),
                _value("ffn output", ffn_out),
                _value("residual after FFN", x_after_ffn),
            ],
            "SwiGLU gates the hidden features before projecting them back to d_model.",
            calculation=[
                _calc(
                    "Normalize the attention residual",
                    "x_norm2 = x_after_attn / sqrt(mean(x_after_attn^2) + eps)",
                    [
                        _value("x_after_attn", x_after_attn),
                        _value("x_after_attn^2", x_square2),
                        _value("mean square", mean_square2),
                        _value("RMS denominator", denom2),
                        _value("x_norm2", x_norm2),
                    ],
                ),
                _calc(
                    "Compute the two FFN projections",
                    "gate_pre = x_norm2 W1 + b1; up = x_norm2 W3 + b3",
                    [
                        _value("W1", params["W1"]),
                        _value("W3", params["W3"]),
                        _value("gate_pre", gate_pre),
                        _value("up", up),
                    ],
                ),
                _calc(
                    "Apply SiLU gate and multiply",
                    "SiLU(z) = z * sigmoid(z); hidden = SiLU(gate_pre) * up",
                    [
                        _value("sigmoid(gate_pre)", sigmoid_gate),
                        _value("SiLU gate", gate),
                        _value("up", up),
                        _value("gated hidden", hidden),
                    ],
                ),
                _calc(
                    "Project FFN output and add residual",
                    "x_after_ffn = x_after_attn + hidden W2 + b2",
                    [
                        _value("W2", params["W2"]),
                        _value("ffn output", ffn_out),
                        _value("x_after_ffn", x_after_ffn),
                    ],
                ),
            ],
        )
    )

    # 最后一行 probabilities 对应“当前上下文之后下一个 token”的分布。
    # The last probability row is the next-token distribution after the current context.
    final_square = x_after_ffn**2
    final_mean_square = np.mean(final_square, axis=-1, keepdims=True)
    final_denom = np.sqrt(final_mean_square + 1e-5)
    final_hidden = rms_norm(x_after_ffn)
    logits = final_hidden @ params["Wout"]
    probs, final_softmax_values = _softmax_trace(logits, "logits")
    last_probs = probs[-1]
    top_ids = np.argsort(last_probs)[::-1][:5]
    top_tokens = [
        {
            "token": TRACE_VOCAB[int(idx)],
            "prob": float(np.round(last_probs[int(idx)], 4)),
            "logit": float(np.round(logits[-1, int(idx)], 4)),
        }
        for idx in top_ids
    ]
    steps.append(
        _step(
            "Logits and next-token probabilities",
            "logits = final_hidden Wout; probs = softmax(logits)",
            [
                _value("final hidden", final_hidden),
                _value("logits", logits),
                _value("probabilities", probs),
                _text_value("top next-token candidates", top_tokens),
            ],
            "The next token is selected from the last row of probabilities.",
            calculation=[
                _calc(
                    "Final RMSNorm",
                    "final_hidden = x_after_ffn / sqrt(mean(x_after_ffn^2) + eps)",
                    [
                        _value("x_after_ffn", x_after_ffn),
                        _value("x_after_ffn^2", final_square),
                        _value("mean square", final_mean_square),
                        _value("RMS denominator", final_denom),
                        _value("final hidden", final_hidden),
                    ],
                ),
                _calc(
                    "Project hidden states to vocabulary logits",
                    "logits = final_hidden Wout",
                    [
                        _value("final hidden", final_hidden),
                        _value("Wout", params["Wout"]),
                        _value("logits", logits),
                    ],
                ),
                _calc(
                    "Stable softmax over vocabulary",
                    "prob(token_i) = exp(logit_i - max(logits)) / sum_j exp(logit_j - max(logits))",
                    final_softmax_values,
                    "For generation, the model uses the last row because it predicts the token after the full prompt.",
                ),
            ],
        )
    )

    return {
        "meta": {
            "parameter_style": TRACE_PARAM_STYLE,
            "d_model": TRACE_D_MODEL,
            "n_heads": TRACE_N_HEADS,
            "n_kv_heads": TRACE_N_KV_HEADS,
            "head_dim": TRACE_HEAD_DIM,
            "d_ff": TRACE_D_FF,
            "tokens": TRACE_TOKENS,
            "vocab": TRACE_VOCAB,
        },
        "steps": steps,
    }


def _format_matrix(value):
    """Format a trace value for terminal output.
    文本模式需要把矩阵、token 列表和候选 token 分别排版成可扫读的形式。
    Text mode formats matrices, token lists, and next-token candidates in forms that are easy to scan.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        rows = []
        for item in value:
            rows.append(
                f"    {item['token']:<8} prob={item['prob']:.4f} logit={item['logit']:.4f}"
            )
        return "\n".join(rows)
    arr = np.asarray(value, dtype=object)
    return np.array2string(arr, precision=3, suppress_small=True)


def format_trace_text(trace=None):
    """Render the trace as plain text.
    `python main.py --trace` 的输出路径，适合在终端直接查看每一步数值。
    This is the output path for `python main.py --trace`, suitable for
    inspecting every numeric step directly in the terminal.
    """
    trace = build_trace() if trace is None else trace
    meta = trace["meta"]
    lines = [
        "CoreSimLLM trace demo",
        (
            f"d_model={meta['d_model']}, n_heads={meta['n_heads']}, "
            f"n_kv_heads={meta['n_kv_heads']}, head_dim={meta['head_dim']}, d_ff={meta['d_ff']}"
        ),
        f"parameter_style={meta['parameter_style']}",
        f"tokens={meta['tokens']}",
        "",
    ]
    for idx, step in enumerate(trace["steps"], start=1):
        lines.append(f"{idx}. {step['title']}")
        lines.append(f"   formula: {step['formula']}")
        lines.append(f"   note: {step['detail']}")
        if step.get("calculation"):
            lines.append("   calculation:")
            for calc_idx, calc in enumerate(step["calculation"], start=1):
                lines.append(f"     {calc_idx}) {calc['title']}")
                lines.append(f"        formula: {calc['formula']}")
                if calc.get("detail"):
                    lines.append(f"        note: {calc['detail']}")
                for value in calc["values"]:
                    shape = f" shape={value['shape']}" if value["shape"] else ""
                    lines.append(f"        - {value['name']}{shape}")
                    if value.get("note"):
                        lines.append(f"          {value['note']}")
                    matrix = _format_matrix(value["value"])
                    for line in matrix.splitlines():
                        lines.append(f"          {line}")
        for value in step["values"]:
            shape = f" shape={value['shape']}" if value["shape"] else ""
            lines.append(f"   - {value['name']}{shape}")
            if value.get("note"):
                lines.append(f"     {value['note']}")
            matrix = _format_matrix(value["value"])
            for line in matrix.splitlines():
                lines.append(f"     {line}")
        lines.append("")
    return "\n".join(lines)


def _render_html(trace):
    """Render an interactive single-file HTML trace viewer.
    HTML 内嵌 CSS、数据 JSON 和少量 JavaScript，方便无需服务端即可打开查看。
    The HTML embeds CSS, trace JSON, and a small JavaScript viewer so it can be opened without a server.
    """
    data_json = json.dumps(trace, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CoreSimLLM Trace Demo</title>
  <style>
    /* 颜色变量集中定义，浏览器根据 light/dark 模式自动选择。
       Color variables are centralized and resolved by browser light/dark mode. */
    :root {{
      color-scheme: light dark;
      --bg: light-dark(#f7f8fb, #111318);
      --fg: light-dark(#171a21, #eef1f6);
      --muted: light-dark(#5c6472, #a9b1c0);
      --panel: light-dark(#ffffff, #191d25);
      --panel-2: light-dark(#eef5ff, #202837);
      --border: light-dark(#d7dce5, #343b49);
      --accent: light-dark(#1d6fd6, #76a9ff);
      --green: light-dark(#1b7f5f, #72d1b0);
      --red: light-dark(#b42318, #ff8a80);
    }}
    /* 基础排版保持克制，重点放在数值表格和公式上。
       Base typography is restrained so formulas and numeric tables stay prominent. */
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 500; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; }}
    h3 {{ font-size: 15px; }}
    p {{ margin: 0; color: var(--muted); }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }}
    /* 顶部区域展示模型尺寸元信息，帮助理解每个张量 shape。
       The top area shows model-shape metadata used to interpret tensor shapes. */
    .top {{
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }}
    .stat {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .stat strong {{
      display: block;
      margin-top: 4px;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
    }}
    /* 左侧是步骤导航，右侧是当前步骤的公式、解释和实际数值。
       The left side is step navigation; the right side shows formula, explanation, and values. */
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .steps {{
      display: grid;
      gap: 8px;
    }}
    button {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      color: var(--fg);
      padding: 10px 12px;
      text-align: left;
      font: inherit;
      cursor: pointer;
    }}
    button[aria-pressed="true"] {{
      border-color: var(--accent);
      background: var(--panel-2);
    }}
    .controls {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    .controls button {{
      text-align: center;
    }}
    .detail {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    /* 矩阵表格使用等宽数字，保证小数列对齐，适合观察计算过程。
       Matrix tables use tabular numbers so decimal columns align for computation tracing. */
    .formula {{
      margin: 12px 0;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--fg);
      overflow-wrap: anywhere;
    }}
    .values {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }}
    .calc-list {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    .calc-block {{
      border-top: 1px solid var(--border);
      padding-top: 12px;
      display: grid;
      gap: 8px;
    }}
    .calc-formula {{
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .section-label {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 14px;
      text-transform: uppercase;
    }}
    .value-block {{
      display: grid;
      gap: 6px;
    }}
    .value-head {{
      display: flex;
      gap: 8px;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .shape {{
      color: var(--muted);
      font-size: 12px;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      color: var(--muted);
      text-align: left;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .token-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .token {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
      background: var(--panel-2);
      font-variant-numeric: tabular-nums;
    }}
    /* 概率候选用条形图表达，直观看出 softmax 后的相对大小。
       Probability candidates use bars to make softmax differences visible. */
    .bar-list {{
      display: grid;
      gap: 8px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 86px minmax(80px, 1fr) 96px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .bar {{
      height: 10px;
      border-radius: 8px;
      background: color-mix(in srgb, var(--accent) 22%, transparent);
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: var(--green);
      width: calc(var(--w) * 100%);
    }}
    /* 窄屏下改为单列，让表格可以横向滚动而不是互相挤压。
       On narrow screens the layout becomes single-column and tables scroll horizontally. */
    @media (max-width: 760px) {{
      main {{ padding: 16px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
      .steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 460px) {{
      .stats, .steps {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 70px minmax(60px, 1fr); }}
      .bar-row .prob {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="top">
    <h1>CoreSimLLM Trace Demo</h1>
    <p>A deterministic d_model=8 forward pass with rounded numeric tensors.</p>
    <div class="stats" id="stats"></div>
  </section>
  <section class="layout">
    <aside>
      <div class="steps" id="stepList"></div>
      <div class="controls">
        <button type="button" id="prevBtn">Previous</button>
        <button type="button" id="nextBtn">Next</button>
      </div>
    </aside>
    <section class="detail" id="detail" aria-live="polite"></section>
  </section>
</main>
<script>
// trace 是 Python 预计算出的完整前向传播数值，不在浏览器里重新计算模型。
// trace contains Python-precomputed forward-pass values; the browser only renders them.
const trace = {data_json};
const stats = document.getElementById("stats");
const stepList = document.getElementById("stepList");
const detail = document.getElementById("detail");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
let active = 0;

// 所有动态插入 HTML 的文本都经过转义，避免 token 文本破坏页面结构。
// Every dynamic text value is escaped before insertion to preserve page structure.
function esc(value) {{
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}}

// 数值统一显示 3 位小数，与 Python 端 trace rounding 保持一致。
// Numbers are shown with three decimals, matching Python-side trace rounding.
function fmt(value) {{
  if (typeof value === "number") return value.toFixed(3);
  return esc(value);
}}

// 空 shape 用空字符串表示，标量或文本块不显示维度。
// Empty shape is displayed as blank; scalar/text blocks do not show dimensions.
function shapeLabel(shape) {{
  return shape && shape.length ? "[" + shape.join(", ") + "]" : "";
}}

// 渲染 d_model、head 数等元信息，先给读者建立张量尺寸上下文。
// Render d_model/head metadata first so tensor dimensions have context.
function renderStats() {{
  const meta = trace.meta;
  const items = [
    ["d_model", meta.d_model],
    ["heads", `${{meta.n_heads}} Q / ${{meta.n_kv_heads}} KV`],
    ["head_dim", meta.head_dim],
    ["d_ff", meta.d_ff],
    ["params", meta.parameter_style],
  ];
  stats.innerHTML = items.map(([k, v]) => `
    <div class="stat"><span>${{esc(k)}}</span><strong>${{esc(v)}}</strong></div>
  `).join("");
}}

// 把任意嵌套数组压平成二维表行，例如 [head, row, col] 会显示为 head.row。
// Flatten nested arrays into table rows, e.g. [head, row, col] becomes head.row.
function flattenRows(value, prefix = "") {{
  if (!Array.isArray(value)) return [[prefix || "value", value]];
  if (!Array.isArray(value[0])) {{
    return [[prefix || "row", ...value]];
  }}
  let rows = [];
  value.forEach((item, idx) => {{
    const label = prefix ? `${{prefix}}.${{idx}}` : String(idx);
    if (Array.isArray(item) && Array.isArray(item[0])) {{
      rows = rows.concat(flattenRows(item, label));
    }} else {{
      rows.push([label, ...item]);
    }}
  }});
  return rows;
}}

// 根据数据类型选择渲染方式：token chip、概率条形图或矩阵表。
// Choose rendering by data type: token chips, probability bars, or matrix table.
function renderTable(value) {{
  if (typeof value === "string") return `<div class="token-grid"><span class="token">${{esc(value)}}</span></div>`;
  if (Array.isArray(value) && value.length && typeof value[0] === "string") {{
    return `<div class="token-grid">${{value.map(v => `<span class="token">${{esc(v)}}</span>`).join("")}}</div>`;
  }}
  if (Array.isArray(value) && value.length && typeof value[0] === "object" && !Array.isArray(value[0])) {{
    const maxProb = Math.max(...value.map(d => d.prob || 0), 0.001);
    return `<div class="bar-list">${{value.map(d => `
      <div class="bar-row">
        <code>${{esc(d.token)}}</code>
        <div class="bar" aria-hidden="true"><span style="--w:${{(d.prob || 0) / maxProb}}"></span></div>
        <span class="prob">p=${{fmt(d.prob)}} logit=${{fmt(d.logit)}}</span>
      </div>
    `).join("")}}</div>`;
  }}
  const rows = flattenRows(value);
  const colCount = Math.max(...rows.map(r => r.length));
  const headers = Array.from({{ length: colCount - 1 }}, (_, i) => `<th>c${{i}}</th>`).join("");
  return `<div class="table-wrap"><table><thead><tr><th>row</th>${{headers}}</tr></thead><tbody>${{
    rows.map(row => `
      <tr>
        <td>${{esc(row[0])}}</td>
        ${{Array.from({{ length: colCount - 1 }}, (_, i) => `<td>${{fmt(row[i + 1] ?? "")}}</td>`).join("")}}
      </tr>
    `).join("")
  }}</tbody></table></div>`;
}}

// 单个 value block 既用于关键结果，也用于 calculation 内部的中间量。
// One value block renderer is shared by key results and calculation internals.
function renderValueBlock(v) {{
  return `
    <section class="value-block">
      <div class="value-head">
        <h3>${{esc(v.name)}}</h3>
        <span class="shape">${{esc(shapeLabel(v.shape))}}</span>
      </div>
      ${{v.note ? `<div class="note">${{esc(v.note)}}</div>` : ""}}
      ${{renderTable(v.value)}}
    </section>
  `;
}}

// 渲染当前步骤的公式、解释和每个中间张量。
// Render the active step's formula, explanation, and intermediate tensors.
function renderDetail() {{
  const step = trace.steps[active];
  detail.innerHTML = `
    <h2>${{active + 1}}. ${{esc(step.title)}}</h2>
    <div class="formula"><code>${{esc(step.formula)}}</code></div>
    <p>${{esc(step.detail)}}</p>
    ${{step.calculation && step.calculation.length ? `
      <div class="section-label">Calculation</div>
      <div class="calc-list">
        ${{step.calculation.map((calc, idx) => `
          <section class="calc-block">
            <h3>${{idx + 1}}. ${{esc(calc.title)}}</h3>
            <div class="calc-formula"><code>${{esc(calc.formula)}}</code></div>
            ${{calc.detail ? `<div class="note">${{esc(calc.detail)}}</div>` : ""}}
            <div class="values">${{calc.values.map(renderValueBlock).join("")}}</div>
          </section>
        `).join("")}}
      </div>
    ` : ""}}
    <div class="section-label">Key result</div>
    <div class="values">
      ${{step.values.map(renderValueBlock).join("")}}
    </div>
  `;
  Array.from(stepList.querySelectorAll("button")).forEach((btn, idx) => {{
    btn.setAttribute("aria-pressed", idx === active ? "true" : "false");
  }});
  prevBtn.disabled = active === 0;
  nextBtn.disabled = active === trace.steps.length - 1;
}}

// 步骤列表只绑定一次点击事件，active 改变后重绘详情区域。
// The step list binds one click handler; changing active redraws the detail area.
function renderStepList() {{
  stepList.innerHTML = trace.steps.map((step, idx) => `
    <button type="button" aria-pressed="${{idx === active}}" data-idx="${{idx}}">
      ${{idx + 1}}. ${{esc(step.title)}}
    </button>
  `).join("");
  stepList.addEventListener("click", event => {{
    const btn = event.target.closest("button[data-idx]");
    if (!btn) return;
    active = Number(btn.dataset.idx);
    renderDetail();
  }});
}}

prevBtn.addEventListener("click", () => {{
  active = Math.max(0, active - 1);
  renderDetail();
}});
nextBtn.addEventListener("click", () => {{
  active = Math.min(trace.steps.length - 1, active + 1);
  renderDetail();
}});

renderStats();
renderStepList();
renderDetail();
</script>
</body>
</html>
"""


def write_trace_html(path="trace_demo.html"):
    """Write the interactive trace viewer to disk.
    默认输出到项目根目录的 trace_demo.html；调用方也可以用 --output 指定路径。
    By default this writes trace_demo.html in the project root; callers may override the path with --output.
    """
    trace = build_trace()
    output_path = Path(path).resolve()
    output_path.write_text(_render_html(trace), encoding="utf-8")
    return output_path


def main():
    """
    `python -m coresim_llm.trace_demo` 这类直接调试
    """
    print(format_trace_text())
