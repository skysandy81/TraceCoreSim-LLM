"""Numerical operators used by the simulator.

Transformer前向计算中最基础的数学算子，例如softmax、RMSNorm、RoPE、SwiGLU、dropout和causal mask。
Transformer math primitives: softmax, RMSNorm, RoPE, SwiGLU, dropout, and causal masks.
"""

import numpy as np

from .config import EPS, MASK_FILL


def softmax(x):
    """Compute a numerically stable softmax along the last axis.

    先减去最大值，避免指数运算溢出；真实LLM中softmax用于把logits或attention scores转成概率分布。
    Subtracting the max prevents overflow. In real LLMs, softmax turns logits or attention scores into probability distributions.
    """
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def log_softmax(x):
    """Compute log-softmax in a stable way.

    log-softmax常用于交叉熵、策略log-prob和偏好优化。
    log-softmax is commonly used for cross-entropy, policy log-probs, and preference optimization.
    """
    max_x = np.max(x, axis=-1, keepdims=True)
    shifted = x - max_x
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + EPS)


def silu(x):
    """SiLU activation: x * sigmoid(x).

    SwiGLU 中常用的门控激活函数。
    The gated activation typically used inside SwiGLU blocks.
    """
    return x / (1 + np.exp(-x))


def swiglu(x, W1, b1, W3, b3):
    """SwiGLU feed-forward hidden projection.

    真实LLaMA类模型常使用`SiLU(xW1) * (xW3)`作为FFN的门控中间层，再通过W2投回d_model。
    LLaMA-like models commonly use `SiLU(xW1) * (xW3)` as the gated FFN hidden state before projecting back with W2.
    """
    x1 = x @ W1 + b1
    x3 = x @ W3 + b3
    return silu(x1) * x3


def rms_norm(x, weight=None, eps=1e-5):
    """Apply RMSNorm over the last dimension.

    RMSNorm只按均方根缩放，不减均值；比LayerNorm更轻量。
    RMSNorm rescales by root mean square without mean subtraction, making it lighter than LayerNorm.
    """
    # 在最后一维求平方的平均值，保持维度方便广播
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    y = x / rms
    if weight is not None:
        y = y * weight
    return y


def dropout(x, drop_prob=0.1, training=True):
    """Apply inverted dropout during training.

    推理阶段关闭dropout；训练阶段用 1/(1-p) 保持期望值不变。
    Dropout is disabled for inference. During training, scaling by 1/(1-p) keeps the expected activation value unchanged.
    """
    if not training or drop_prob <= 0:
        return x
    keep_prob = 1 - drop_prob
    # 二项分布采样：输出和x形状一样的数组，每个元素以概率keep_prob取1。dropout掩码：1代表保留该位置，0代表丢弃。
    mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
    return x * mask


def get_causal_mask(query_len, key_len=None, query_start=0):
    """Build a causal attention mask.

    因果掩码（decoder自注意力上三角掩码）
    位置i只能看自己和过去token，不能看未来token；`query_start`让增量KV cache推理时新token的绝对位置保持正确。
    Position i can attend to itself and past tokens only. `query_start`
    preserves absolute positions during incremental KV-cache decoding.
    """
    if key_len is None:
        key_len = query_len
    q_pos = np.arange(query_start, query_start + query_len)[:, None]
    k_pos = np.arange(key_len)[None, :]
    return np.where(k_pos > q_pos, MASK_FILL, 0.0)


def rope_embedding(x, start_pos=0):
    """Apply Rotary Position Embedding to one attention head.

    RoPE通过旋转偶/奇维特征对来注入位置信息，避免单独相加位置向量。把向量两两一组，按位置做复数旋转，给query/key注入位置信息
    RoPE injects position by rotating even/odd feature pairs instead of adding a separate position embedding.
    """
    seq_len, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    theta = np.arange(0, dim // 2) / (dim // 2)
    theta = 10000 ** (-theta)
    pos = np.arange(start_pos, start_pos + seq_len)[:, None]
    freqs = pos * theta
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = np.zeros_like(x)
    x_rot[..., ::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos
    return x_rot
