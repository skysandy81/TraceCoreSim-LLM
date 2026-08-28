"""Global configuration for the CoreSimLLM simulator.

集中保存模型结构、训练超参和随机种子工具。主模型保持`d_model=32`，用于稳定展示 untrained/pretrain/SFT/PPO/DPO 的阶段效果。
centralizes model shape, training hyperparameters, and seed helpers. The main model keeps `d_model=32` so staged behavior remains stable.
"""

import os

import numpy as np

# 极小常量用于避免 log(0)；MASK_FILL 用于因果 mask 中屏蔽未来 token。
# EPS prevents log(0); MASK_FILL blocks future tokens in causal masks.
MASK_FILL = -1e9
EPS = 1e-8
REWARD_MIN = 0.0
REWARD_MAX = 5.0

def _env_bool(name, default):
    """Read a boolean experiment switch from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    """Read an integer experiment value from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name, default):
    """Read a float experiment value from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


seed = _env_int("CORESIM_SEED", 42)


# 模型尺寸很小，便于在 NumPy 中快速运行和观察阶段差异。环境变量只用于实验脚本。
# Model dimensions stay tiny so the simulator runs quickly in NumPy. Environment
# overrides are reserved for reproducibility and ablation scripts.
seq_max_len = _env_int("CORESIM_SEQ_MAX_LEN", 48)
d_model = _env_int("CORESIM_D_MODEL", 32)
n_heads = _env_int("CORESIM_N_HEADS", 2)
n_layers = _env_int("CORESIM_N_LAYERS", 2)
d_ff = _env_int("CORESIM_D_FF", d_model * 2)
head_dim = d_model // n_heads

# GQA使用较少的K/V头，模拟真实LLM常用的推理加速结构。
# GQA uses fewer K/V heads than Q heads, matching a common modern LLM pattern.
use_gqa = _env_bool("CORESIM_USE_GQA", True)
n_kv_heads = _env_int("CORESIM_N_KV_HEADS", 1 if use_gqa else n_heads)
kv_head_dim = head_dim
kv_dim = n_kv_heads * kv_head_dim
use_rope = _env_bool("CORESIM_USE_ROPE", True)
use_sft_answer_mask = _env_bool("CORESIM_USE_SFT_ANSWER_MASK", True)
enable_dpo = _env_bool("CORESIM_ENABLE_DPO", True)

# 检查防止配置组合产生不可reshape的注意力张量。
# check prevent attention tensors from becoming unshapable.
if d_model % n_heads != 0:
    raise ValueError("d_model must be divisible by n_heads")
if n_heads % n_kv_heads != 0:
    raise ValueError("n_heads must be divisible by n_kv_heads")

# LoRA参数保留为演示接口；当前主训练仍是简化的embedding/output更新。
# LoRA parameters are exposed for demonstration; default training remains simplified.
lora_rank = 4
lora_alpha = 8
lora_scaling = lora_alpha / lora_rank

batch_size = _env_int("CORESIM_BATCH_SIZE", 2)

# 阶段学习率。PPO/DPO故意很小，避免小型优化器破坏SFT学到的回答。
# Stage learning rates. PPO/DPO are intentionally small to preserve SFT behavior.
lr_pretrain = _env_float("CORESIM_LR_PRETRAIN", 0.06)
lr_sft = _env_float("CORESIM_LR_SFT", 0.04)
lr_ppo = _env_float("CORESIM_LR_PPO", 0.0001)
lr_dpo = _env_float("CORESIM_LR_DPO", 0.0001)
lr_rm = _env_float("CORESIM_LR_RM", 0.05)

weight_decay = 0.01
grad_clip_norm = 1.0

# 默认轮次经过调参，目标是让阶段对比清楚且运行时间保持在数秒级。
# Default epochs are tuned for clear stage comparison and short runtime.
epoch_pretrain = _env_int("CORESIM_EPOCH_PRETRAIN", 600)
epoch_sft = _env_int("CORESIM_EPOCH_SFT", 300)
epoch_ppo = _env_int("CORESIM_EPOCH_PPO", 20)
epoch_dpo = _env_int("CORESIM_EPOCH_DPO", 20)
epoch_rm = _env_int("CORESIM_EPOCH_RM", 200)
warmup_epoch = _env_int("CORESIM_WARMUP_EPOCH", 60)

# PPO/DPO 的关键超参，只保留教学版核心概念。
# Core PPO/DPO knobs; this simulator keeps only the teaching essentials.
ppo_epsilon = 0.2
PPO_CLIP_LOW = 1 - ppo_epsilon
PPO_CLIP_HIGH = 1 + ppo_epsilon
gamma = 0.95
kl_coeff = 0.02

dpo_beta = 0.1


def set_seed(seed=42):
    """Set NumPy's global random seed.
    用于让训练和演示结果可复现。
    Makes training and demo outputs reproducible.
    """
    np.random.seed(seed)


def print_shape(name, tensor):
    """Print a tensor shape for quick debugging.
    简单的调试辅助函数，帮助检查矩阵维度。
    Small debugging helper for checking matrix dimensions.
    """
    print(f"{name} shape = {tensor.shape}")
