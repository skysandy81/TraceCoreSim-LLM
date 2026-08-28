"""Public package exports for CoreSimLLM.

将项目中最常用的配置、数据、模型、训练、评估和trace/demo API集中导出，方便交互式实验或外部脚本直接`import coresim_llm`使用。
re-exporting the most common configuration, data, model, training, evaluation, and trace/demo APIs
so interactive experiments or external scripts can simply `import coresim_llm`.
"""

from .alignment import dpo_loss, ppo_update, train_dpo
from .config import *
from .data import *
from .evaluation import (
    clone_model,
    compare_stage_models,
    evaluate_stage,
    print_stage_comparison,
    train_reward_model,
)
from .generation import generate, generate_answer, generate_stream, sample_next_token
from .model import (
    CoreSimLLM,
    DecoderLayer,
    GQAAttention,
    LoRALinear,
    copy_policy,
    load_checkpoint,
    save_checkpoint,
)
from .optim import AdamW, get_lr_cosine
from .plotting import plot_attn_heatmap, plot_curve
from .reward import RewardModel
from .train import run_training, train_ppo, train_pretrain, train_sft
from .trace_demo import build_trace, format_trace_text, write_trace_html
