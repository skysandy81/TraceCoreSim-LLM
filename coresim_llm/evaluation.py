"""Stage comparison utilities.

直观比较 untrained、pretrain、SFT、PPO、DPO 阶段。除了生成文本，还使用 teacher-forced 指标，让阶段差异更稳定、更真实。
compares untrained, pretrain, SFT, PPO, and DPO stages.
It uses teacher-forced metrics in addition to generated text so differences are stable and meaningful.
"""

import numpy as np

from .config import epoch_rm, lr_rm, seq_max_len
from .data import (
    make_chat_example,
    make_lm_example,
    make_prompt,
    pair_rm_data,
    pretrain_train,
    rlhf_prompts,
    sequence_logprob,
    sft_train,
    tokenizer,
)
from .generation import generate_answer
from .model import CoreSimLLM, copy_policy
from .optim import AdamW
from .reward import RewardModel


def clone_model(model):
    """Clone a model without perturbing NumPy's random state.

    构造新模型会消耗随机数；这里保存并恢复RNG状态，保证后续训练和生成结果不会因为“保存快照”而改变。
    Constructing a new model consumes random numbers. Saving and restoring RNG state keeps snapshot creation from changing later results.
    """
    rng_state = np.random.get_state()
    clone = CoreSimLLM()
    np.random.set_state(rng_state)
    copy_policy(clone, model)
    return clone


def train_reward_model(epochs=epoch_rm):
    """Train the tiny reward model on response-level preference pairs.

    只用good/bad response训练，避免prompt token淹没回答差异。
    Trains only on good/bad responses so prompt tokens do not dominate the answer signal.
    """
    rm = RewardModel()
    opt = AdamW(lr_rm, weight_decay=0.0)
    for _ in range(epochs):
        for pair in pair_rm_data:
            good = tokenizer.encode(pair["good"])
            bad = tokenizer.encode(pair["bad"])
            rm.update_pairwise(good, bad, opt, lr_rm)
    return rm


def mean_lm_ppl(model, samples=pretrain_train):
    """Compute mean perplexity on plain language-model samples.

    PPL越低，表示模型越能预测普通语料中的下一个token。
    Lower PPL means the model predicts next tokens in plain text better.
    """
    total = 0.0
    for text in samples:
        inp, tgt, mask = make_lm_example(text, max_len=seq_max_len)
        probs, _, _ = model.forward(inp, training=False)
        total += model.compute_ppl(probs, tgt, mask)
    return total / len(samples)


def mean_sft_loss(model, samples=sft_train):
    """Compute mean assistant-only SFT loss.

    该指标只评估回答部分，能够反映SFT是否学会对话格式和答案。
    This metric evaluates only answer tokens, reflecting whether SFT learned the chat format and answers.
    """
    total = 0.0
    for item in samples:
        inp, tgt, mask = make_chat_example(item["query"], item["answer"], max_len=seq_max_len)
        probs, _, _ = model.forward(inp, training=False)
        total += model.compute_ce_loss(probs, tgt, mask)
    return total / len(samples)


def preference_margin(model):
    """Measure how much the model prefers chosen over rejected answers.

    margin = logp(good response) - logp(bad response)，越高越符合偏好数据。
    margin = logp(good response) - logp(bad response); higher is more aligned.
    """
    margins = []
    for pair in pair_rm_data:
        pos = make_chat_example(pair["prompt"], pair["good"], max_len=seq_max_len)
        neg = make_chat_example(pair["prompt"], pair["bad"], max_len=seq_max_len)
        pos_prob, _, _ = model.forward(pos[0], training=False)
        neg_prob, _, _ = model.forward(neg[0], training=False)
        pos_logp = sequence_logprob(pos_prob, pos[1], pos[2], normalize=True)
        neg_logp = sequence_logprob(neg_prob, neg[1], neg[2], normalize=True)
        margins.append(pos_logp - neg_logp)
    return float(np.mean(margins))


def mean_generated_score(model, rm, prompts=rlhf_prompts):
    """Score greedy generated answers with the reward model.

    这是生成质量的粗略 reward 视角；不替代 teacher-forced 指标。
    This gives a rough reward-model view of generated answers; it does
    not replace teacher-forced metrics.
    """
    scores = []
    for prompt in prompts:
        answer = generate_answer(model, prompt, gen_len=16, temp=0, top_p=1.0)
        score_ids = tokenizer.encode(answer)
        scores.append(rm.get_raw_score(score_ids))
    return float(np.mean(scores))


def greedy_answers(model, prompts=rlhf_prompts):
    """Generate deterministic answers for each comparison prompt.

    使用 greedy decoding，避免随机采样让阶段对比忽高忽低。
    Uses greedy decoding so random sampling does not obscure stage comparisons.
    """
    return {
        prompt: generate_answer(model, prompt, gen_len=16, temp=0, top_p=1.0)
        for prompt in prompts
    }


def evaluate_stage(model, rm):
    """Return all metrics for one model snapshot.

    每个阶段都用相同评估函数，保证横向比较公平。
    Every stage uses the same evaluator for fair comparison.
    """
    return {
        "lm_ppl": mean_lm_ppl(model),
        "sft_loss": mean_sft_loss(model),
        "pref_margin": preference_margin(model),
        "rm_score": mean_generated_score(model, rm),
        "answers": greedy_answers(model),
    }


def compare_stage_models(stage_models, rm=None):
    """Evaluate a dictionary of stage name -> model snapshot.

    如果调用者未提供 reward model，则临时训练一个统一 RM。
    If no reward model is provided, a shared RM is trained for comparison.
    """
    if rm is None:
        rm = train_reward_model()
    return {stage: evaluate_stage(model, rm) for stage, model in stage_models.items()}


def print_stage_comparison(results):
    """Print a compact stage comparison report.

    表格展示稳定数值指标，随后展示固定 prompt 的 greedy 输出。
    The table shows stable numeric metrics, followed by greedy outputs for fixed prompts.
    """
    print("\n===== Stage comparison: teacher-forced metrics =====")
    print("Lower LM PPL and SFT loss are better. Higher preference margin and RM score are better.")
    header = f"{'stage':<12} {'lm_ppl':>10} {'sft_loss':>10} {'pref_margin':>13} {'rm_score':>10}"
    print(header)
    print("-" * len(header))
    for stage, metrics in results.items():
        print(
            f"{stage:<12} "
            f"{metrics['lm_ppl']:>10.2f} "
            f"{metrics['sft_loss']:>10.3f} "
            f"{metrics['pref_margin']:>13.3f} "
            f"{metrics['rm_score']:>10.2f}"
        )

    print("\n===== Stage comparison: greedy answers =====")
    prompts = list(next(iter(results.values()))["answers"].keys())
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for stage, metrics in results.items():
            answer = metrics["answers"][prompt] or "<empty>"
            print(f"  {stage:<10} -> {answer}")
