"""End-to-end staged training workflow for CoreSimLLM.
生命周期串起来：未训练快照 -> next-token pretraining -> supervised fine-tuning (SFT) -> RLHF/PPO -> DPO -> 阶段对比。
每个阶段都刻意保留真实LLM流程中的核心信号，但训练规模和梯度实现是教学级简化。
This module wires together the LLM lifecycle:
untrained snapshot -> next-token pretraining -> supervised fine-tuning (SFT) -> RLHF/PPO -> DPO -> stage comparison.
Each stage keeps the core signal used by real LLM workflows, while scale and gradients are simplified for teaching.
"""

from .alignment import ppo_update, train_dpo
from .config import (
    batch_size,
    epoch_ppo,
    epoch_pretrain,
    epoch_rm,
    epoch_sft,
    enable_dpo,
    lr_pretrain,
    lr_sft,
    seed,
    seq_max_len,
    set_seed,
    use_sft_answer_mask,
    warmup_epoch,
)
from .data import (
    DataLoader,
    collate_batch,
    make_chat_example,
    make_lm_example,
    pretrain_train,
    pretrain_val,
    rlhf_prompts,
    sft_train,
    sft_val,
    tokenizer,
)
from .evaluation import clone_model, compare_stage_models, print_stage_comparison, train_reward_model
from .generation import generate, generate_answer
from .model import CoreSimLLM, copy_policy
from .optim import get_lr_cosine
from .plotting import plot_curve


def train_pretrain(model, loss_rec, ppl_rec):
    """Run next-token language-model pretraining.
    真实LLM会在海量文本上预测“下一个 token”。这里用极小语料模拟同一目标，记录训练loss和验证perplexity，让用户看到模型从随机输出到语言片段记忆的变化。
    Real LLMs pretrain by predicting the next token on massive corpora.
    Here a tiny corpus simulates the same objective and records loss/perplexity
    so the transition from random output to learned language fragments is visible.
    """
    print("*" * 60)
    print("Stage 1: next-token pretraining")
    print("Initial sample:", generate(model, "I li", gen_len=6))

    # DataLoader 只负责批量和 shuffle；真正的 LM 样本构造在 data.py 中完成。
    # DataLoader handles batching/shuffle; LM sample construction lives in data.py.
    loader = DataLoader(pretrain_train, batch_size, shuffle=True)
    for epoch in range(epoch_pretrain):
        total_loss = 0.0
        for batch in loader:
            batch_inputs, batch_targets, batch_masks = collate_batch(batch)
            for inp, tgt, loss_mask in zip(batch_inputs, batch_targets, batch_masks):
                # forward 得到每个位置的 next-token 概率和最终 hidden states。
                # forward returns next-token probabilities and final hidden states.
                pred, x, _ = model.forward(inp)
                loss = model.compute_ce_loss(pred, tgt, loss_mask)
                total_loss += loss

                # 使用 warmup + cosine decay，模拟真实训练常见的学习率调度。
                # Warmup plus cosine decay mirrors a common real-training LR schedule.
                curr_lr = get_lr_cosine(epoch, epoch_pretrain, warmup_epoch, lr_pretrain)
                model.update_pretrain_sft(inp, tgt, pred, x, curr_lr, loss_mask)
        avg_loss = total_loss / len(pretrain_train)
        loss_rec.append(avg_loss)

        # 验证集 perplexity 不参与更新，只用于观察 next-token 预测质量。
        # Validation perplexity does not update weights; it tracks prediction quality.
        val_ppl = 0.0
        for sample in pretrain_val:
            inp, tgt, loss_mask = make_lm_example(sample, max_len=seq_max_len)
            pred, _, _ = model.forward(inp, training=False)
            val_ppl += model.compute_ppl(pred, tgt, loss_mask)
        val_ppl /= len(pretrain_val)
        ppl_rec.append(val_ppl)

        if epoch % 200 == 0:
            sample = generate(model, "I li", 6)
            print(f"Pretrain epoch {epoch:4d} loss: {avg_loss:.3f} ppl: {val_ppl:.2f} sample: {sample}")


def train_sft(model, loss_rec, ppl_rec):
    """Run supervised fine-tuning on prompt/answer examples.

    SFT 把 base LM 调成“会按指令回答”的 assistant。loss_mask 会屏蔽 prompt，
    只让 answer token 产生监督信号，贴近真实 instruction tuning 的训练方式。
    SFT turns the base LM into an instruction-following assistant. The
    loss mask hides the prompt and trains only answer tokens, matching the core
    shape of real instruction tuning.
    """
    print("\n" + "=" * 60)
    print("Stage 2: supervised fine-tuning")
    for item in sft_train:
        q = item["query"]
        print(f"Before SFT, prompt: {q} -> {generate_answer(model, q, 16)}")

    loader = DataLoader(sft_train, batch_size, shuffle=True)
    for epoch in range(epoch_sft):
        total_loss = 0.0
        for batch in loader:
            for pair in batch:
                # chat example 会拼接 prompt+answer，并仅对 answer 部分计算 loss。
                # The chat example joins prompt+answer and computes loss on the answer only.
                inp, tgt, loss_mask = make_chat_example(
                    pair["query"],
                    pair["answer"],
                    max_len=seq_max_len,
                    assistant_only=use_sft_answer_mask,
                )
                pred, x, _ = model.forward(inp)
                loss = model.compute_ce_loss(pred, tgt, loss_mask)
                total_loss += loss
                curr_lr = get_lr_cosine(epoch, epoch_sft, warmup_epoch // 2, lr_sft)
                model.update_pretrain_sft(inp, tgt, pred, x, curr_lr, loss_mask)
        avg_loss = total_loss / len(sft_train)
        loss_rec.append(avg_loss)

        # SFT 验证同样只衡量回答 token 的 perplexity。
        # SFT validation also measures perplexity on answer tokens only.
        val_ppl = 0.0
        for pair in sft_val:
            inp, tgt, loss_mask = make_chat_example(
                pair["query"],
                pair["answer"],
                max_len=seq_max_len,
                assistant_only=use_sft_answer_mask,
            )
            pred, _, _ = model.forward(inp, training=False)
            val_ppl += model.compute_ppl(pred, tgt, loss_mask)
        val_ppl /= len(sft_val)
        ppl_rec.append(val_ppl)

        if epoch % 150 == 0:
            sample = generate_answer(model, "What do you like?", 16)
            print(f"SFT epoch {epoch:3d} loss: {avg_loss:.3f} ppl: {val_ppl:.2f} answer: {sample}")


def train_ppo(new_model, ppl_rec, rm=None, return_reward_model=False):
    """Run a compact RLHF/PPO-style alignment stage.

    真实 RLHF 会先训练 reward model，再用 PPO 让 policy 生成更高奖励的回答。
    这里保留 old policy、new policy、reward、KL/clip 的关键关系，但 rollout 和梯度都简化。
    Real RLHF trains a reward model, then uses PPO to move the policy
    toward higher-reward answers. This keeps old/new policy, reward, KL, and
    clipping relationships, while simplifying rollout and gradients.
    """
    print("\n" + "=" * 60)
    print("Stage 3: RLHF-style PPO alignment")
    if rm is None:
        rm = train_reward_model(epochs=epoch_rm)

    # old_policy 是 PPO 的参考行为策略；周期性刷新，限制 new_policy 漂移。
    # old_policy is PPO's behavior/reference policy and is refreshed periodically.
    old_policy = CoreSimLLM()
    copy_policy(old_policy, new_model)
    reward_rec = []

    for epoch in range(epoch_ppo):
        total_rew = 0.0
        for prompt in rlhf_prompts:
            # 先让当前模型生成回答，再由 reward model 打分，模拟 RLHF rollout。
            # Generate with the current model, then score with the reward model like an RLHF rollout.
            ans = generate_answer(new_model, prompt, 16)
            score_ids = tokenizer.encode(ans)
            inp, tgt, loss_mask = make_chat_example(prompt, ans, max_len=seq_max_len)
            rew = rm.get_raw_score(score_ids)
            total_rew += rew
            ppo_update(old_policy, new_model, rm, inp, tgt, loss_mask, rew)
        avg_rew = total_rew / len(rlhf_prompts)
        reward_rec.append(avg_rew)
        if epoch % 80 == 0:
            # 刷新 old_policy，使后续 PPO ratio 继续围绕近期策略计算。
            # Refresh old_policy so later PPO ratios are computed against a recent policy.
            copy_policy(old_policy, new_model)
            test_out = generate_answer(new_model, "Is coding fun?", 16)
            print(f"PPO epoch {epoch:3d} average RM score: {avg_rew:.2f} answer: {test_out}")
    if return_reward_model:
        return reward_rec, rm
    return reward_rec


def run_training(show_plots=True, show_comparison=True, run_seed=None):
    """Run the complete demonstration pipeline and return training artifacts.
    保存每个阶段后的模型快照，因此可以直观比较各阶段的的输出和指标。
    It snapshots the model after each stage so untrained, pretrain, SFT, PPO,
    and DPO can be compared by generated answers and reward/perplexity metrics.
    """
    active_seed = seed if run_seed is None else run_seed
    set_seed(active_seed)
    model = CoreSimLLM()
    print(f"CoreSimLLM parameter count: {model.count_parameters():,}")

    loss_records = []
    ppl_records = []

    # 先记录随机初始化模型，作为“未训练”基线。
    # Capture the randomly initialized model as the untrained baseline.
    stage_models = {"untrained": clone_model(model)}

    train_pretrain(model, loss_records, ppl_records)
    stage_models["pretrain"] = clone_model(model)

    train_sft(model, loss_records, ppl_records)
    stage_models["sft"] = clone_model(model)

    # reward model 只在偏好/奖励阶段使用，SFT 之前不需要它。
    # The reward model is only needed for preference/reward alignment.
    rm = train_reward_model(epochs=epoch_rm)
    reward_records, rm = train_ppo(model, ppl_records, rm=rm, return_reward_model=True)
    stage_models["ppo"] = clone_model(model)

    dpo_loss_records = []
    if enable_dpo:
        # DPO 直接利用 chosen/rejected 偏好对，不再显式 rollout。
        # DPO uses chosen/rejected preference pairs directly, without explicit rollout.
        dpo_loss_records = train_dpo(model)
        stage_models["dpo"] = clone_model(model)
    else:
        print("\nStage 4: DPO alignment skipped by CORESIM_ENABLE_DPO=0")

    print("\n===== Final KV-cache generation test =====")
    test_q = ["What do you like?", "Is coding fun?"]
    for q in test_q:
        # temp=0 使用贪心解码，便于稳定复现实验输出。
        # temp=0 uses greedy decoding for reproducible demo output.
        res = generate_answer(model, q, gen_len=16, temp=0, top_p=1.0)
        print(f"Prompt: {q} -> {res}")

    comparison = None
    if show_comparison:
        # 比较表是本项目展示阶段效果差异的核心输出。
        # The comparison table is the key output for stage-effect demonstration.
        comparison = compare_stage_models(stage_models, rm=rm)
        print_stage_comparison(comparison)

    if show_plots:
        plot_curve(loss_records, ppl_records, reward_records, dpo_loss_records)
    return model, loss_records, ppl_records, reward_records, dpo_loss_records, comparison
