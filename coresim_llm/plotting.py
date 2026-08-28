"""Plotting helpers for training curves.

绘图不是模型核心逻辑，只用于直观看训练 loss、PPL、PPO 分数和 DPO loss。
Plotting is not core model logic; it visualizes loss, PPL, PPO score, and DPO loss trends.
"""

import matplotlib.pyplot as plt


def plot_attn_heatmap(attn_weight, head_idx=0):
    """Show one attention head as a heatmap.

    横轴是 key 位置，纵轴是 query 位置，颜色表示注意力权重。
    Columns are key positions, rows are query positions, and color is attention weight.
    """
    mat = attn_weight[head_idx]
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()
    plt.title(f"Attention heatmap, head {head_idx}")
    plt.show()


def plot_curve(loss_list, ppl_list, reward_list, dpo_loss_list=None):
    """Plot the main training curves.

    该函数用于跑完整训练时快速查看阶段性趋势
    Use this after full training to inspect stage trends.
    """
    fig, axes = plt.subplots(3 if dpo_loss_list is None else 4, 1, figsize=(10, 10))
    ax1 = axes[0]
    ax1.plot(loss_list)
    ax1.set_title("Pretrain and SFT cross-entropy loss")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(ppl_list, c="green")
    ax2.set_title("Validation perplexity")
    ax2.grid(True)

    ax3 = axes[2]
    if len(reward_list) > 0:
        ax3.plot(reward_list, c="orange")
        ax3.set_title("Average PPO RM score")
    ax3.grid(True)

    if dpo_loss_list is not None:
        ax4 = axes[3]
        ax4.plot(dpo_loss_list, c="red")
        ax4.set_title("DPO preference loss")
        ax4.grid(True)
    plt.tight_layout()
    plt.show()
