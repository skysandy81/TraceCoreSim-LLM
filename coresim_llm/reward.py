"""Reward model and critic used by the alignment demo.

真实RLHF的reward model通常是另一个Transformer；这里用embedding平均池化 + 线性层来展示“偏好打分”的角色。
Real RLHF reward models are usually Transformer-based;
here mean-pooled embeddings plus linear heads demonstrate the role of preference scoring.
"""

import numpy as np

from .config import REWARD_MAX, REWARD_MIN
from .data import pad_id


class RewardModel:
    """pairwise reward model with a scalar score head and value head.

    `fc_score`给response打分，`fc_critic`模拟PPO中的value baseline。
    `fc_score` scores responses, while `fc_critic` approximates PPO's value baseline.
    """

    def __init__(self):
        """Initialize reward embeddings and heads.

        延迟导入`d_model/vocab_size`避免包初始化时出现循环依赖。
        Delayed imports avoid circular-import issues during package init.
        """
        from .config import d_model
        from .data import vocab_size

        self.emb = np.random.randn(vocab_size, d_model) * 0.1
        self.fc_score = np.random.randn(d_model, 1) * 0.1
        self.fc_critic = np.random.randn(d_model, 1) * 0.1

    def get_hidden(self, token_ids):
        """Mean-pool token embeddings while ignoring padding.

        真实模型会用最后token或pooled hidden state；这里用均值池化保持计算过程透明。
        Real models may use the final token or a pooled hidden state.
        Mean pooling keeps this simulator transparent.
        """
        token_ids = np.asarray(token_ids, dtype=int)
        mask = (token_ids != pad_id).astype(float)
        denom = max(1.0, np.sum(mask))
        return np.sum(self.emb[token_ids] * mask[:, None], axis=0) / denom

    def get_raw_score(self, token_ids):
        """Return the unbounded scalar reward score.

        PPO演示使用raw score，避免sigmoid后过早饱和。
        PPO uses the raw score here to avoid early sigmoid saturation.
        """
        vec = self.get_hidden(token_ids)
        return (vec @ self.fc_score)[0]

    def get_reward(self, token_ids):
        """Map the raw score into a bounded reward range.

        用于展示“奖励范围”概念，不作为默认PPO优化信号。
        Demonstrates bounded rewards, but the default PPO demo uses raw scores.
        """
        score = self.get_raw_score(token_ids)
        score = 1 / (1 + np.exp(-score))
        return REWARD_MIN + (REWARD_MAX - REWARD_MIN) * score

    def get_value(self, token_ids):
        """Estimate a value baseline for PPO.

        value baseline 用于把 reward 转成 advantage。
        The value baseline helps convert reward into advantage.
        """
        vec = self.get_hidden(token_ids)
        return (vec @ self.fc_critic)[0]

    def pairwise_loss(self, good_ids, bad_ids, margin=0.2):
        """Compute a smooth pairwise ranking loss.

        当 good 分数不高于 bad 分数足够多时，loss 会变大。
        The loss grows when the good response does not outrank the bad response.
        """
        r_good = self.get_raw_score(good_ids)
        r_bad = self.get_raw_score(bad_ids)
        return np.logaddexp(0, margin + r_bad - r_good)

    def update_pairwise(self, good_ids, bad_ids, opt, lr, margin=0.2):
        """Train the reward score head on one preference pair.

        这是手写梯度的教学版本，只更新 reward embedding 和 score head。
        This is a hand-derived teaching update that only modifies reward embeddings and the score head.
        """
        good_ids = np.asarray(good_ids, dtype=int)
        bad_ids = np.asarray(bad_ids, dtype=int)
        good_mask = good_ids != pad_id
        bad_mask = bad_ids != pad_id
        h_good = self.get_hidden(good_ids)
        h_bad = self.get_hidden(bad_ids)
        diff = margin + (h_bad @ self.fc_score)[0] - (h_good @ self.fc_score)[0]
        loss = np.logaddexp(0, diff)   #log(1+e^diff)
        grad_diff = 1 / (1 + np.exp(-diff))

        # score head 梯度推动 good hidden 上升、bad hidden 下降。
        # The score-head gradient pushes good hidden up and bad hidden down.
        grad_fc = grad_diff * (h_bad - h_good)[:, None]
        grad_emb = np.zeros_like(self.emb)
        grad_good = -grad_diff * self.fc_score.ravel() / max(1.0, np.sum(good_mask))
        grad_bad = grad_diff * self.fc_score.ravel() / max(1.0, np.sum(bad_mask))
        np.add.at(grad_emb, good_ids[good_mask], grad_good)
        np.add.at(grad_emb, bad_ids[bad_mask], grad_bad)

        opt.lr = lr
        self.fc_score = opt.update("rm_fc_score", self.fc_score, grad_fc)
        self.emb = opt.update("rm_emb", self.emb, grad_emb)
        return loss
