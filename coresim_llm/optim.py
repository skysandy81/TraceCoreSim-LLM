"""Optimization helpers.

实现一个最小 AdamW 优化器和余弦学习率调度。它用于展示训练阶段的参数更新流程，但不是完整深度学习框架。
implements a minimal AdamW optimizer and cosine learning rate schedule.
It demonstrates training-time parameter updates without becoming a full deep-learning framework.
"""

import numpy as np

from .config import EPS, grad_clip_norm


class AdamW:
    """Small stateful AdamW optimizer.

    真实 AdamW 会在框架中跟踪每个参数的动量和二阶矩；这里用`name -> state`字典模拟同样的概念。
    Real AdamW tracks first and second moments per parameter in a
    framework optimizer. Here a `name -> state` dictionary simulates that idea.
    """

    def __init__(self, lr, weight_decay, beta1=0.9, beta2=0.999):
        """Store optimizer hyperparameters and empty state.

        `m/v/t` 分别是 Adam 的一阶矩、二阶矩和 step 计数。
        `m/v/t` are Adam's first moment, second moment, and step count.
        """
        self.lr = lr
        self.wd = weight_decay
        self.b1 = beta1
        self.b2 = beta2
        self.eps = EPS
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, name, param, grad):
        """Apply one AdamW update and return the modified parameter array.

        这个函数会原地修改 `param`；先做梯度裁剪，再做 Adam 偏差校正，最后加入 decoupled weight decay。
        This mutates `param` in place. It clips gradients, applies Adam bias correction, then adds decoupled weight decay.
        """
        if name not in self.m or self.m[name].shape != param.shape:
            # 第一次看到参数或形状改变时初始化优化器状态。
            # Initialize optimizer state when a parameter is first seen or resized.
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
            self.t[name] = 0
        self.t[name] += 1

        grad_norm = np.linalg.norm(grad)
        if grad_norm > grad_clip_norm:
            # 全局范数裁剪，防止小型训练中梯度过大导致数值爆炸。
            # Global norm clipping prevents numerical blow-ups in the 小型 loop.
            grad = grad * grad_clip_norm / grad_norm

        # Adam 动量更新和偏差校正。
        # Adam moment updates and bias correction.
        self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * grad
        self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (grad**2)
        m_hat = self.m[name] / (1 - self.b1 ** self.t[name])
        v_hat = self.v[name] / (1 - self.b2 ** self.t[name])
        param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * param)
        return param


def get_lr_cosine(epoch, total_epoch, warmup, base_lr):
    """Warmup plus cosine decay learning-rate schedule.

    真实训练通常先 warmup，再逐步衰减学习率；这里保留这个流程形状。
    Real training often warms up first and then decays the learning rate; this keeps that workflow shape.
    """
    if warmup > 0 and epoch < warmup:
        return base_lr * epoch / warmup
    if total_epoch <= warmup:
        return base_lr
    progress = (epoch - warmup) / (total_epoch - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
