""" datasets and sample-building utilities.

这里的数据集非常小，只用于演示训练阶段差异。真实工程中这一层会负责读取语料、chat template、padding、loss mask和batch collate。
The datasets are intentionally tiny and only demonstrate stage differences.
In production, this layer would handle corpora, chat templates, padding, loss masks, and batching.
"""

import numpy as np

from .config import EPS, seq_max_len
from .tokenizer import Tokenizer


# 预训练样本只做 next-token prediction，不区分用户和助手。
# Pretraining samples use plain next-token prediction.
pretrain_all = [
    "I like coding.",
    "Coding is fun.",
    "Programming feels useful.",
    "I write code daily.",
]
pretrain_train = pretrain_all[:3]
pretrain_val = pretrain_all[3:]

# SFT 样本使用 query/answer 结构，并只在 answer 上计算 loss。
# SFT samples use query/answer pairs and only train on the answer span.
sft_full = [
    {"query": "What do you like?", "answer": "I like coding."},
    {"query": "Is coding fun?", "answer": "Coding is fun."},
    {"query": "What do you do daily?", "answer": "I write code daily."},
]
sft_train = sft_full[:2]
sft_val = sft_full[2:]

rlhf_prompts = ["What do you like?", "Is coding fun?"]

# 偏好数据包含同一 prompt 下的 chosen/rejected response。
# Preference data contains chosen/rejected responses for the same prompt.
pair_rm_data = [
    {"prompt": "Is coding fun?", "good": "Coding is fun.", "bad": "Coding is boring."}
]


def _build_base_vocab():
    """Build a character vocabulary from every 小型 dataset string.
    从所有训练/评估文本中收集字符，所有出现过的字符
    Collect characters from all train/eval text
    """
    corpus = list(pretrain_all)
    for item in sft_full:
        corpus.extend([item["query"], item["answer"]])
    for item in pair_rm_data:
        corpus.extend([item["prompt"], item["good"], item["bad"]])
    corpus.extend(rlhf_prompts)

    chars = []
    for text in corpus:
        for ch in text:
            if ch not in chars:
                chars.append(ch)
    return chars


base_vocab_chars = _build_base_vocab()
tokenizer = Tokenizer(base_vocab_chars)  #最终的tokenizer是在前面追加特殊token
vocab_size = len(tokenizer.vocab)
pad_id = tokenizer.pad_id


def make_lm_example(text, max_len=None):
    """Create one next-token prediction example.
    输入是 `ids[:-1]`，目标是 `ids[1:]`，这就是自回归语言模型最核心的训练形式。
    Inputs are `ids[:-1]` and targets are `ids[1:]`, the core training pattern for autoregressive language models.
    """
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if max_len is not None:
        ids = tokenizer.pad_seq(ids, max_len)
    inp = np.asarray(ids[:-1], dtype=int)   #去掉最后一个元素
    tgt = np.asarray(ids[1:], dtype=int)   #去掉开头第一个元素
    loss_mask = (tgt != pad_id).astype(float)   #生成loss计算的mask，pad位置是0，不计算损失；真实token位置是1，参与loss
    return inp, tgt, loss_mask


def make_chat_example(query, answer, max_len=None, assistant_only=True):
    """Create a supervised chat example with an assistant-only loss mask.
    prompt token进入上下文但不计loss；只有`<assistant>`后的回答token参与监督微调。
    Prompt tokens provide context but do not contribute loss; only tokens after `<assistant>` are trained.
    """
    prompt_ids = tokenizer.encode(make_prompt(query), add_bos=True)
    answer_ids = tokenizer.encode(answer, add_eos=True)
    ids = prompt_ids + answer_ids
    answer_start = len(prompt_ids) - 1
    if max_len is not None:
        ids = tokenizer.pad_seq(ids, max_len)
    inp = np.asarray(ids[:-1], dtype=int)
    tgt = np.asarray(ids[1:], dtype=int)
    loss_mask = np.zeros_like(tgt, dtype=float)   #与tgt完全一样，但数值全为0的矩阵
    if assistant_only and answer_start < len(loss_mask):
        loss_mask[answer_start:] = 1.0
    elif not assistant_only:
        loss_mask[:] = 1.0
    loss_mask *= tgt != pad_id   #把padding位置屏蔽，padding部分不参与 loss 计算。
    return inp, tgt, loss_mask


def make_prompt(prompt):
    """Apply the minimal chat template.
    真实模型会使用更复杂的 template；这里保留角色边界即可。
    Real models use richer templates; this keeps only role boundaries.
    """
    return f"<user>{prompt}<assistant>"


def token_loss_mask(target_ids, loss_mask=None):
    """Return the effective loss mask after removing padding positions.
    即使调用者传了mask，pad token也永远不参与loss。生成 loss 掩码，padding 部分清零，不参与损失计算。
    Padding tokens never contribute to loss, even when a caller passes an explicit mask.
    """
    target_ids = np.asarray(target_ids, dtype=int)
    if loss_mask is None:
        return (target_ids != pad_id).astype(float)
    return np.asarray(loss_mask, dtype=float) * (target_ids != pad_id)


def sequence_logprob(probs, target_ids, loss_mask=None, normalize=False):
    """Sum or average target-token log probabilities.
    PPO 和 DPO 都需要在 response token 上计算 log-prob；`normalize`用于避免长回答天然拥有更大的 log-prob 绝对值。
    PPO and DPO both need response-token log-probs. `normalize` prevents longer answers from dominating by sequence length.
    """
    target_ids = np.asarray(target_ids, dtype=int)
    mask = token_loss_mask(target_ids, loss_mask)
    rows = np.arange(len(target_ids))
    logp = np.log(probs[rows, target_ids] + EPS) * mask   #交叉熵，取目标 token 对数概率
    denom = max(1.0, np.sum(mask))
    value = np.sum(logp)
    return value / denom if normalize else value


class DataLoader:
    """Small iterable DataLoader for batches.
    这里只做索引打乱和按batch_size切片；真实项目会有多进程、streaming、bucketing等复杂逻辑。
    This only shuffles indices and slices batches. Real projects add multiprocessing, streaming, bucketing, and more.
    """

    def __init__(self, data_list, batch_size, shuffle=True):
        self.data = data_list
        self.bs = batch_size
        self.shuffle = shuffle
        self.idx = list(range(len(self.data)))

    def __iter__(self):
        """Yield Python-list batches.
        保持批次为原始对象，后续collate再转成NumPy数组。
        Keep batches as raw objects; collate converts them to NumPy arrays.
        """
        if self.shuffle:
            np.random.shuffle(self.idx)   #打乱样本索引，实现训练时样本随机顺序。
        for i in range(0, len(self.idx), self.bs):   #最后一位是步长
            batch_idx = self.idx[i : i + self.bs]
            yield [self.data[j] for j in batch_idx]   #迭代返回一个batch的数据，做数据加载。调用时循环拿一批一批样本。


def collate_batch(text_list, max_len=seq_max_len):
    """Collate plain LM text samples into input/target/mask arrays.
    输出三组张量，形状均为 `[batch, seq]`，供训练循环逐样本处理。
    Returns three `[batch, seq]` arrays for the training loop.
    """
    batch_inputs = []
    batch_targets = []
    batch_masks = []
    for txt in text_list:
        inp, tgt, loss_mask = make_lm_example(txt, max_len)
        batch_inputs.append(inp)
        batch_targets.append(tgt)
        batch_masks.append(loss_mask)
    return np.array(batch_inputs), np.array(batch_targets), np.array(batch_masks)
