"""A character-level tokenizer with chat special tokens.

真实大模型通常使用 BPE/SentencePiece；这里用字符级 tokenizer
Real LLMs usually use BPE/SentencePiece. This simulator uses a character-level tokenizer
"""

import numpy as np


class Tokenizer:
    """Tokenizer that recognizes full chat special tokens before characters.
    `<user>` 和 `<assistant>` 必须作为整体token识别，否则会被拆成普通字符并污染SFT数据。
    `<user>` and `<assistant>` must be matched as full tokens;
    otherwise they would be split into characters and pollute SFT examples.
    """

    def __init__(self, base_chars):
        """Create vocabulary mappings.
        special tokens放在词表前面，便于在生成阶段屏蔽或检测。
        Special tokens are placed first so generation can block or detect them easily.
        """
        # 开始标记、结束标记、填充标记，把短序列补到统一长度，不参与计算损失、未知词标记，遇到词表没有的字符、用户角色标记，标识用户输入的内容开始、助手角色标记，标识模型回复内容开始
        self.specials = ["<bos>", "<eos>", "<pad>", "<unk>", "<user>", "<assistant>"]
        self.vocab = self.specials + list(base_chars)
        self.word2id = {w: idx for idx, w in enumerate(self.vocab)}
        self.id2word = {idx: w for idx, w in enumerate(self.vocab)}
        self.unk_id = self.word2id["<unk>"]
        self.bos_id = self.word2id["<bos>"]
        self.eos_id = self.word2id["<eos>"]
        self.pad_id = self.word2id["<pad>"]
        self.user_id = self.word2id["<user>"]
        self.assist_id = self.word2id["<assistant>"]

    def encode(self, text, add_bos=False, add_eos=False):
        """Convert text or token-like input into integer ids.
        字符串输入会先贪婪匹配特殊token，再逐字符编码；列表输入既可包含整数id，也可包含字符token。
        String input greedily matches special tokens before character encoding.
        List input may contain integer ids or character tokens.
        """
        ids = []
        if isinstance(text, str):  #判断对象是不是字符串
            i = 0
            specials = sorted(self.specials, key=len, reverse=True)   #把特殊标记按字符串长度从长到短排序
            while i < len(text):
                matched = None
                for sp in specials:
                    if text.startswith(sp, i):
                        matched = sp
                        break
                if matched is not None:
                    ids.append(self.word2id[matched])
                    i += len(matched)
                else:
                    ids.append(self.word2id.get(text[i], self.unk_id))
                    i += 1
        else:
            for tok in text:
                if isinstance(tok, (int, np.integer)):  #判断对象是不是整数索引
                    ids.append(int(tok))
                else:
                    ids.append(self.word2id.get(tok, self.unk_id))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        """Convert ids back to visible text.
        解码时跳过 special tokens，只返回用户可见文本。
        Special tokens are skipped so decoding returns user-visible text.
        """
        words = []
        for idx in ids:
            w = self.id2word.get(int(idx), "<unk>")   #默认是unknown
            if w not in self.specials:
                words.append(w)
        return "".join(words)

    def pad_seq(self, ids, max_len):
        """Pad or truncate a token sequence to a fixed length.
        超过最大长度时强制最后一个token为`<eos>`，模拟数据管线中常见的截断结束标记处理。
        When truncating, the final token is forced to `<eos>`, matching a common data-pipeline convention.
        """
        if len(ids) > max_len:
            ids = ids[:max_len]
            if ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
            return ids
        pad_len = max_len - len(ids)
        return ids + [self.pad_id] * pad_len
