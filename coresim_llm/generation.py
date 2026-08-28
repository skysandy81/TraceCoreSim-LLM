"""Text generation and sampling utilities.

实现自回归生成：用 prompt 做首轮前向，然后依靠 KV cache每次只输入新 token。采样支持 greedy、temperature、top-k、top-p 和重复惩罚。
implements autoregressive generation: run the prompt once, then use KV cache so each next step only feeds the new token.
Sampling supports greedy, temperature, top-k, top-p, and repetition penalty.
"""

import numpy as np

from .config import EPS
from .data import make_prompt, tokenizer, vocab_size
from .ops import softmax


def sample_next_token(logits, temp=0.8, top_p=0.9, top_k=0):
    """Sample one token id from logits.

    `temp <= 0` 表示 greedy decoding；否则先做温度缩放，再可选执行top-k/top-p截断，最后按概率采样。
    `temp <= 0` means greedy decoding. Otherwise logits are temperature scaled, optionally filtered by top-k/top-p, then sampled.
    """
    logits = np.asarray(logits, dtype=float).copy()
    if temp <= 0:
        return int(np.argmax(logits))

    # 温度越低分布越尖锐，越高越随机。
    # Lower temperature sharpens the distribution; higher temperature adds randomness.
    logits = logits / max(temp, EPS)
    curr_prob = softmax(logits)

    if top_k and top_k > 0:
        # top-k 只保留概率最高的k个 token。
        # top-k keeps only the k most likely tokens.
        top_k = min(top_k, len(curr_prob))
        keep_idx = np.argsort(curr_prob)[-top_k:]
        keep = np.zeros_like(curr_prob)
        keep[keep_idx] = 1.0
        curr_prob *= keep

    if top_p < 1.0:
        # top-p 保留累积概率达到 p 的最小候选集合。
        # top-p keeps the smallest candidate set whose cumulative probability reaches p.
        sorted_idx = np.argsort(curr_prob)[::-1]
        cum_prob = 0.0
        keep = np.zeros_like(curr_prob)
        for idx in sorted_idx:
            cum_prob += curr_prob[idx]
            keep[idx] = 1.0
            if cum_prob >= top_p:
                break
        curr_prob *= keep

    prob_sum = np.sum(curr_prob)
    if prob_sum <= 0:
        curr_prob = softmax(logits)
    else:
        curr_prob = curr_prob / prob_sum
    return int(np.random.choice(vocab_size, p=curr_prob))


def generate_stream(model, start_text, gen_len=4, temp=0.8, top_p=0.9, top_k=0, rep_penalty=1.1):
    """Yield generated text chunks one token at a time.

    首步输入完整 prompt，后续只输入最新 token，并把 past_kv 传回模型。
    The first step consumes the full prompt. Later steps feed only the
    newest token and pass `past_kv` back into the model.
    """
    ids = tokenizer.encode(start_text, add_bos=True)
    generated = []
    past_kv = None
    step_input = ids
    blocked = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.user_id,
        tokenizer.assist_id,
    }
    for _ in range(gen_len):
        probs, _, past_kv = model.forward(step_input, training=False, past_kv=past_kv)
        logits = np.log(probs[-1] + EPS)
        for idx in set(generated[-8:]):
            # 重复惩罚会降低最近出现 token 的再次出现概率。
            # Repetition penalty reduces the chance of recently generated tokens.
            if logits[idx] > 0:
                logits[idx] /= rep_penalty
            else:
                logits[idx] *= rep_penalty
        next_id = sample_next_token(logits, temp=temp, top_p=top_p, top_k=top_k)
        if next_id in blocked:
            break
        ids.append(next_id)
        generated.append(next_id)
        step_input = [next_id]
        yield tokenizer.decode([next_id])
    yield ""


def generate(model, start_text, gen_len=4, temp=0.8, top_p=0.9, top_k=0):
    """Generate a complete string from a prompt.

    这是 `generate_stream` 的便捷封装，把逐 token 输出拼成字符串。
    Convenience wrapper around `generate_stream` that joins token text.
    """
    out = []
    for token_text in generate_stream(model, start_text, gen_len, temp, top_p, top_k):
        out.append(token_text)
    return "".join(out)


def generate_answer(model, prompt, gen_len=4, temp=0.8, top_p=0.9, top_k=0):
    """Generate an assistant answer using the chat template.

    把普通用户问题包装成 `<user>...<assistant>` 再生成。
    Wraps a user question as `<user>...<assistant>` before generation.
    """
    return generate(model, make_prompt(prompt), gen_len=gen_len, temp=temp, top_p=top_p, top_k=top_k)
