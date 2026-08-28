# TraceCoreSim-LLM

TraceCoreSim-LLM is a minimal, traceable, end-to-end simulation framework for
the pretraining, supervised fine-tuning, and preference alignment pipeline of
large language models.
It is intentionally small: the goal is not benchmark performance, but a clear
and reproducible implementation of the core mechanisms that connect modern LLM
training stages.

Proposed manuscript title:

**TraceCoreSim-LLM: A Traceable Minimal Simulator for the Pretraining,
Supervised Fine-Tuning, and Preference Alignment Pipeline of Large Language
Models**

## What This Project Demonstrates

- A tiny decoder-only Transformer implemented in NumPy.
- Modern LLM building blocks: RMSNorm, RoPE, SwiGLU, grouped-query attention,
  causal masking, KV cache, AdamW, warmup plus cosine decay.
- A complete staged pipeline: untrained baseline, next-token pretraining,
  supervised fine-tuning, reward modeling, PPO-style RLHF, and DPO.
- Deterministic tensor-level tracing with a `d_model=8` forward pass.
- Stage-wise evaluation using LM perplexity, SFT loss, preference margin,
  reward-model score, and greedy answer snapshots.

## Quick Start

Run the full training simulation:

```powershell
python main.py
```

Print the deterministic tensor trace:

```powershell
python main.py --trace
```

Generate the interactive trace viewer:

```powershell
python main.py --trace-html
```

## Repository Map

- `main.py`: command-line entry point.
- `coresim_llm/model.py`: tiny decoder-only Transformer.
- `coresim_llm/train.py`: staged training workflow.
- `coresim_llm/alignment.py`: PPO-style and DPO preference alignment.
- `coresim_llm/reward.py`: minimal reward model and critic.
- `coresim_llm/evaluation.py`: stage comparison metrics.
- `coresim_llm/trace_demo.py`: deterministic numeric trace and HTML viewer.

## Scope and Limitations

TraceCoreSim-LLM is not a production training framework. The datasets are tiny,
the gradients are teaching approximations, and most updates affect only the
token embedding and output projection. These constraints are part of the design:
they make the full pretraining-SFT-RLHF-DPO path inspectable in seconds.
