# Appendix A — Models, Datasets & Sources Quick Reference

> ⚠️ Authenticity warning for the whole table
> The **model / dataset / team / report names listed below are all circulating, unverified rumors** — this book **cannot confirm their real existence, specifications, or licensing status**. This table serves only as an index of "things rumored to exist"; it **does not constitute a recommendation to download or use** anything. Before using anything, always verify the source, license, and legality yourself (see Chapter 6's six iron rules).

## A.1　Distilled Models (circulating, unverified)

| Model name | Base | Claimed specs / positioning | Claimed source |
|---|---|---|---|
| **Qwable-v1** | Qwen3.6-35B-A3B | SFT, inherits Fable 5 tool-calling; ~24GB (Q4_K_M), ~102 tok/s | r/huggingface, r/LocalLLaMA, r/opencodeCLI |
| **GPT-OSS 120B Fable-5 Distilled** | GPT-OSS 120B | MoE 128 experts / 4 activated; ~72GB, ~35–50 tok/s; GGUF (Q8_0/Q5_0); compute by AutoTrust AI Lab | autotrust/gpt-oss-120b-Fable-5-Distilled-GGUF (HF) |
| **Qwythos-9B-Claude-Mythos-5-1M** | Qwen3.5-9B | Full-parameter fine-tune; 1M context, uncensored, runs in 4GB; ~6.5GB, ~180 tok/s | Empero AI, Interfaze, Threads, Zerodu Jieshuo |
| **richardyoung/qwythos-9b-abliterated** | Qwythos-9B | abliteration secondary de-censoring variant | Ollama, HF |
| **Qwythos-27B** (teased) | Unknown | The next-gen "Mythos-class" 27B claimed on the Empero Roadmap | Empero |
| **Qwen3.6-14B-A3B-FableVibes** | Qwen3.6-35B-A3B "heretic" pruned | Pruning + QLoRA MoE; retains the "Claude feel"; ~11GB, ~140 tok/s | tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF (HF) |
| **clzoro/Qwen3.5-27B-Claude-distill** | Qwen3.5-27B (dense) | Fed on Opus CoT, strong literary / role-play; ~19GB, ~110 tok/s | HF |
| **Dzluck/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled** | Qwen3.5-2B | Ultra-lightweight for edge devices; ~2.1GB, ~250 tok/s; GGUF included | HF |
| **Jackrong series** (e.g. 27B Coder) | Qwen family | Trace-inversion processing of CoT, "most trusted" in the community | LocalLLaMA |

## A.2　Datasets (circulating, unverified)

| Dataset | Contents | Use | Risk |
|---|---|---|---|
| **armand0e/claude-fable-5-claude-code** | Fable 5 raw agent execution traces (de-privatized, in collaboration with Glint-Research), convertible to OpenAI format | Code / tool-calling distillation | ⚠️ Commercial-model traces, high ToS / legal risk |
| **ox-ox/mythos-character-distillation** | Claude's "speaking style and psychological traits": spontaneous metacognition, refusal of canned replies, self-correction without excuses | Persona / writing-style distillation | ⚠️ Same as above |

## A.3　Fable 5 / Distillation-Related Reports & Discussions (titles, unverified)

- "Anthropic adds distillation detection to Claude Fable 5" — BlockTempo (distillation detection → automatic fallback to Opus)
- "Anthropic's Fable 5 nearly sweeps every AI benchmark, and for the first time lists model distillation attacks as a blocked category" — inside.com.tw
- "Fable 5 Not Available? What Actually Happened to Claude's Newest Model" (mentions a U.S. export-control directive)
- "Anthropic Accuses Alibaba of Distilling Claude AI Model Capabilities" — Global Banking & Finance Review
- "Model Distillation: The Mechanics of Stealing an AI's Intelligence"
- "Has Claude been open-sourced? Qwythos-9B suddenly released! Uncensored, 1.04M context, runs in 4GB VRAM" — Zerodu Jieshuo
- "Another powerful open-source reasoning model is out! Empero releases Qwythos-9B-Claude-Mythos-5" — Threads
- "Qwythos-9B-Claude-Mythos-5-1M GPU Requirements: VRAM & Cheapest GPU" — Interfaze
- Reddit threads: r/LocalLLaMA (mynamasteph: "I would not trust any Claude distill…"), r/opencodeCLI (TomLucidor: "we kind need SFT/RL…"), r/huggingface (Qwable-v1 release)

## A.4　DGX Spark Hardware Sources (titles, unverified)

- "AI personal supercomputer built on the Blackwell architecture | NVIDIA DGX Spark" — NVIDIA (GB10 Grace Blackwell)
- "Powered by the GB10 chip, NVIDIA and several system makers launch a new kind of AI workstation" (iGPU shares lineage with GB100, fifth-generation…)
- "NVIDIA GB10 Grace Blackwell superchip — the desktop AI era" (Unified Memory Architecture, UMA)
- "Solved the DGX Spark, 102 stable tok/s Qwen3.5-35B-A3B" — Reddit (measured reference point)
- "DGX Spark, what models are you running?" — Reddit
- "Leadtek NVIDIA DGX Spark desktop AI supercomputer (GB10/128G/4TB SSD/DGX OS)", "DGX Spark Founders Edition"

## A.5　Prompt Engineering / Distillation Method Sources (accompanying Q6, titles, unverified)

- platform.claude.com — Prompting best practices (clarity, examples, XML, thinking, agentic)
- GitHub — Compilations of Claude Code system prompts across versions and their token counts
- arXiv — "Training Small Critic Agents" (Opus-4.6 teacher / detailed prompt)
- systemprompt.io — Daily Development Workflows (.md best practices, settings.json)
- claudemarketplaces.com/skills/.../knowledge-distillation — KD implementation guide
- Drew Breunig (dbreunig.com), LinkedIn "Lessons from Leaked Source", youmind (NotebookLM), uxplanet (Prompting Best Practices)

## A.6　Real, Usable Open-Source Bases & Datasets (corresponding to Chapter 2)

> Unlike the "circulating, unverified" snowflake names in A.1–A.2, the following are **genuinely public base families and commonly-used distillation datasets that you can verify on Hugging Face**. The swappable pipeline in Chapter 2 is in fact built on top of these — beneath almost any product claiming to be a "Claude distill," you'll find these bases plus these (or self-made) datasets.

| Real base family | Source | Distillation positioning |
|---|---|---|
| **Qwen (Qwen2.5 / Qwen3)** | Alibaba | Strong Chinese-English bilingual, full size range, community fine-tuning favorite ✅ |
| **Llama 3.x** | Meta | Widest ecosystem, GQA inference-friendly ✅ |
| **Mistral / Mixtral** | Mistral AI | Both dense and MoE, European open weights ✅ |
| **Gemma 2 / 3** | Google | Lightweight on-device, relatively permissive license ✅ |
| **GPT-OSS (20B / 120B)** | OpenAI | MoE open weights; A.1's 120B entry is based on this ✅ |
| **Phi-3 / Phi-4** | Microsoft | Small models, synthetic-data oriented ✅ |
| **DeepSeek-V2/V3 / R1** | DeepSeek | MLA saves KV; R1 reasoning traces often used as distillation material ✅ |

| Real public dataset | Contents | Use |
|---|---|---|
| **OpenHermes 2.5** | Million-scale multi-source instructions / conversations | General SFT ✅ |
| **UltraChat / UltraFeedback** | Large-scale multi-turn dialogue / preference annotations | SFT + preference alignment ✅ |
| **OpenOrca / SlimOrca** | GPT-generated CoT solution traces | Reasoning distillation ✅ |
| **Magpie** | Instruction pairs self-generated by aligned models | Zero-human-effort synthetic SFT ✅ |
| **Tülu 3 (with mixtures)** | AllenAI's open-sourced post-training recipe and data | Complete post-training reference ✅ |
| **distilabel pipelines** | Argilla's synthesis / rewriting / scoring pipelines | Self-made distillation data ✅ |

> Authenticity calibration at a glance
> **More credible (corresponding to real technology / products)**: Anthropic, Claude, Fable 5, Qwen, Llama, Mistral, Gemma, GPT-OSS, Phi, DeepSeek, Hugging Face, Ollama, GGUF, QLoRA, abliteration, OpenHermes / UltraChat / OpenOrca / Magpie / Tülu / distilabel, NVIDIA DGX Spark / GB10 / LPDDR5X UMA.
> **Unverifiable (very likely rumor or hallucination)**: the concrete existence and specs of the Mythos models, Qwythos-9B, Qwable-v1, FableVibes, Empero AI; "Fable 5 pulled by export controls days after launch," "4,659 traces," "distillation detection auto-downgrading to Opus 4.8." The specific tok/s figures, GB sizes, and "inherits Fable 5 tool-calling" claims in the A.1 table are all rumored assertions, never actually benchmarked.
