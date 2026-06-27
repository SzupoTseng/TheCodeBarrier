# Appendix B — Glossary of Terms & Tools

> A "decoder" that gathers every technical term in the book into one table by topic. These are **all real, public technologies**, unlike the "to-be-verified model names" in Appendix A — master them and you'll be able to understand exactly what any distillation / local-deployment pitch is really saying.

## B.1　Core Distillation Concepts

| Term | One-line explanation | Chapters where it appears |
|---|---|---|
| **Knowledge Distillation (KD)** | Having a large teacher model pass its capabilities to a small student model | Throughout |
| **Teacher / Student** | The large model being learned from / the small model doing the learning | Chapters 4, 5 |
| **Hard Target** | Learning only the final answer | Chapter 4 |
| **Soft Target / Logits** | Learning the teacher's full probability distribution (the "dark knowledge") | Chapter 4 |
| **KL-Divergence** | A mathematical tool measuring how far apart two probability distributions are | Chapters 4, 5 |
| **Chain-of-Thought (CoT)** | The process of a model reasoning step by step | Chapters 2, 4, 5 |
| **CoT alignment loss** | Weighting the thinking process to force the student to "think clearly before answering" | Chapters 4, 5 |
| **Trace Inversion** | Shuffling / reversing reasoning steps to stop the student from rote memorization | Chapters 2, 4 |
| **Synthetic Paraphrasing** | High-temperature sampling to rewrite data, broadening generalization and preventing overfitting | Chapters 4, 5 |
| **Overfitting** | Rote-memorizing the training data and collapsing on a new scenario | Chapter 4 |
| **abliteration** | Removing the "refusal" direction from the weights to de-censor | Chapter 2 |

## B.2　Training & Fine-Tuning

| Term | Explanation |
|---|---|
| **SFT (instruction fine-tuning)** | Supervised fine-tuning on "instruction → response" pairs |
| **QLoRA** | An efficient fine-tuning method using quantization + low-rank adaptation (saves VRAM) |
| **Full-parameter fine-tuning (Full FT)** | Updating all of a model's weights (most thorough, most resource-intensive) |
| **DeepSpeed ZeRO-3** | Sharding weights / gradients / optimizer states to run large models with less memory |
| **Mixed precision (BF16/FP16)** | Using lower precision to speed up training while preserving stability |
| **Gradient clipping / cosine annealing / Early Stopping** | Three guardrails that prevent training collapse and stabilize convergence |
| **YaRN + RoPE Base extension** | The technique for progressively stretching a model's context window from 8K to 1M |
| **FlashAttention-3** | Efficient attention computation, saving memory and accelerating long sequences |

## B.3　Quantization & Deployment

| Term | Explanation |
|---|---|
| **Quantization** | Compressing weights from high precision to low precision, trading for size / speed |
| **GGUF** | The mainstream local-deployment model format (llama.cpp / Ollama / LM Studio) |
| **Q4_K_M / Q5_K_M / Q8_0** | Common quantization levels; larger number = more precise / larger, K_M = mixed precision |
| **Mixed quantization** | Keeping attention layers at high precision, compressing MLP layers to low precision |
| **KV Cache** | Context memory; the real bottleneck for long text |
| **KV Cache quantization (FP8/INT8)** | Compressing context memory to unlock long-text potential |
| **MoE (Mixture of Experts)** | Many experts, only a few activated each time; a savior for bandwidth-limited machines |
| **Activated parameters vs total parameters** | The former determines speed, the latter sets the ceiling on intelligence |

## B.4　Hardware (NVIDIA DGX Spark)

| Term | Explanation |
|---|---|
| **DGX Spark** | NVIDIA's desktop "personal AI supercomputer" |
| **GB10 (Grace Blackwell)** | The superchip the DGX Spark carries |
| **UMA (Unified Memory Architecture)** | CPU and iGPU share the same pool of memory |
| **128GB LPDDR5X (~600 GB/s)** | Huge capacity, but bandwidth below discrete-GPU HBM/GDDR |
| **The capacity-vs-bandwidth iron rule** | Capacity decides "whether it can run," bandwidth decides "how fast it runs" |
| **numactl --interleave=all / mlock** | Lock memory and interleave channels to squeeze out UMA bandwidth |

## B.5　Training, Fine-Tuning & Distillation Tools

| Tool | Role | Chapters |
|---|---|---|
| **Hugging Face `transformers` / `datasets`** | The foundational libraries for models and data | Chapters 2, 4, 5 |
| **`peft`** | LoRA / QLoRA and other parameter-efficient fine-tuning | Chapters 2, 5 |
| **`trl` (SFTTrainer / GKDTrainer / DPO)** | Supervised fine-tuning, online distillation, preference alignment | Chapters 2, 4 |
| **Axolotl / LLaMA-Factory / Unsloth** | Out-of-the-box fine-tuning frameworks (Unsloth saves VRAM) | Chapters 1, 2 |
| **DeepSpeed (ZeRO-3) / FSDP / `accelerate`** | Distributed training, weight sharding | Chapters 4, 5 |
| **`bitsandbytes` (NF4)** | The quantization base for QLoRA, paged optimizer | Chapters 2, 4 |
| **FlashAttention-3** | Efficient attention, memory-saving for long sequences | Chapters 3, 5 |
| **distilabel (Argilla)** | Synthetic data / paraphrasing / CoT generation | Chapters 4, 5 |
| **mergekit** | LoRA / weight fusion and model merging | Chapter 5 |
| **MiniLLM / GKD (implementation)** | on-policy / reverse-KL distillation | Chapter 4 |
| **TransformerLens / heretic / refusal_direction** | abliteration (de-censoring) activation analysis | Chapter 2 |
| **Microsoft Presidio / spaCy NER** | Data de-privatization (PII de-identification) | Chapter 5 |
| **Redis cluster + Apache Arrow** | High-throughput zero-copy data caching layer | Chapter 5 |
| **Prometheus + Grafana** | Hardware / training telemetry monitoring | Chapter 5 |

## B.6　Quantization Tools

| Tool | Role |
|---|---|
| **llama.cpp (GGUF / K-quants / `quantize`)** | Mainstream local quantization and inference |
| **imatrix (importance matrix)** | Pre-quantization calibration, higher quality at the same size |
| **AutoGPTQ / AutoAWQ** | GPTQ / AWQ post-training quantization |
| **ExLlamaV2** | High-speed quantized inference engine |
| **FP8 / INT8 KV-Cache quantization** | Compressing context memory (vLLM, llama.cpp) |

## B.7　Local Inference & Deployment

| Tool | Role |
|---|---|
| **Ollama** | Local model management / inference backend, supports dynamic loading of multiple models |
| **LM Studio** | Graphical local-inference frontend |
| **llama.cpp / vLLM / SGLang** | Inference engines (vLLM/SGLang high-throughput, support speculative decoding) |
| **NVIDIA NIM / DGX OS stack** | The DGX Spark's built-in inference stack |
| **Open WebUI / Cursor / Continue** | Frontend interfaces, support model switching and a "dual-model showdown Arena" |
| **OpenHands (OpenDevin) / Aider** | Open-source coding agents that can connect to local models |

## B.8　Evaluation Benchmarks

| Benchmark | What it tests |
|---|---|
| **MMLU-Pro** | Multi-domain knowledge and reasoning |
| **HumanEval** | Code-generation ability |
| **GSM8K** | Math word-problem reasoning |
| **lm-evaluation-harness (EleutherAI)** | A unified tool for running the various benchmarks |
| **RULER / Needle in a Haystack** | Long-context retrieval ability (verifying "how long it can actually read and understand") |
