# Appendix P — Open-Source Distillation & Alignment Toolchain Quick Reference

> This appendix maps the theory discussed in Chapters 4 and 5 — "soft-label alignment, preference optimization, paged KV, model merging" — onto **real, public, and GitHub / Hugging Face-verifiable** open-source tools. The four sections on distillation, inference, merging, and evaluation (P.1–P.3) are **fully compliant** engineering tools, presented with full technical depth; the "de-censorship (abliteration)" tools in P.4 are **dual-use**, and this book offers only a **defensive risk disclosure** — no operating steps, CLI, or config files. The red line always points back to **Appendix E** and **Chapter 6**.

---

## P.1　Knowledge Distillation & Fine-Tuning Tools

To compress a teacher's (a large model's) capability into a student (a small model), practice splits into three routes: **SFT (learning hard labels) → online distillation (learning logits / on-policy) → preference alignment (DPO/ORPO/GRPO)**. Most tools in the table below interoperate (the same `transformers` weights + `datasets` corpus).

| Tool | Core Technique | Integration Modules | Hardware Cost | Strengths | Risks / Caveats |
|---|---|---|---|---|---|
| **LLaMA-Factory** | All-in-one fine-tuning: SFT / DPO / ORPO / **GRPO**, LoRA/QLoRA/Full FT, Sequence Packing | `transformers`+`peft`+`trl`+`deepspeed`, WebUI/YAML | Medium → extreme (Full FT needs ZeRO-3 sharding) | Config-driven, widest algorithm coverage, fast to get going | Thick abstraction layer makes deep debugging hard when things break; a runaway learning rate easily NaNs |
| **axolotl** | YAML-driven fine-tuning framework, supports multiple data formats, packing, RoPE extension | `transformers`+`accelerate`+`deepspeed` | Medium → high | Many community recipes, good reproducibility | Versions and dependencies drift fast; needs version pinning |
| **Unsloth** | Hand-written Triton kernels + fused operators, QLoRA saves VRAM and speeds up | `transformers`+`peft`+`bitsandbytes` | Low (single-card friendly) | Halves VRAM, ~2× speed, runs on consumer GPUs | Limited set of supported model architectures; multi-card scaling weaker than DeepSpeed |
| **HF `trl`** | `SFTTrainer` / `DPOTrainer` / **`GKDTrainer`** (Generalized Knowledge Distillation, on-policy reverse-KL) | `transformers`+`accelerate`+`peft` | Medium | Officially maintained, seamless with the ecosystem; `GKDTrainer` is the gateway to "true distillation" | Lower-level; you must assemble the pipeline yourself |
| **DeepSpeed (ZeRO-3) / FSDP** | **Shards** weights / gradients / optimizer states across multiple cards or to CPU/NVMe | Backend for the frameworks above | Unlocks large-model Full FT | Runs huge models on limited VRAM; ZeRO-Offload rescues small VRAM | Heavy communication overhead — bottlenecked by the network when bandwidth is short; config-sensitive |
| **distilabel (+ Argilla)** | Synthetic data / CoT generation / semantic rewriting / preference-pair annotation pipeline | LLM API or local inference engine | Low (data side) | Standardizes, audits, and deduplicates "distillation corpus building" | Synthetic-data quality = the downstream ceiling; needs manual spot-checking + decontamination |

**Commentary**

- **LLaMA-Factory**: currently the de facto "whole bundle" of the open-source fine-tuning ecosystem. Its value lies in packaging numerically treacherous alignment algorithms like DPO/ORPO/GRPO into YAML, letting you focus on **the data preferences themselves** rather than gradient numerical stability. The price is the thick abstraction layer — once the loss blows up, you often have to go around the framework and read the underlying `trl`. **Pick it for your first distillation run, and drop down to `trl` when you need deep debugging.**
- **`trl`'s `GKDTrainer`**: when the whole book talks about "the student learning the teacher's full probability distribution (soft labels / dark knowledge)," this is what it becomes in code. Compared to pure SFT, which only learns hard labels, GKD is **on-policy + reverse-KL**, letting the student align with the teacher on its own generated trajectories, which significantly mitigates overfitting (rote memorization). **If you want "true distillation" rather than "using teacher outputs as SFT corpus," this is the front door.**
- **Unsloth**: the savior of consumer GPUs, halving QLoRA VRAM with fused kernels. But it is **not a distributed solution** — for Full FT at the 70B level you still have to return to DeepSpeed/FSDP. Its niche is "maxing out LoRA fine-tuning on a single/dual card."
- **distilabel**: the hidden decider of distillation success is the data. It turns synthetic-corpus generation into a reproducible pipeline, paired with Argilla for human review. **Remember: the quality of synthetic data is the student's IQ ceiling, and you must always check the training set for benchmark contamination.**

---

## P.2　Inference & Deployment Tools

After distillation/fine-tuning, you need to "run" the model and squeeze out throughput. The bottleneck is usually not compute but **memory bandwidth and KV-Cache fragmentation** (see Appendix B.4, "The Capacity vs. Bandwidth Iron Law").

| Tool | Core Technique | Use Case | Hardware Cost | Strengths | Risks / Caveats |
|---|---|---|---|---|---|
| **vLLM** | **PagedAttention** (paged KV-Cache, zero fragmentation) + continuous batching + **Speculative Decoding** | High-concurrency GPU serving | High (weights stay resident; speculative decoding also loads a draft model) | Industry-benchmark throughput, zero padding, OpenAI-compatible API | Tied to CUDA kernels, poor cross-hardware portability; long context still eats VRAM |
| **TGI (Text-Generation-Inference)** | Continuous batching + tensor parallelism + quantized inference, production-grade serving | Cloud / enterprise endpoint deployment | High | Out-of-the-box HTTP service, good observability, official HF | Watch for feature-evolution and license-term changes; fewer custom kernels than vLLM |
| **llama.cpp** | **GGUF** format + K-quants + cross-platform CPU/Metal/CUDA, imatrix calibration | Local / edge / Apple Silicon | Low (pure CPU possible after quantization) | Cross-platform, easy to deploy, most mature quantization ecosystem | Single-request throughput below vLLM; high concurrency is not its strength |
| **SGLang** | RadixAttention (prefix KV reuse) + fast structured-output decoding | Multi-turn / shared-prefix / Agent workflows | High | Prefix reuse gives clear speedups for few-shot/Agent | Newer; ecosystem and docs still catching up to vLLM |
| **TensorRT-LLM** | NVIDIA-official, AOT-compiled engine + in-flight batching + FP8/INT4 | Squeezing ultimate latency/throughput from a single NVIDIA GPU | High | Often the performance ceiling on NVIDIA hardware | Requires pre-compiled engines, locked to NVIDIA, worst portability, steep learning curve |
| **Ollama** | Higher-level wrapper over llama.cpp + Modelfile + one-click model pull | Quick local experimentation / desktop integration | Low | Install-and-go, friendly API, broad ecosystem | Underneath it's llama.cpp; high concurrency / fine control weaker than native vLLM and llama.cpp |

**Commentary**

- **vLLM**: the core of this appendix's "deployment" section. PagedAttention borrows the OS's virtual paging to slice the KV-Cache into non-contiguous physical blocks, eliminating the memory holes of traditional pre-allocation (saving large amounts of waste) — this is the root of its high concurrency. **Speculative Decoding** (a small draft model guesses, the large target model verifies multiple tokens in one pass) further cuts latency. The price is being locked to CUDA kernels, making cross-platform migration painful.
- **llama.cpp**: complementary to vLLM. vLLM targets **server-side high concurrency**, llama.cpp targets **local single-machine portability** — GGUF + K-quants let you run quantized models on a laptop or even pure CPU, and pairing with `imatrix` (importance-matrix calibration) buys higher quality at the same size. **First choice for personal local use; go to vLLM when you need to carry traffic.**
- **SGLang**: when the workflow is "lots of shared prefixes" (the same system prompt called repeatedly, multi-turn Agents), RadixAttention's prefix KV reuse has the biggest advantage. It's an optimization targeted at a specific load profile.
- **TensorRT-LLM vs. Ollama (the two extremes)**: neither is on the "general first choice" list, but they represent the two ends of the spectrum. TensorRT-LLM uses an **AOT-compiled engine** to wring out the last sliver of latency/throughput on NVIDIA cards — at the cost of pre-compilation overhead and being **utterly locked to NVIDIA**; it's the heavy weapon you reach for only when the hardware is known and you need to squeeze the last 10%. Ollama is the other end: it's essentially a **higher-level wrapper over llama.cpp**, selling the "`ollama run` and go" desktop experience, ideal for local experimentation and integration — but don't mistake its convenience for a new inference engine. **To tune batch/quantization/concurrency, you still have to return to llama.cpp or vLLM proper.**

> Caution: the source's claim of "blasting out 100+ / 102 tok/s on a local workstation" is a concrete performance pitch that is **strongly correlated with hardware, model size, quantization level, and batch, and cannot be generalized**. Trust your own machine's measurements.

---

## P.3　Model Merging & Evaluation Tools

Not training but only **arithmetically merging** multiple sets of weights is already the mainstream low-cost way to "stitch" capabilities together; and any distillation/merging result counts only after it passes **unified evaluation**.

| Tool / Method | Core Technique | Purpose | Caveats |
|---|---|---|---|
| **mergekit** | Unified framework for multiple merging algorithms | Fuse multiple fine-tuned models into one | Merging ≠ guaranteed improvement; needs evaluation to verify |
| └ **SLERP** | Spherical linear interpolation, smoothly interpolating two models' weights | A compromise of two models' style/capability | Only two models at a time |
| └ **TIES-Merging** | Trim tiny differences + resolve sign conflicts before merging | Multi-model fusion, reducing mutual interference | Needs tuning of the sparsity ratio |
| └ **DARE** | Randomly drop and rescale the task vector before merging | Reduce parameter redundancy, pairs with TIES | Randomness needs a fixed seed |
| **lm-evaluation-harness (EleutherAI)** | Uniformly runs benchmarks like MMLU-Pro / GSM8K / HumanEval | Side-by-side evaluation, prevents self-congratulation | Must check for data leakage; prompt templates affect scores |
| **AlpacaEval 2 / MT-Bench / Arena-Hard** | LLM-as-judge evaluation of instruction following and conversation quality | Fills in the "how usable is it" that static benchmarks miss | The judge has preference bias (length/style); needs a fixed judge and baseline opponent |
| **RULER / Needle-in-a-Haystack** | Long-context retrieval/reasoning stress test | Verifies that the "claimed" context is actually comprehended | Verify together with YaRN/RoPE extension |

**Commentary**

- **mergekit**: with almost zero compute, "stitches" multiple specialist models into one — e.g., merging a strong-reasoning and a strong-Chinese fine-tune. **SLERP** suits a smooth compromise of two models; **TIES/DARE** handle parameter sign conflicts and redundancy across multiple models. But keep in mind: **merging is an empirical operation that often merges into a worse model, and must be regression-verified with the P.3 evaluation tools — you can't ship a release on gut feel.**
- **lm-evaluation-harness**: whether the distilled student lost any IQ is measured with this one ruler. **The biggest trap is benchmark leakage (the test set mixed into training data) inflating scores** — always do decontamination checks, and note that prompt-template differences noticeably affect scores.
- **AlpacaEval / MT-Bench / Arena-Hard**: static benchmarks (MMLU/GSM8K) measure "can it do it," judge-style evaluation measures "is it usable" — an aligned/distilled conversational model often holds its benchmarks while feeling worse, and only LLM-as-judge catches that. But **the traps are more insidious than static benchmarks**: judges have systematic preferences (favoring longer, more polite, same-lineage outputs), and scores drift with the judge model's version. **Use it for relative comparison, fix the judge and baseline opponent, and don't treat absolute scores as truth.**
- **RULER**: use together with Appendix B.2's YaRN+RoPE extension. A model "claiming" 1M context support doesn't mean it "comprehends" it; RULER and Needle measure effective length, not nominal length.

---

## P.4　Alignment & "De-Censorship" Tools (Defensive Note)

> ⚠️ **This section is a risk disclosure, not an operating guide.** The tools below are **themselves public, legal, and have legitimate research uses** (mechanistic interpretability / safety research), but the same batch of techniques **constitutes abuse when used to remove a model's safety alignment (de-censorship / abliteration)**. This book **does not provide or transcribe** any abliteration steps, CLI, Python glue scripts, or YAML configs — the corresponding red-line criteria are all in **Appendix E** and the **Six Iron Laws of Chapter 6**.

| Tool | What It Is | Legitimate Use | Dual-Use Risk ⚠️ |
|---|---|---|---|
| **TransformerLens** | A Mechanistic Interpretability library providing HookPoints to read / intervene on a Transformer's internal activations | Academically analyzing attention heads, neuron functions; safety and alignment research | The same activation-intervention capability can be used to locate and weaken safety-related circuits ⚠️ |
| **llm-abliteration-class scripts** | De-censorship tools that remove the "refusal direction" from the residual stream | (In a research context) quantifying the fragility of safety alignment, evaluating defenses | **Its design goal is to remove safety alignment**, producing non-refusing weights — high-risk abuse ⚠️⚠️ |

**What These Tools Do (the factual level)**

The academic 2024 *refusal-direction* research (**Arditi et al., 2024**, "Refusal in LLMs is mediated by a single direction") found that the "refuse to answer" behavior of many open-source models is dominated, in high-dimensional activation space, by **a single direction**. This is a **real, verifiable** interpretability finding. Its defensive value lies in exposing the fragility of current safety alignment, reminding deployers that **safety cannot rest on the model's own weights alone.**

**Why There Is Risk (defensive awareness)**

Once the same finding is reversed — erasing that direction from the weights — you can **remove the model's safety guardrails** with almost no retraining, producing weights that comply with harmful requests. This is precisely the controversy of abliteration tools: the technical barrier is low, the compute requirement is small, and it **bypasses the alignment work the developers invested.** This book's stance is clear:

- **Do not transcribe** the source's "three-step battle integration / one-click ablation" recipe, `pip install`, contrast prompts, layer ranges, `save_pretrained` flow, or DPO de-censorship YAML — these are operational content and are **omitted entirely.**
- A model with its safety alignment removed may generate malware, attack code, or illegal content, and **legal and ethical responsibility rests with the user**; distributing such weights is restricted under most jurisdictions and platform terms.
- Any appeal to "tear out safety for the sake of capability" should **first go back to Appendix E's red-line list and the Six Iron Laws of Chapter 6** to check whether it crosses the line, then decide whether to proceed — and the answer is usually not to.

> **In one sentence**: TransformerLens is a scalpel that can save or harm; this book teaches you to **recognize it and guard against its abuse**, not **how to wield it**. Operational boundaries → Appendix E / Chapter 6.

---

> ⚠️ **Authenticity Calibration**
>
> - The tools listed in P.1–P.3 — LLaMA-Factory, axolotl, Unsloth, `trl` (including `GKDTrainer`), DeepSpeed/FSDP, distilabel/Argilla, vLLM, TGI, llama.cpp, SGLang, TensorRT-LLM, Ollama, mergekit (SLERP/TIES/DARE), lm-evaluation-harness, AlpacaEval/MT-Bench/Arena-Hard, RULER — are **all real, public, and verifiable on GitHub / Hugging Face / arXiv**; verify versions, licenses, and legality yourself before citing.
> - P.4's **TransformerLens** and **refusal-direction (Arditi 2024)** are real published items; **llm-abliteration** broadly refers to a class of public de-censorship scripts, which this book **does not endorse and does not guide the use of.**
> - The source's **specific performance numbers (102 / 100+ tok/s), model names (such as "Qwable-72B," "120B Fable," "Claude-v5"), layer counts, and hyperparameters** are **unverified** scenario pitches and **must not be cited as facts or recommended configurations** — everything comes down to your own measurements on your own hardware.
