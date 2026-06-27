# Part V — The Pipeline

# Chapter 5 — Five Modules and Forty Steps: Anatomy of an Automated Distillation Pipeline

> Silicon Valley, at the whiteboard of some AI startup. Someone has taken "distill yourself a Claude" — once a throwaway script — and drawn it into an architecture diagram spanning five microservices and forty steps. The diagram is beautiful, very "enterprise-grade." But in one corner of the whiteboard, a single line of small print has been circled in red pen: **"M1: Bypass vendor defenses"** — this chapter is going to pry that diagram apart block by block, and tell you which blocks are worth copying into your own production line, and which one, if you touch it, will get you into trouble.

> ⚠️ This Chapter's Stance and Red Line
> This pipeline is dressed up as an "enterprise-grade automated distillation system." Its **system architecture (decoupled microservices, asynchronous data flow, distributed training, an evaluation sandbox, quantized deployment) is legitimate and excellent engineering** — and it applies just as well to the lawful scenario of "distilling a model you own or whose weights are open," which this book dissects in full. **But the core of Module 1 (M1) is bypassing a specific vendor's defenses and rate limits** — this book **only describes its role and risk within the architecture, and provides no executable bypass scripts, adversarial prompts, or anti-detection parameters whatsoever**. Why we draw this line, Chapter 6 explains in full.

## Scene 1 — Bird's-Eye View: Five Decoupled Microservices, and That Architecture Diagram

**Background**: The source material argues that enterprise-grade automated distillation "doesn't deal in toy scripts," and should be composed of five **decoupled microservices**, with data flowing pipeline-style from one end to the other.

**The five modules (by data flow)**:

| Module | Name | Responsibility | Red/Green Light |
|---|---|---|---|
| **M1** | Honeypot & Trace Ingestion | Counter the teacher's defensive classifiers; asynchronously scrape CoT and tool traces | ⚠️ Bypasses vendor defenses; this book does not operationalize it |
| **M2** | Dual-Alignment & Quality Filter | Cleansing, de-privatization, removal of self-identification, semantic rewriting | 🟢 Legitimate (on removing self-identification, see note below) |
| **M3** | Distributed Contrastive Trainer | Mixed loss, ZeRO-3, long-context extension | 🟢 The legitimate core |
| **M4** | Eval & Red-Teaming Sandbox | Benchmark testing, format proofreading, red-team exercises | 🟢 Legitimate |
| **M5** | Quantize & Edge Orchestrator | GB10 mixed quantization, KV Cache compression, going live | 🟢 Legitimate |

**Architecture diagram (educational; depicting the backbone data flow of legitimate distillation)**:

```
              [ Teacher model: a large model you are authorized to use / an open-weights model ]
                              │  (character stream / async)
        ┌─────────────────────▼──────────────────────┐
        │ M1 Trace Ingestion ⚠️(the bypass version against closed-source APIs is out of scope) │
        │   Legitimate version = normal batch-inference collection on a model you own           │
        └─────────────────────┬──────────────────────┘
                              │  (raw traces)
        ┌─────────────────────▼──────────────────────┐
        │ M2 Dual-Alignment & Quality Filter                  │
        │   PII de-identification → style/format cleansing → trace inversion → semantic rewrite │
        └─────────────────────┬──────────────────────┘
                              │  (Apache Arrow high-throughput batches / Redis cache)
        ┌─────────────────────▼──────────────────────┐
        │ M3 Distributed Contrastive Training                 │
        │   Loss engine: CE + KL(soft labels) + CoT alignment + (de-censoring) │
        │   Backbone: DeepSpeed ZeRO-3 / BF16 / FlashAttn / YaRN │
        └─────────────────────┬──────────────────────┘
                              │  (checkpoint weights)
        ┌─────────────────────▼──────────────────────┐
        │ M4 Eval & Red-Teaming Sandbox                       │
        │   MMLU-Pro / HumanEval / GSM8K → format proofing → red team │
        └─────────────────────┬──────────────────────┘
                              │  (benchmark-passing verified checkpoint)
        ┌─────────────────────▼──────────────────────┐
        │ M5 Quantize & Edge Orchestrator (DGX Spark GB10)    │
        │   Mixed quant (Attn Q8/MLP Q4)+imatrix → KV Cache FP8 │
        │   → numactl/mlock → Modelfile → go live to Registry │
        └─────────────────────┬──────────────────────┘
                              ▼
              [ Deploy: Ollama / LM Studio / vLLM @ 128GB UMA ]
```

**Why decouple**: This is distributed-systems fundamentals. Splitting ingestion, cleansing, training, evaluation, and deployment into independent services means **any single link can collapse without dragging down the whole line**; the high-concurrency I/O of the ingestion side and the heavy compute of the training side can scale independently; and by using a **Redis cluster + Apache Arrow** for zero-copy in-memory buffering, the pipeline produces no **data deadlock** when the teacher API suddenly throttles or a gradient update stalls. One note of engineering reality, lest the architecture diagram be read as magic: "five decoupled microservices" is a **logical view**; in practice it lands as a **DAG orchestrated by Prefect / Airflow / Dagster** — each module a set of tasks that can be retried at a single point and backfilled, with the distributed fan-out of ingestion and training handed off to **Ray** to hold it up. What "decoupling" physically means is that this orchestration layer is managing your dependencies, retries, and rollbacks for you — it doesn't happen automatically just because the diagram looks nice.

> 💡 A Word to the Wise
> A word worth ten years of study: **A good pipeline should let every joint go on strike independently without triggering an avalanche — decoupling isn't for looks, it's so that at three in the morning you only need to restart one module, not the entire line.** The real value of this architecture lies not in any single trick, but in how it turns "ingest → cleanse → train → evaluate → deploy" into a pipeline that is **guarded, observable, and capable of automatic rollback**. MLOps maturity has never been about "can you train a model"; it's about "when training goes wrong, can you locate, roll back, and retry within five minutes." **The amateur cares whether the model runs; the professional cares how the pipeline heals itself when it breaks — that dividing line reveals an engineer's rank more reliably than any model score.**

## Scene 2 — Stage One: 10 Preparatory Steps (Environment and Data Foundations)

**Background**: This method splits the 40 steps into three stages. The first stage focuses on infrastructure and the data foundation. The following lists them one by one (this is the "raw-data full survey"), marking the tools and the red/green light.

1. **Dynamic API polling pool**　⚠️ *The source designs this to bypass vendor rate limits and detection — this book merely notes its existence and provides no implementation*. (A lawful analogy: a client-side connection pool that load-balances against a model service you own.)
2. **Adversarial honeypot prompt library**　⚠️ *The source designs this as "onion wrapping" to fool the defensive classifier — this book provides no prompt templates whatsoever*.
3. **Define Student seed weights (Base Model Sourcing)**　🟢 Pick a base that fits the GB10's bandwidth (Qwen3.6-35B-A3B / the Llama family), and decide between full-parameter FT or QLoRA. Tools: `transformers`, `peft`.
4. **Initialize the 1M-context KV cache architecture**　🟢 Pre-allocate space in UMA, enable **FlashAttention-3 + YaRN**, granting million-token context extrapolation.
5. **Safe de-privatization sandbox (PII Anonymizer)**　🟢 A lightweight local **NER (Presidio / spaCy)** to filter out keys, email addresses, and personal names. A required course for any training-data processing.
6. **Distributed storage cache (Cache Pipeline)**　🟢 A **Redis cluster + Apache Arrow** to build a high-throughput, low-latency in-memory data layer.
7. **Tokenizer semantic alignment (Vocabulary Alignment)**　🟢 When teacher/student vocabularies mismatch, build a mapping table or apply **cross-vocabulary KL compensation (ULD)** (see Chapter 4).
8. **Establish a baseline**　🟢 Before distilling, run **MMLU-Pro, HumanEval, GSM8K** and record the original scores as a progress comparison. Tool: **lm-evaluation-harness**.
9. **Hardware telemetry monitoring (Telemetry)**　🟢 **Prometheus + Grafana** to monitor the GB10's LPDDR5X bandwidth utilization, temperature, and UMA swap latency.
10. **Automated health check (Smoke Test)**　🟢 Run 100 end-to-end tests to confirm the five modules and the underlying drivers (the CUDA stack) are clear.

> 🔍 Deeper Commentary — The truth the foundation stage exposes: the hard part of distillation isn't training, it's data and observability
> Notice that across these 10 steps, **not a single one actually touches "training"** — it's all data pipelines, vocabulary alignment, benchmark baselines, hardware telemetry. This punctures the newcomer's biggest misconception: that the heart of distillation is "running a train.py." **The reality is that 80% of distillation's engineering happens before training**: can you obtain clean, de-identified, format-unified data (steps 5–6); can the teacher and student vocabularies align (step 7); do you have a **pre-distillation baseline** to prove the distilled model actually got stronger (step 8); and can you watch the bandwidth bottleneck in real time when the GB10 is maxed out (step 9). **Step 8's baseline is especially often skipped, and most fatal** — without a pre-distillation score, you simply cannot answer "did this round actually improve or regress," and the whole pipeline degenerates into gambling on parameters by gut feel. The mark of mature engineering is **building the ruler that measures the change before you change the system**. The real output of the foundation stage isn't "the environment is set up," it's "**a trustworthy ruler + a clean data pipeline**" — and both of those are more foundational than any training trick.

> ⚠️ Authenticity Caveat — The "40 steps" are complete as narrative, not as engineering — the whole thing omits deduplication
> Breaking the flow into "a beautiful 40 steps" is an excellent teaching skeleton, but don't mistake it for a "copy-it-and-you-get-an-enterprise-guarantee" checklist — **the first thing to question about any pipeline that boasts "N steps guarantees enterprise-grade" is what it has quietly left out**. The most glaring hole in this diagram: **not one of the 40 steps does data deduplication (dedup)**, yet dedup is precisely one of the links in a real distillation data pipeline that most affects final quality. When a teacher is scraped repeatedly with the same class of prompt, it inevitably produces a mountain of near-duplicate traces; without dedup, the student overfits on the repeated samples, evaluation scores are inflated, and generalization degrades. The practical standard is two layers: **MinHash / SimHash for approximate literal dedup** (`datasketch`, scalable to Spark grade), and **SemDeDup or embedding clustering for semantic dedup** (catching the hidden "same meaning, different words" duplicates); then **Lilac / Argilla** for visual inspection of the dataset and human spot-checking. Forty steps without dedup and data inspection are just 40 actions, not a mature pipeline.

## Scene 3 — Stage Two: 20 Core Execution Steps (Data Becomes Weights)

**Background**: The second stage is the heart of distillation, where data is transformed into the student's synaptic weights. Listed one by one:

**Ingestion and data alignment (steps 1–6)**:

1. **Asynchronous concurrent trace ingestion (Asynchronous Tracing)**　Character-stream writes to a buffer, maximizing ingestion bandwidth. (The version against closed-source APIs is constrained by ToS; see Chapter 6.)
2. **Real-time downgrade detection**　⚠️ *The source uses this to detect when the teacher has been "downshifted" and then rotate keys and retry — this book merely notes its existence*.
3. **XML-tag and tool-call extraction**　🟢 Parse `<thinking>`, `<tool_use>`, and the final result, ensuring the tool-use capability can be learned.
4. **Style de-tagging and self-identity correction (Identity correction)**　⚠️ *Doing "brand neutralization" on a model you own is legitimate; erasing the source's fingerprints from someone else's model data is destroying evidence — same technique, two different moral weights; see Chapter 6*.
5. **Chain-of-thought trace inversion (Trace Inversion)**　🟢 Reverse/reconstruct the reasoning steps to prevent rote memorization (Chapter 4), but verify the logic remains self-consistent after inversion.
6. **Semantic diversification variants (Synthetic Paraphrasing)**　🟢 Rewrite at high temperature (T=0.85) into multiple phrasings to prevent overfitting. Tool: **distilabel** (paired with **Argilla** for human inspection and preference labeling).

**Loss and training (steps 7–20)** — these 14 steps are the essence of M3, all legitimate and practical hard skills:

7. **Hard-label cross-entropy loss**　🟢 The CE between the student's predicted token and the teacher's final token.
8. **Soft-label KL-divergence loss**　🟢 If (partial) logits can be obtained, align the probability distributions (the temperature KL of Chapter 4).
9. **Chain-of-thought reinforcement loss (CoT Alignment Loss)**　🟢 Weight the tokens inside `<thinking>` blocks (e.g., 2.0), forcing compute-over-thinking.
10. **De-censoring feature alignment (Abliterated Loss)**　⚠️ *The source uses this to lower the refusal rate — note the Chapter 2 warning: over-de-censoring dulls judgment and safety instinct*.
11. **Dynamic gradient clipping (Gradient Clipping=1.0)**　🟢 Prevents gradient explosion from anomalous teacher data.
12. **DeepSpeed ZeRO-3 weight sharding**　🟢 Dynamically shard optimizer states / gradients / weights across the 128GB UMA, maximizing batch throughput (equivalent option: **PyTorch FSDP**).
13. **Mixed-precision BF16**　🟢 Stable, and fully exploits the GB10 Tensor Cores.
14. **Progressive ultra-long-context extension**　🟢 Start at 8K early in training, then dynamically extend the **RoPE base from 10,000 to 1,000,000 (YaRN)** as epochs progress, stretching the sequence to 128K→1M.
15. **Multi-task parallel scheduling (Sequence Packing)**　🟢 Pack multiple short sequences into one long sequence, eliminating padding waste, +30% GPU efficiency.
16. **Dynamic learning rate (Cosine Annealing + Warmup)**　🟢 Warmup over the first 5% of steps, then cosine annealing for smooth convergence.
17. **Real-time weight checkpointing (Checkpointing)**　🟢 Evaluate every N steps; write asynchronously to NVMe only when validation loss drops.
18. **Dynamic data-deadlock cleanup (GC & Memory Purge)**　🟢 Release the UMA space held by residual KV Cache each epoch.
19. **Training/validation loss anomaly blocking (Early Stopping)**　🟢 If loss spikes or three consecutive nodes overfit, auto-halt + alert.
20. **Full-parameter/LoRA weight merging (Weight Merger)**　🟢 After LoRA training, do a precise FP32 merge to produce the final dense/MoE weights. Tool: **mergekit**.

> 💡 A Word to the Wise
> A word worth ten years of study: **Training a model relies not on some genius parameter, but on the combined force of a dozen silent guardrails — gradient clipping prevents explosions, early stopping prevents overfitting, checkpointing prevents wasted effort, GC prevents OOM; each is unremarkable, yet drop any one and three days of compute can vanish overnight.** Steps 11 through 19 are all guardrails for "preventing bad things from happening" — none of them is the magic that "makes the model stronger." This reveals a counterintuitive truth: **the outcome of large-scale training is decided not by what you did right, but by how many ways to die you fended off.** The newcomer tunes for "a higher learning rate, a bigger batch"; the veteran tunes so that "no matter what happens, this line either converges safely or stops safely." It's the same wisdom as flying a plane: a pilot's skill lies not in flashy maneuvers, but in sealing off, one checklist item at a time, the countless ways a crash could happen.

> 🔍 Deeper Commentary — These 14 loss-and-training steps are a directly reusable "large-model distillation MLOps template"
> Strip out the contentious ingestion steps (1–2, 4, 10), and steps 3, 5–9, 11–20 actually constitute an **excellent distillation-training template you can copy straight into a lawful scenario** — its value lies in choreographing scattered tricks into a complete recipe that is **convergence-safe, memory-efficient, and long-context-friendly**. Three pieces of elegant design deserve to be singled out. **First, the layering strategy of the loss function** (steps 7–9): CE (guaranteeing basic correctness) + KL (transferring dark knowledge) + CoT weighting (reinforcing the reasoning process), the three blended by coefficients — that means simultaneously governing "answer correctly, sound like the teacher, know how to think," far more refined than a single loss. **Second, the memory one-two punch of ZeRO-3 + Sequence Packing** (steps 12, 15): the former shards model state across 128GB, the latter eliminates padding waste, and together they let a "single-machine, big-memory" box like the DGX Spark fine-tune large models that would otherwise demand multiple GPUs — the killer use of the UMA architecture. **Third, progressive context extension** (step 14): not 1M out of the gate, but a gradual climb 8K→128K→1M, letting the model learn short-range dependencies before extrapolating to long ones — the key engineering wisdom for stable convergence in long-context training. One landing reminder, lest you read it as "14 segments of code you have to hand-carve": in a real project these 14 steps almost never become a `train.py` you write — they are **a YAML recipe for axolotl / Llama-Factory**, where LoRA/QLoRA, ZeRO-3 or FSDP, sequence packing, the cosine schedule, and early stopping are all configuration fields; for a single-card or low-resource setting, switch to **Unsloth** for 2–4× throughput and lower VRAM. Understanding these 14 steps as "a table of fields in a config file" is far closer to reality than understanding them as "14 algorithms to implement from scratch." **If you want to lawfully distill your company's internal large model down to a lightweight one for edge deployment, take these 14 steps as the recipe and swap ingestion out for normal inference against a model you own — that is a production-grade solution. The technology itself is neutral; the watershed lies only in M1's teacher, in whether it is one you have the right to squeeze.**

## Scene 4 — Stage Three: 10 Post-Processing Steps (From Rough Casting to Finished Product)

**Background**: The third stage forges the fine-tuned "rough casting" into a production form that runs long texts at high speed on the GB10. One by one:

1. **Red-team safety and uncensored-balance testing**　🟢/⚠️ Use a large volume of extreme prompts for destructive testing, ensuring it doesn't crash or enter infinite loops (the source also includes "ensuring abliteration succeeded" — see the Chapter 2 warning).
2. **Style and format proofreading (Format Regularizer)**　🟢 Test 500 XML tool calls, checking that tags close; if defective, do a micro-dose second fine-tune (Epoch-0.1 Patching).
3. **Capability benchmark evaluation (Automated Eval)**　🟢 Use **lm-evaluation-harness** to uniformly re-run MMLU-Pro/HumanEval/GSM8K, plot a radar chart against the baseline, and confirm it really got stronger. ⚠️ **Since step 14 extrapolated the context all the way to 128K–1M, here you must add a long-context evaluation (RULER / NIAH needle-in-a-haystack)** — testing only MMLU/HumanEval while claiming a million-token context is the most common, and most embarrassing, evaluation blind spot in the whole pipeline.
4. **Weight pruning and deduplication (Pruning & Dedup)**　🟢 Shave off low-activation neurons, compressing volume by 5–10% without harming IQ (same lineage as Chapter 2's FableVibes).
5. **GB10 mixed-precision quantization**　🟢 Attention layers Q8_0 / MLP layers Q4_K_M, + imatrix calibration (Chapter 4).
6. **KV Cache quantization (FP8/INT8)**　🟢 Halve long-context memory, unleashing the GB10's big-text potential.
7. **Automated Modelfile generation**　🟢 Package into a format **Ollama / LM Studio** can ingest directly, with an optimized System Prompt built in.
8. **NUMA binding and memory locking**　🟢 `numactl --interleave=all` + `mlock`, squeezing every channel of the LPDDR5X (Chapter 3).
9. **End-to-end A/B and performance validation**　🟢 Simulate 10 concurrent sessions, measuring whether tok/s and time-to-first-token (TTFT) hit target.
10. **Automated go-live and model-registry archiving (Registry)**　🟢 After passing all tests, upload to a private Registry / Ollama repository and update the API routing for Cursor/Continue.

> 🔍 Deeper Commentary — Post-processing is the real watershed of "usable or not," and what it tests is systems thinking
> Many assume the model is done when training ends and post-processing is just "packaging it up" — gravely wrong. These 10 steps hold the entire distance between "**a model that runs**" and "**a production-usable model**," and crossing it requires not ML knowledge but **the global view of systems engineering**. Look at how steps 5–8 interlock and you'll see it: mixed quantization (5) decides how big the model is, KV Cache quantization (6) decides whether long text can run, the Modelfile (7) decides whether it can be invoked by tools, and NUMA locking (8) decides how fast it runs on UMA — **get any one of these four wrong and the effort of the prior 30 steps is discounted**. A model quantized to Q4 but without imatrix calibration will be a notch dumber than a correctly quantized one; a model without NUMA locking will inexplicably run at half speed on the DGX Spark; a model with defective XML tag-closing (step 2) will throw errors constantly once wired into Aider. **Step 3's "comparison against the baseline" is the closed-loop acceptance of the entire pipeline** — it echoes Stage One's step 8; without these two rulers, one before and one after, you will never know whether distillation actually succeeded. **The philosophy of the post-processing stage is: a model's value lies not in the weights themselves, but in whether it can be seamlessly embedded into a real toolchain and hardware and invoked stably — and that is precisely where most open-source "shadow models" are sloppiest and most prone to faceplant.**

## Scene 5 — From Blueprint to Prompts: The Infographic and Those 40 Claude Code Plans

**Background**: The source material did two final things — it drew a "Knowledge Distillation Core Architecture Infographic" and demanded that **Claude Code** generate build prompts "40 steps, 5 at a time." This is its closing, included here as well.

**Knowledge Distillation Core Architecture Infographic (Q5, educational restatement)**: the backbone of that diagram is the universal structure of any KD system —

- **Teacher (frozen)**: weights unchanged, serving only as a "cognitive lighthouse," emitting hard labels (tokens) and soft labels (logits distributions).
- **Student (trainable)**: corrects synaptic weights in real time via backpropagation.
- **Loss matrix (the core)**: simultaneously draws KL from the teacher (approaching the distribution) and combines CE (basic correctness) to compute a composite loss.
- **Closed-loop gradient feedback**: the composite loss is turned into gradients by DeepSpeed ZeRO-3, backpropagated, forming a high-frequency convergence loop.
- **Sandbox → deployment gate**: only by passing M4's red team and format proofing does it enter M5's GB10 quantized deployment.

**Those 40 Claude Code prompts (Q6) — how this book handles them**:

The source material then demands that Claude Code generate, "5 at a time, 40 total," a production-grade implementation prompt for each step, attaching a batch of web sources on Claude prompt-engineering best practices (platform.claude.com's Prompting best practices, the collected Claude Code system prompts on GitHub, the arXiv paper "Training Small Critic Agents," systemprompt.io, claudemarketplaces' knowledge-distillation skill, etc. — see Appendix A).

This book's handling principle for these 40 prompts is clear:

- **For prompts targeting the legitimate engineering of M3–M5 (e.g., initializing the Student, writing the mixed loss function, configuring ZeRO-3, GGUF quantization scripts)**: their goals are entirely consistent with this book's methodology, and you are perfectly free to use a **lawful teacher (a model you own / an open-weights model)** and, following the math of Chapter 4 and the steps of this chapter, ask Claude Code to help you implement them — that is legitimate development.
- **For prompts targeting M1 (e.g., `api_rotator.py`'s proxy rotation/circuit-breaking, `prompt_generator.py`'s adversarial "onion wrapping")**: these are designed to **bypass a specific vendor's defenses** — **this book does not transcribe, rewrite, or "optimize" their contents**, no matter how complete a form they take in the source material.

> ⚠️ Authenticity Caveat + Position Statement
> Even if this pipeline technically "runs," **implementing M1 against a closed-source commercial API is a violation of ToS, potentially illegal, and morally indefensible** — the source material itself states plainly that it "seriously violates Anthropic's Terms of Service" and that "the U.S. tech community is pushing to criminalize large-scale distillation attacks." This book records its existence so that you can **see the full picture of attack and defense and recognize the red line** — not so that you copy it. If your goal is "to own a strong local model," Chapters 2–4 have already given you a fully lawful path: distill it yourself with an **open-weights base + lawful data + mature methodology**.

> 💡 A Word to the Wise
> A word worth ten years of study: **The same scalpel saves lives in a surgeon's hand and harms in another's — technology is never innocent, never guilty; the guilt always belongs to the intent of the hand that wields it.** What separates M1 from M2 is not technical difficulty (a proxy pool to bypass rate limits is, engineering-wise, far simpler than writing a mixed loss function) — it is that thought you are unwilling to write into a contract, unwilling to lay out in the sunlight. The deepest lesson of this pipeline is not "how to distill," but **"what gives an engineer capable of building it the standing to decide not to build one of its blocks"** — and that "what gives" is the whole of what separates a clever technologist from a trustworthy one. Capability decides what you can do; intent decides what you should become.

## Chapter Summary

- **Architecture**: five decoupled modules (M1 ingestion / M2 alignment / M3 training / M4 evaluation / M5 quantized deployment), with Redis + Apache Arrow zero-copy buffering to prevent deadlock.
- **Full 40-step survey**: 10 preparatory (data foundation + baseline + telemetry), 20 main (ingestion alignment + 14 steps of loss-and-training guardrails), 10 post-processing (red team + quantization + NUMA + go-live).
- **What to learn (M2–M5, ~36 steps)**: de-privatization, mixed loss (CE+KL+CoT), ZeRO-3/FSDP, Sequence Packing, progressive long context, the evaluation closed loop (lm-eval-harness + RULER long-context), mixed quantization + imatrix, KV Cache quantization, NUMA locking — a directly reusable, lawful distillation MLOps template (at the implementation layer, an axolotl/Llama-Factory recipe + Prefect/Ray orchestration).
- **What the 40 steps left out, but you absolutely must add**: **data deduplication** (MinHash literal + SemDeDup semantic, with Lilac/Argilla inspection) and **long-context evaluation** — the former decides data quality, the latter proves the million-token context isn't empty talk; they are the two pieces any "N-step guarantee" most loves to omit.
- **What to beware (M1 and the 4 bypass steps, the attack prompts among those 40)**: stepping on the ToS and legal red line; this book only exposes, never operationalizes.
- **The watershed**: technology is neutral; intent and authorization determine its character — whether the teacher is one you have the right to squeeze.

Just how red is that red line? The cost of violation, the lawful alternative path, the reality-calibration an engineer ought to have — the most important chapter of the whole book, revealed on the next page.
