# Part III — The Taming Ground

# Chapter 3 — Taming Dragons on the DGX Spark: Hardware Roofline & Selection

> Silicon Valley. On the reseller's invoice, that **NVIDIA DGX Spark** carries a price that makes you wince. It's small as a Mac mini, yet it bills itself a "personal AI supercomputer." The moment the fan spins up, the real question isn't "how big a model can it run." It's something far sharper: **"I paid a steep premium for this 128GB — what on earth should I run on it so I'm not squandering it?"**

> This is the chapter with the highest technical density, and the firmest footing. GB10, unified memory (UMA), GGUF, Roofline — all real engineering. The selection verdict is a common line of reasoning, but the **bandwidth logic behind it is verifiable hardware common sense** — the very ruler that punctures the "myth marketing" of the previous chapter.

## Scene 1 — Meeting the Beast: Reading Its Temper Through the Roofline

**Background**: The hardware is the **NVIDIA DGX Spark (GB10 chip, not the data-center-grade GB200/GB100)**. Let's nail the specs down first, because **every step of selection is derived from these few numbers**:

| Spec | Value | Confidence |
|---|---|---|
| Chip | GB10 Grace-Blackwell superchip (20-core Arm CPU + Blackwell iGPU) | 🟢 Official |
| Memory | **128GB LPDDR5X unified memory (UMA)**, CPU and iGPU physically share one pool, no PCIe in between | 🟢 Official |
| Memory bandwidth | **~273 GB/s** (LPDDR5X class) | ⚠️ Common nominal figure; a theoretical peak of ~301 GB/s derived from bus width is also cited — **don't take either as exact gospel** |
| Compute | ~1 PFLOP **FP4 (NVFP4)** sparse peak (5th-gen Tensor Cores) | ⚠️ The vendor's "sparse FP4" number; dense/measured will be considerably lower |
| Native quant | Tensor Cores natively support **FP4 / NVFP4 / FP8 / BF16** | 🟢 Architectural feature |

**Note that bandwidth figure, which many second-hand articles write up as "600 GB/s" or even higher — that's wrong.** The DGX Spark uses LPDDR5X, not the HBM3e of data-center cards; its bandwidth is in the "two-or-three-hundred GB/s" range, not HBM's "TB/s." **Get this number wrong by 2×, and every tok/s estimate below is off by 2×** — so calibrate to the ~273 GB/s class first. As for NVFP4 — it's the 4-bit floating format Blackwell introduced (E2M1 plus two-level micro-block scaling), using per-block scale to drag 4-bit quality up close to FP8, and it's precisely the confidence behind daring to use Q4 in the selection below (details saved for the distillation math of Chapter 4).

**The problem**: If you don't understand its temper, you'll get dragged along by claims like "running Claude on 4GB of VRAM" — squandering the one thing it's actually worth paying for. And the tool for seeing through its temper is a model every engineer must know — the **Roofline**.

**The two phases of LLM inference each slam into a different wall**:

- **Prefill (processing your input)**: it feeds the prompt's hundreds-to-thousands of tokens in all at once for a big matrix multiply — the same weights are reused across this whole batch of tokens, so it's **compute-bound** — it feeds on FLOPS.
- **Decode (generating token by token)**: autoregressive, one token at a time; for each token generated, it must **read the participating weights out of memory in full**, and those weights serve only "this one" token — read in, compute once, discard. **Memory-bandwidth-bound** — it feeds on GB/s.

**Why does prefill hit the compute wall while decode hits the bandwidth wall?** In a word: **arithmetic intensity (the number of floating-point operations you get per byte of weight read) differs by worlds.** Prefill processes N tokens at once, so one weight read is shared across N tokens, and arithmetic intensity scales linearly with N, charging easily into the "compute-bound" region; Decode has batch=1, so one weight read computes the multiply-add for just one token — arithmetic intensity approaches the theoretical floor (~2 FLOP per parameter), so most of the GPU's trillion-op compute sits idle and the bottleneck is locked dead at "getting the weights out of memory." **This is why, when you run a single conversation on the DGX Spark, GPU utilization is often only twenty or thirty percent yet you're already at top speed — it isn't underpowered, it's waiting on memory.**

**This yields an iron law you can use to estimate speed directly**:

```
Decode speed (tok/s) ≈  memory bandwidth (GB/s)  ÷  weights read per token (GB)
```

Plug in the DGX Spark's ~273 GB/s (this is the bandwidth "ceiling"; subtract attention KV reads and framework overhead and real-world measurements usually take another 60–80% of it):

- **70B dense model @ Q4_K_M** (each token reads **all** ~40GB of weights): 273 ÷ 40 ≈ **~6.8 tok/s in theory, ~5 tok/s measured** — squeezing out one word at a time, unusable.
- **120B-A4 MoE @ Q4** (each token reads only "the ~4 activated experts at ~2–3GB + the per-layer resident attention/shared weights at ~3–4GB" ≈ 6GB): 273 ÷ 6 ≈ **~45 tok/s in theory, 35–50 tok/s measured** — smooth.

See the trick? **On the same machine, the 70B dense crawls at ~5 tok/s, while the "larger-total-param" 120B MoE runs nearly 9× faster** — the difference isn't total parameters, it's "how many bytes you actually have to move per token." **This one equation lets you compute how fast a model will run before you even hit download**, instead of regretting it only after 70GB has finished downloading.

> 🔍 Deeper Commentary — The Roofline's "Ridge Point": One Number That Reveals a Machine's Character
> The entire essence of the Roofline model can be distilled into a single coordinate — the **ridge point = peak compute ÷ peak bandwidth**, in units of "FLOP/byte." It's the machine's "character divide": if a workload's arithmetic intensity is **above** it, you hit the **compute wall** (the roof's flat top); **below** it, you hit the **bandwidth wall** (the roof's sloped side). Plug in the DGX Spark: ~1 PFLOP(FP4) ÷ ~273 GB/s ≈ **~3600 FLOP/byte**. Which means: **any workload with arithmetic intensity below 3600 is choked by bandwidth.** And what's decode's arithmetic intensity? At batch=1, about **2–4 FLOP/byte** — fully **three orders of magnitude** below the ridge point. In other words, **token-by-token LLM generation, on this machine (and on all UMA / Apple Silicon), is an extreme memory-bound workload getting ground into the dirt by the bandwidth wall**, and that 1 PFLOP of compute is nearly ornamental to it. Grasp this and you understand the root of three things at once: ① **why speeding up means "reducing bytes read per token"** (quantization, MoE, KV-cache quantization) rather than "buying more compute"; ② **why raising the batch (serving multiple requests at once) can boost total throughput almost linearly** — multiple tokens share one weight read, pushing arithmetic intensity toward the ridge point (this is exactly the value of vLLM's continuous batching); ③ **why speculative decoding (Scene 3) works** — it lets the big model "read weights once and verify multiple draft tokens in parallel," which is also raising arithmetic intensity at heart. **The ridge point is a ruler that measures whether your bottleneck is compute or bandwidth; and for personal LLM inference, the answer is almost always bandwidth.**

> 🧠 The Hardware Iron Law You Must Weld Into Your Brain
> **Capacity decides whether you "can run it," bandwidth decides "how fast it runs," and for MoE it's the "active parameters" that are the real number on the bandwidth bill.** The DGX Spark is "a giant in capacity, a commoner in bandwidth": 128GB lets it swallow models others can't, but its ~273 GB/s LPDDR5X is an order of magnitude slower than the discrete card's 1TB/s (GDDR7) or even ~8TB/s (HBM3e). So its optimal strategy is always "**use the vast capacity to run the big MoE that no one else can**," not "use it to race small models to see who's faster."

**How this iron law punctures the previous chapter's myth**: Run a 9B model on the DGX Spark and you'll **of course** hit 180+ tok/s — but that's waste. A 9B runs beautifully on a $2,000-something M-chip Mac; you bought 128GB for capacity, not to make small models faster.

> 💡 A Word to the Wise
> A word worth ten years of study: **Choosing hardware and choosing a model are fundamentally the same problem — maximizing output on your bottleneck resource.** The DGX Spark's bottleneck isn't capacity (it has a surplus), it's bandwidth (it's scarce). Feeding the scarce resource to a MoE with "small active params, large total params" means using surplus capacity to compensate for the bandwidth shortfall — and that's the essence of Roofline thinking: **don't be dazzled by the biggest number on the spec sheet; find the smallest wall in the system, then make your workload run right up against that wall.** Grasp this and what you're selecting is no longer just a model — it's the optimal operating point of the whole system.

## Scene 2 — Six Shadows Side by Side: Who Deserves This Beast

**Background**: Taking the previous chapter's models, here's a head-to-head comparison against the DGX Spark GB10 (evaluated under Q4_K_M quantization).

| Model | Memory Footprint | Expected DGX Spark Speed | Strength | Weakness |
|---|---|---|---|---|
| **GPT-OSS 120B Fable-5** (MoE) | ~72GB | 35–50 tok/s | 128-expert MoE; complex multi-step code and math | Bulky; noticeably slows past 32K context |
| **Qwable-v1** (35B) | ~24GB | ~102 tok/s | Perfectly inherits Fable 5's Agent trajectories; best fit for coding agents | Less breadth of knowledge than the 120B |
| **clzoro/Qwen3.5-27B** | ~19GB | 110+ tok/s | Fed on Opus dialogue; superb at literature/role-play | Tool calling weaker than Qwable-v1 |
| **Qwen3.6-14B-FableVibes** | ~11GB | 140+ tok/s | Pruned MoE; retains the "Claude thinking feel" | Occasional rumination (hallucination) on complex logic |
| **Qwythos-9B** (incl. abliterated) | ~6.5GB | 180+ tok/s | Blazing fast, thoroughly uncensored; runs on junk GPUs | Too small; falls short on the completeness of deep reasoning |
| **Dzluck/Qwen3.5-2B** | ~2.1GB | 250+ tok/s | Ultra-lightweight, designed for edge devices | Overkill on this box; obvious IQ downgrade |

> ⚠️ Authenticity Caveat
> The speed and memory figures are **all unmeasured estimates / circulating claims**. The relative ordering matches Roofline intuition (smaller params run faster, MoE beats same-size dense, more aggressive quantization shrinks footprint), so treat them as "reasonable estimates," not "measured benchmarks." The community circulates one genuine reference point: "Solved the DGX Spark, 102 stable tok/s on Qwen3.5-35B-A3B" (see Appendix A) — which lines up with the table's Qwable-v1 (same 35B-A3B class) at 102 tok/s, because 3B active @ Q4 ≈ 1.7GB, and 273 ÷ 1.7 ≈ 160 theoretical ceiling, with the measured 102 falling within a reasonable discount. **But the same ruler exposes a flaw in the table**: Qwythos-9B is listed at 180+ tok/s — 9B dense @ Q4 ≈ 5.4GB, and 273 ÷ 5.4 ≈ **50 tok/s is the physical ceiling**; 180 already exceeds it by more than 3×, which **single-request decode cannot possibly reach, unless it's the total throughput of high-concurrency batching or the table is simply padded**. This perfectly demonstrates the chapter's core discipline: **for any tok/s number, first run it through the "bandwidth ÷ bytes read per token" ruler; whatever fails the physical ceiling gets flagged as suspect.**

**Don't just read the table — learn to compute "will it fit" yourself**: the "memory footprint" in the table is only the weights, but what truly goes into 128GB is three things stacked together, and undercounting any one of them means OOM on long context. Commit the formula to memory first:

```
Total footprint = weights + KV cache + framework/activation overhead (~2–4GB)

Weights (GB) ≈ params(B) × bits per param ÷ 8
   BF16/FP16 : 16 bit → ×2.00      FP8 / Q8_0 : 8 bit → ×1.00 (Q8_0 actually ~8.5 bit)
   Q4_K_M    : ~4.8 bit → ×0.60     NVFP4/Q4   : 4 bit → ×0.50

KV cache (GB) ≈ 2 × layers × KV heads × head dim × context length × bytes per element ÷ 1e9
   ★ ×2 = one each for K and V
   ★ GQA makes "KV heads" far smaller than attention heads (e.g. 8 vs 64) — the lifeline of KV savings
   ★ Rule of thumb: GQA-8 + FP16 KV eats ~0.3–0.5GB per 1K tokens (depends on layers)
```

Plugging four representative configurations into 128GB for a real reckoning (⚠️ numbers are illustrative, floating with actual layer count / GQA ratio):

| Model @ quant | Weights | KV @ 32K | KV @ 256K | 128GB verdict |
|---|---|---|---|---|
| **70B dense Q4_K_M** (GQA) | ~40GB | ~10GB | ~80GB | 32K comfortable (~50GB); **256K = 120GB nearly bursts, KV quantization is mandatory** |
| **120B-A4 MoE Q4** | ~72GB | ~5GB | ~40GB | 32K easy; 256K ≈ 112GB is tight, KV FP8 can save it |
| **35B-A3B Q8_0** (GQA) | ~37GB | ~6GB | ~48GB | comfortable single-model; running dual with a big model means controlling length |
| **8B dense Q4** | ~5GB | ~2GB | ~16GB | runs anywhere, but using 128GB for it is overkill |

Stare at those two KV columns — **weights are a one-time fixed cost, but the KV cache "inflates linearly" with context length.** At 32K you barely feel it, but stretch to 256K and the KV can approach, even exceed, the memory the weights occupy. **This is exactly why "fits a 70B" and "fits a 70B and can still run 256K long context" are two completely different problems** — and the true memory killer for the latter was never the model weights, it's the KV cache. (This is a seed; Chapters 10 and 11 will repay the debt across whole chapters.)

**A common selection verdict**: **First choice — GPT-OSS 120B Fable-5 or Qwable-v1 (35B)** — "not running a big model on 128GB is a waste." Check it against the Roofline and the fit formula above, and the verdict holds:

- **GPT-OSS 120B-A4 is the "IQ workhorse"**: 120B total params give it depth of knowledge, but only ~4 experts activate per token (roughly 20–30B actually participating), letting it dodge the bandwidth death-trap — 35–50 tok/s is acceptable. **This is the only pragmatic path to breaking the "70B-class IQ ceiling" on a personal device.**
- **Qwable-v1 (35B) is the "speed workhorse"**: its 24GB footprint is no strain at all on 128GB, 102 tok/s is near-"instant spray," and it natively emits `str_replace_editor` XML — zero translation to plug into OpenHands/Aider, making it several times more efficient than the 120B as a coding agent.

> 🔍 Deeper Commentary — Why MoE Is the Savior of Bandwidth-Bound Machines, and Its Hidden Costs
> This is the most valuable insight in the chapter: **on any bandwidth-bound hardware (DGX Spark, Apple Silicon, every UMA architecture), "active parameter count" matters far more than "total parameter count."** A 120B-A4 runs at speeds near a 20–30B dense model while its IQ approaches 120B — which is why nearly every open-weight model aimed at "personal supercomputers" in 2026 bets entirely on MoE. But MoE is no free lunch; it carries three hidden costs you must weigh when selecting. **First, the memory tax**: a MoE must keep **all 128 experts** resident in memory (you can't know which the next token routes to), so it "saves bandwidth and spends capacity" — which dovetails perfectly with the DGX Spark's "surplus capacity, scarce bandwidth," a match made in heaven, yet a disaster on a small-VRAM device. **Second, logic rumination**: routing switches between experts can make long reasoning chains incoherent (the table's "occasional rumination on complex logic" for FableVibes is exactly this) — which is also why clzoro's **dense 27B** is preferred for "coherence of thought" in literature/role-play scenarios. **Third, batch efficiency**: uneven expert load under high-concurrency batching is an old production-deployment headache. **So the complete mental model for selection is: ask "how much is activated" (sets speed), ask "how much in total" (sets IQ), ask "dense or MoE" (sets coherence and memory profile) — only after these three questions can you talk about choosing right.**

## Scene 3 — The Smartest Play: Dual-Model Parallelism, Plus an Advanced Easter Egg

**Background**: With 128GB, the smarter play isn't "picking one" but **dual-model parallelism** — loading a large and a small one at once, switching by task.

**A dual-model plan worth considering**:

- **Load simultaneously**: GPT-OSS 120B (~72GB) + Qwable-v1 35B (~24GB) = ~96GB, leaving ~32GB for both models' **KV cache**.
- **Backend**: **Ollama**, with built-in dynamic multi-model management; open two terminals or APIs in parallel: `ollama run gpt-oss:120b` / `ollama run qwable:35b`.
- **Frontend**: **Open WebUI / Cursor**, one-click swap from a dropdown — even "**dual-model showdown (Arena)**," where both models answer the same prompt and you pick the better.
- **Hot-swap**: while the big model runs, automatically free the idle model from LPDDR5X to clear room for ultra-long context; thanks to the ~273 GB/s UMA bandwidth and its "zero-copy, no PCIe" nature, swapping in and out is far smoother than a discrete card hauling over PCIe (but don't mistake it for HBM-class TB/s speed).

**The golden workflow**:

- **Qwable-v1 (35B) handles "the high-frequency daily grind"**: write-and-debug, edit code, draft short messages, translate, look up commands — at 102 tok/s of jet-speed, your train of thought never stalls waiting on the AI.
- **GPT-OSS 120B handles "heavy thinking"**: bugs the 35B can't crack, planning a whole architecture, throwing three long papers at it for cross-domain analysis — switch over and let the 128 experts storm the fortress.

> 💡 A Word to the Wise
> A word worth ten years of study: **People who truly know their tools don't chase one Swiss Army knife for everything — they make the right blade appear, imperceptibly, in hand at the right moment; and this "layered cognition" is in fact a mirror of the human brain.** Psychology's System 1 (fast, intuitive) and System 2 (slow, deliberate) map exactly onto "the fast model handles ninety percent of the everyday, the big model tackles the ten percent of hard bones." The essence of dual-model parallelism isn't "having two models" — it's "the switch itself becoming imperceptible." When the cost of invoking deep thought approaches zero, you'll actually think deeply when you should, instead of making do with the fast model because "switching is a hassle." **The highest art of tooling is making the correct choice the most effortless one.**

> 🔍 Deeper Commentary — The Dual-Model Easter Egg: Speculative Decoding, Where Two Models Merge to Go Faster
> The discussion so far only covered "switching by task," but since you've already loaded one large and one small model into the 128GB, there's an advanced technique it never mentioned that lets the two **accelerate cooperatively** — worth knowing: **speculative decoding**. The principle: let the **small model (Qwable-v1, or an even smaller draft model) quickly generate a string of draft tokens**, then let the **big model (GPT-OSS 120B) verify that whole draft in a single parallel pass**, accepting the right ones and rejecting the wrong. Because "verifying K tokens" is far cheaper on memory bandwidth than "generating K tokens one at a time" (one weight read verifies many), measurements often show the big model's effective output speed rising 1.5–3×, **with output distribution identical to the big model's (lossless acceleration)**. This pushes the DGX Spark's "having both a big and small model at once" condition to its limit — the small model isn't just a backup for "fast everyday answers," it's the big model's "oracle accelerator." Tooling-wise, **llama.cpp (`--draft` / draft model), vLLM, and SGLang** all support it out of the box. **One more hidden-cost reminder**: don't forget the KV cache is the true memory killer for long context — when context piles up to 200K tokens, that remaining ~32GB gets devoured by a KV cache that **rises linearly with length** (look back at Scene 2's fit formula: a 70B-class model's KV at 256K can reach ~80GB), and that's when you lean on **FP8/INT8 KV-cache quantization** (supported in vLLM, llama.cpp) to halve it, rather than naively assuming "unloading one model" is enough. **The bottleneck of long context was never the weights — it's the KV cache.**

## Scene 4 — Deployment in Practice: Nailing the Model Into That 128GB

**Background**: You've picked your model; the last mile is "how to wring peak speed out of it on the DGX Spark." A few concrete suggestions follow, with the tooling details filled in.

**The deployment toolchain**:

- **Format and engine**: download GGUF (first choice **Q4_K_M**, with core layers bumped to Q8_0 for mixed quantization); for the engine, use **llama.cpp / Ollama** (easiest to start), or **vLLM / SGLang** when chasing throughput (continuous batching pushes arithmetic intensity toward the ridge point, plus KV PagedAttention); DGX OS ships the **NVIDIA AI software stack / NIM / TensorRT-LLM** to spin things up directly.
- **Cross-platform note (don't grab the wrong ecosystem)**: the DGX Spark runs the **CUDA** ecosystem (llama.cpp-CUDA / Ollama / vLLM / SGLang / TensorRT-LLM); **Apple Silicon**, of the same UMA camp, runs **MLX / llama.cpp-Metal**. The two share **the same principle** — both bet on "unified memory + on-device quantized models" — but their kernels and binaries don't interoperate: **an MLX model that runs well on a Mac can't be dropped straight onto the GB10**, and vice versa. Get this clear and you won't copy the wrong deployment script.
- **Quantization calibration**: when doing GGUF quantization, use an **imatrix (importance matrix)** — run a small batch of calibration corpus to gauge each weight's importance, so low-bit quantization preferentially preserves the critical weights — **same Q4 footprint, noticeably higher quality**. This is a standard advanced move in the llama.cpp community.
- **Core-level pinning (the key to squeezing UMA bandwidth)**:
  - `numactl --interleave=all`: **interleave** the model weights across all LPDDR5X memory channels, forcing every channel to work at full load simultaneously and avoiding cache misses when CPU/iGPU contend for bandwidth.
  - `mlock`: **lock the weights in physical memory** to stop the OS from paging (swapping) them out to slow disk — a hardcore detail for sustaining high tok/s.
  - **Don't rely on generic default Docker containers**: their memory strategy may not suit UMA; only by manually tuning NUMA and mlock do you unleash the GB10.

**The final cut that punctures the "1.04M context" myth** — don't just read the vendor spec, **verify it yourself**:

```
How long does it fit?       →  look at RoPE/YaRN settings (the easiest number to market)
How long can it run?        →  the real usable length, bounded by KV cache and bandwidth
How long does it comprehend?→  run RULER / Needle-in-a-Haystack — truth and lie laid bare
```

**Of these three questions, the third — "how long does it comprehend" — is the most often skipped and the most lethal** — and it's exactly what **NIAH** and **RULER** measure:

- **NIAH (Needle-in-a-Haystack)**: in a long, irrelevant passage of hundreds of thousands of words (the haystack), hide one key fact (the needle, e.g. "the 2026 passphrase is nol9981"), pad the context to a target length, then ask the model for that sentence — it tests the **lowest bar of pure retrieval**. Fail even this and "million-token context" is pure advertising.
- **RULER**: the industrial-strength version of NIAH, which beyond single-needle retrieval stacks on **multi-needle retrieval, variable tracking (multi-hop), aggregation statistics, long-document QA** and a dozen-odd other task types, lengthening the context segment by segment and recording the length at which the model's accuracy **drops below threshold (often set at 85%)** — and *that* length is the "**effective context length**." The brutal reality: **a model nominally rated at 1M often has a RULER-measured effective length of only 64K–128K**, with the long tail beyond being "fits but doesn't comprehend" bloat. So what you should trust is not the spec sheet's RoPE base, but the inflection point on the RULER curve where accuracy collapses — and that curve, with `lm-eval-harness` plus the open-source RULER suite, you can produce on your own DGX Spark in a single night.

> 💡 A Word to the Wise
> A word worth ten years of study: **The numbers on the spec sheet are the truth the vendor wants you to see; running the benchmark once is the truth the machine is willing to tell you.** "1M context," "180 tok/s," "uncensored" — each of these slogans reveals only the prettiest of the three dimensions behind the problem. The mature engineer's instinct is to break any single number back into its independent dimensions — "fits / runs / comprehends," "total params / active / architecture" — then measure it by hand with off-the-shelf open-source benchmarks (RULER, lm-eval-harness, HumanEval). **Reverence for numbers isn't disbelief — it's "trust, but verify" — and the tools for verification are all free and open-source; not verifying is your own choice.**

> 🔍 Deeper Commentary — UMA Is a Gamble on Memory Architecture, and What the DGX Spark Bet Right
> The DGX Spark's **unified memory architecture (UMA)** isn't a clever NVIDIA trick — it's a gamble on "what personal AI compute looks like in the future," worth understanding from a higher vantage. Traditional GPUs solder "fast but small" HBM/GDDR onto the card, while CPU memory is "big but slow," with the narrow PCIe bridge between them — running a big model, weights shuttle back and forth between the two memories, and PCIe becomes the bottleneck. UMA tears down that wall: CPU and iGPU share one pool of 128GB LPDDR5X — **no shuttling, no PCIe bottleneck, no "won't fit in VRAM."** The price is that this pool's bandwidth (~273 GB/s) sits in the awkward zone between HBM3e (~8TB/s, scorching fast) and ordinary dual-channel DDR5 (under ~100 GB/s, slow) — **it bets that "for personal/edge scenarios, capacity is scarcer than peak bandwidth."** For personal/edge scenarios, that bet is very likely right: you're far more often blocked by "the model won't fit in 24GB of VRAM" than by "bandwidth isn't fast enough." Apple Silicon (the M series) long ago validated this path; the DGX Spark is the version that pushes it to 128GB + Blackwell compute. **This also foreshadows a long-term shift in selection philosophy: on machines like these, the future winner won't be "the fastest small model" but "the largest MoE with controllable active params" — the hardware architecture's bet will, in the end, define which models are worth distilling and deploying.** The logic you use to select for the DGX Spark today is a miniature of personal AI compute over the next few years.

## Chapter Summary

- **Calibrate the specs first**: DGX Spark = GB10 + 128GB LPDDR5X UMA + **~273 GB/s** (⚠️ not the 600 that second-hand articles love to write; off by 2× and every tok/s is off by 2×) + ~1 PFLOP FP4/NVFP4.
- **Read the DGX Spark through the Roofline**: ridge point = compute ÷ bandwidth ≈ **~3600 FLOP/byte**, while decode's arithmetic intensity is only **2–4** — deep in the bandwidth wall; decode speed ≈ bandwidth ÷ weights read per token. A giant in capacity, a commoner in bandwidth.
- **The three questions of selection**: how much is active (speed), how much in total (IQ), dense or MoE (coherence / memory profile). Verdict: first choice a big MoE (GPT-OSS 120B) + a speed model (Qwable-v1 35B).
- **Compute fit yourself**: total footprint = weights + KV cache + framework overhead; weights = params × bpw ÷ 8, KV inflates linearly with context. At 32K watch the weights, at 256K watch the KV — **the memory killer of long context is the KV cache, not the weights** (a seed for Chapters 10 and 11).
- **Dual-model parallelism = layered cognition**; advance to **speculative decoding** to merge big and small models for lossless acceleration; KV-cache quantization (FP8) is the true bottleneck solution for long context.
- **Deploy to wring out the UMA**: GGUF + imatrix calibration, `numactl --interleave=all` + `mlock`, steer clear of generic Docker.
- **Break long context into three questions**, verify it yourself with RULER/Needle — don't trust the spec sheet.

With hardware and selection laid bare, the next question returns to the essence: **how exactly are these shadows distilled?** Leaving hardware behind, we step into the science of distillation — Chapter 4.
</content>
</invoke>
