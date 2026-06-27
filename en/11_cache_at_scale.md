# Part XI — The Armory: When Cache Becomes a Schedulable Strategic Asset

# Chapter 11 — Cache at Hyperscale and the Frontier Armory: From Single-GPU VRAM to the Cross-Cluster KV War

> Silicon Valley, the war room of some AI inference platform. In Chapter 10 you tamed the single-machine KV Cache into perfect obedience — PagedAttention no longer fragments, Prefix Caching hits beautifully, single-card throughput doubled. You thought the war was over. Then the product shipped, the number of concurrent online Agents grew from a dozen to several thousand, and you sat staring at a bizarre compute curve on Grafana: you'd built prefix caching, yet the cluster's share of prefill compute kept climbing instead of falling. You logged in to investigate, and a chill ran down your spine — your round-robin load balancer was **taking the same user's five rounds of conversation and flinging them, one after another, onto five different machines**. Every machine that caught a request dutifully, pointlessly, recomputed the prefill of those fifty thousand tokens of history from scratch. The single-machine cache you so carefully designed hit 99% on one machine — but the next sentence got routed to the machine next door, and the hit rate dropped to zero. The whole data center was quietly burning electricity into ash for one scheduling blind spot: *the cache was hidden on the wrong node.* In that moment you finally understood: on a single machine, cache is a slab of VRAM you must spend sparingly; in a cluster, **cache is a strategic asset that must be *scheduled*** — it has a location, an affinity, a migration cost. Who holds it, and which machine it sits on, is — exactly like "do we have enough compute" — the kind of thing that gets an SRE out of bed at three in the morning.

> ⚠️ Chapter-wide Authenticity Caveat
> The raw material this chapter integrates is an architectural treatise written in the voice of a "Google DeepMind staff engineer / principal architect leader" — **the narrative carries a strong internal vantage point and a forward-looking sheen**. The discipline applied here is as follows: anything that is **publicly known, verifiable research and tooling** (MLA / DeepSeek, H2O, StreamingLLM / Attention Sink, KV-Quant, vLLM / PagedAttention, SGLang / RadixAttention, LMDeploy, TensorRT-LLM, LightLLM, DeepSpeed-FastGen, GPTCache, Milvus / Qdrant, Mamba-2 / Griffin, Anthropic Claude `cache_control`) is **stated plainly, as usual**; but anything touching **"how DeepMind / OpenAI does it internally," Gemini's specific context length, the Jupiter network's 1.6 Tbps bandwidth, KV cut by 90% / cost down 80% / hit rate several times higher — these precise numbers** are all treated as "plausible engineering hypotheses or vendor claims," flagged with `> ⚠️ Authenticity Caveat`, to be taken as directional reference, not cited as established fact. This chapter has a higher authenticity-density than the preceding ones — because the material itself speaks from the high platform of a "self-described top-lab leader," and the prettier the internal number, the more it should be discounted in the listening.

---

## Scene 1 — The Three-Tier Heterogeneous Storage Pyramid: Managing the KV Cache Like an OS's Virtual Memory

**Background**: In models that routinely support million-token contexts, if you deadlock every user's KV Cache inside the GPU/TPU's HBM (High-Bandwidth Memory), the cluster will seize up the instant concurrency ticks up, OOM-ing into instant paralysis. HBM is the fastest, costliest, and scarcest resource in the entire system — using it to store "the conversation history of a user who hasn't come back in three days" is a waste extravagant to the point of criminality. The solution the source argues for is, in essence, to take a trick the operating system has managed for fifty years — **virtual memory and multi-level caching** — and transplant it, untouched, onto the KV Cache: build a three-tier heterogeneous storage pyramid (Hierarchical KV Cache Management).

The three-tier pyramid, from fast to slow, costly to cheap, small to large:

```
            ┌───────────────────────────────────────────┐
            │  L1: TPU/GPU HBM  (High-Bandwidth Memory)  │  ← fastest · costliest · smallest
            │  Stores: the Active Segments now Decoding   │
            │  Capacity scale: tens of GB / card  BW: ~TB/s│
            └───────────────────────────────────────────┘
                          ▲   100 GB/s ~ TB/s
                          │   PCIe Gen5 / NVLink / ICI
                          ▼
            ┌───────────────────────────────────────────┐
            │  L2: Host Memory  (CPU DDR5)               │  ← mid-speed · mid-price · mid-volume
            │  Stores: momentarily inactive but soon-needed multi-turn history │
            │  Capacity scale: hundreds of GB ~ TB / node │
            └───────────────────────────────────────────┘
                          ▲   Direct I/O
                          │   NVMe-over-Fabrics / RDMA
                          ▼
            ┌───────────────────────────────────────────┐
            │  L3: Distributed NVMe SSD Pool             │  ← slow · cheap · vast
            │  Stores: cold data (long-video multimodal, frozen Sessions) │
            │  Capacity scale: tens of TB ~ PB / pool     │
            └───────────────────────────────────────────┘
```

The correspondence is unmistakable: **L1 is to the KV Cache what the CPU's L1 cache is to main memory; L3 is to L1 what the hard disk is to RAM.** When a user is typing and the model is generating token by token, his KV must be in L1; when he finishes a sentence and enters the few-second lull of "a human thinking about what to say next," swap (Offload) his KV from HBM down to L2's DDR5, and HBM instantly frees up to serve someone else; when he hasn't come back in half an hour, sink the whole Session down to L3's SSD pool to freeze.

**Key technique: the asynchronous prefetching pipeline.** The pyramid itself isn't hard; the hard part is "the hauling" — HBM and Host Memory are separated by PCIe, and moving several GB of KV at a time takes time. If computation stops and waits while the hauling happens, the whole tiering becomes a negative optimization: you saved VRAM but paid in latency. The source's solution is, inside the compiler (XLA) and the runtime, to **fully decouple the compute thread from the I/O thread**, and to bury the hauling cost under one ancient, powerful word — **Overlapping**:

```python
# Conceptual sketch: use "someone else's decode time" to mask "my KV hauling latency"
# A real system implements this with async DMA at the XLA runtime / CUDA stream layer;
# this only illustrates the scheduling semantics

while serving:
    batch = scheduler.next_batch()          # grab a batch of requests currently decoding

    # ── I/O thread (runs in parallel with compute, non-blocking) ─────────
    if detect_user_typing_next_turn(user_X):
        # Detected that user X is typing the next sentence → asynchronously prefetch
        # his history KV from L2 DDR5 back into L1 HBM. The compute thread doesn't
        # know and doesn't have to wait for this.
        io_thread.async_prefetch(user_X.kv, src="DDR5", dst="HBM")

    # ── Compute thread (stays fully loaded running the current batch) ─────
    compute_thread.decode_step(batch)        # the duration of this step "covers" the haul above

    # By the time user X actually hits send, his KV has long been lying in HBM → zero-felt latency
```

The essence is in that comment: **use the decode-compute time of other in-flight requests to "mask" the hauling latency of the current request's KV Cache.** In large-batch inference there is always a heap of requests decoding, keeping the TPU's Tensor Cores well-fed; the I/O engine hides in the shadow of that computation and quietly hauls the next KV it will need from DDR5 back to HBM. By the time the user actually presses Enter, his history has long been waiting in HBM — the latency is hidden inside a stretch of time the hardware was busy with anyway, and hardware utilization (MFU) runs high throughout.

> 💡 A Word to the Wise
> A word to the wise: **Every high trick for "making a system faster," dissected to the bottom, is almost the same sentence — never let the expensive resource wait on the cheap one; and when hauling is unavoidable, hide it in the shadow of computation.** The three-tier pyramid is no new AI invention; it is the virtual memory of Chapter 3 of the OS textbook, it is the CPU's L1/L2/L3, it is the database buffer pool — it has simply put on a "KV Cache" costume and walked the same road again. The truly valuable part is not the structure of "having three tiers" — that's common sense; the valuable part is **the single move of asynchronous prefetch, "using someone else's compute time to mask my hauling time."** A novice building tiered cache builds the synchronous version where "both swap-out and swap-in have to stop and wait," and ends up saving VRAM at the cost of latency — a bad trade. A veteran building tiers thinks first of "the hauling latency — can it be hidden behind some stretch of time that has to be spent anyway?" **This is the philosophy of overlapping: the highest art of systems engineering is not to make everything fast, but to make the slow thing happen when no one is waiting on it.** Whether you can spin two independent timelines — the compute thread and the I/O thread — in your head at once, and interleave them at exactly the right points: this single distinction separates "people who use cache" from "people who schedule cache."

> 🔍 Deeper Commentary — Why "KV Cache paging" is more treacherous than "OS paging," and more worth it
> Treating the KV Cache as virtual-memory paging is absolutely the right direction, but you must see clearly its **three key differences** from OS paging, or copying it wholesale will flip the car. **First, granularity and semantics differ**: the OS pages in units of fixed 4KB pages whose contents are semantically meaningless; the KV Cache's unit is "the attention state of a token sequence," and it carries a strong **temporal prefix dependency** — you cannot swap out only the middle of a history, because later tokens, when computed, must see the earlier ones. So the KV swap unit is usually "an entire prefix" or vLLM's fixed block, and it must be prefix-contiguous. **Second, the cost structure is inverted**: OS paging swaps out "data," and swapping it back gives you the same thing; the KV Cache, if not swapped but discarded, must be **recomputed by prefill** to restore — and prefill is heavy, compute-bound work. This means there's an accounting problem between "swap out to DDR5" and "discard and recompute": the I/O cost of one PCIe round-trip versus the compute cost of re-running prefill once — which is cheaper? The longer the sequence, the more expensive the recompute, the more it deserves to be hauled and kept; the shorter the sequence, the cheaper the recompute, and discarding actually saves trouble. This accounting is exactly the microscopic version of Scene 3's "speculative cache migration." **Third, this is a direction vLLM has already productized**: vLLM's CPU offloading, `--swap-space`, and the community's KV tiering work are precisely the L1↔L2 realization of this pyramid; push further toward L3 SSD and you enter the territory of frontier systems like Mooncake (the KVCache-centric disaggregated architecture behind Kimi). **So this section is no castle in the air — it takes a direction already running on production lines and makes its physics clear.** The only discount to apply to the source is: "zero-felt latency" is the ideal state — the instant the user replies faster than the prefetch, or the prefetch guesses the wrong person, the tail of the hauling shows. There is no true zero latency, only latency "hidden well enough or not."

---

## Scene 2 — Subtracting from the Cache: From MHA to MLA, the Slimming Art at the Model-Structure Layer

**Background**: Scene 1 was pure engineering — don't touch the model, just build a pyramid outside it and haul things around. But the source punctures its ceiling in one sentence: **"Pure engineering optimization has a ceiling; you must join forces with the algorithm team (Co-design) and slim the Cache from the model structure itself."** This is the most important methodological sentence in the chapter. The reason the KV Cache is large is rooted in Attention's structure: every layer, every attention head, must cache a K and a V for every token. To cut it at the source, you have to operate on Attention itself. This evolutionary line — MHA → MQA → GQA → MLA — is one of the most important hidden threads in large-model inference efficiency over the past few years.

First, get the KV Cache volume formula straight. In classic **Multi-Head Attention (MHA)**, the KV each token must cache is proportional to: `2 (K and V) × L (layers) × H (heads) × D (per-head dimension)`. That **H (number of heads)** is the part successive improvements have repeatedly gone under the knife to cut:

```
MHA  : every Q head gets its own K head, V head        KV heads = H      (baseline · largest)
        Q1 Q2 Q3 Q4 ... QH
        K1 K2 K3 K4 ... KH
        V1 V2 V3 V4 ... VH

MQA  : all Q heads share ONE single set of K, V         KV heads = 1      (saves H× · hurts ability)
        Q1 Q2 Q3 Q4 ... QH
        K (only one, shared by all)
        V (only one, shared by all)

GQA  : Q heads grouped, each group shares one K, V       KV heads = G (groups, e.g. 8)  (industry mainstream)
        [Q1 Q2 Q3 Q4][Q5 Q6 Q7 Q8]...
         K_grp1        K_grp2
         V_grp1        V_grp2

MLA  : don't store high-dim K/V, store a low-rank compressed latent vector c_t   (cache = one tiny latent; decompressed on-the-fly at decode)
        K, V ──low-rank projection──► c_t (tiny dimension)   at decode ──linear projection──► restore K, V
```

Unpacking the trade-offs along this slimming line one by one:

| Mechanism | KV Cache relative size | Core approach | Cost / status |
|---|---|---|---|
| **MHA** | 1× (baseline) | Each Q head owns its own KV head pair | Strongest expressive power, but largest KV volume; long context eats the most VRAM |
| **MQA** | ~1/H | All Q heads **share one single set of KV** | VRAM plunges by H×, but expressive power drops too far; large models suffer severe capability loss |
| **GQA** | ~G/H (e.g. 8/H) | Q heads **grouped**, each group shares one KV pair (Llama family commonly uses 8 groups) | A compromise of speed and quality, **currently the industry mainstream**, widely adopted by Llama 3 etc. |
| **MLA** | Claimed to save 90%+ | **Low-rank compression**: project K/V into a low-dim latent space c_t, decompress on-the-fly at decode | Proposed by DeepSeek (the V2/V3 flagship work); greatly shrinks KV while maintaining representational power; an exemplar of algorithm-infra co-design |

MQA and GQA are about "reducing the number of KV heads" — MQA cuts in one stroke down to a single group, too brutal, and the model goes dumb; GQA learned its lesson and cuts down to a few groups (say, 8), finding a sweet spot between "save VRAM" and "keep ability," and thereby became standard equipment for mainstream open models like Llama 3.

**MLA (Multi-head Latent Attention)** takes a completely different road. It doesn't subtract from "head count"; it compresses on "dimension." Its insight is: the high-dimensional K and V hide a great deal of redundancy, which can be projected and squeezed into a latent-space vector `c_t` of tiny dimension via **low-rank matrix decomposition (Low-rank Compression)** — when caching, you store only this tiny `c_t`, not that original big lump of high-dimensional K, V; and at the decode stage, you use linear projection matrices to **decompress `c_t` on-the-fly** back into the real K and V to compute attention. What each token originally had to store as `2 × L × H × D` now needs only one small vector.

> ⚠️ Authenticity Caveat
> The number "MLA cuts the KV Cache by 90%+ and lifts concurrency several-fold" comes from DeepSeek's paper and technical reports, and is **a measured value under a specific model and specific configuration**, not a universal constant — the compression ratio depends strongly on the choice of latent-space dimension, model scale, and sequence length. MLA was proposed by DeepSeek and validated on its open models — verifiable, real work; but "DeepMind is also closely tracking similar directions" is the source's internal-voice **claim**, which this book cannot independently verify — take it as directional reference only. The facts that can be stated plainly are: MQA / GQA / MLA are all published technologies already adopted by production-grade models, and this evolutionary line of "slimming Attention's KV" is real.

> 💡 A Word to the Wise
> A word to the wise: **The deepest performance optimization never happens inside the separate walls of "engineering" and "algorithm"; it happens the moment that wall is torn down and the people on both sides sit at the same table — MLA is what grew up when that wall fell.** Scene 1's pyramid is something one infra-team person can finish alone: the model doesn't move, I haul things outside it. But it has a ceiling — however big the KV itself is, the pyramid must haul that much; you've only turned "expensive hauling" into "cheap hauling," you haven't made the thing that needs hauling smaller. MLA is different: what it changes is the mathematical form of attention, which is the algorithm team's domain; yet deciding "how many dimensions you can squeeze to without losing accuracy, and whether the decompression operator runs at all on a TPU" is infra's domain. **Neither side alone can produce MLA: the algorithm team, not understanding the hardware cost of the decompression operator, will squeeze out something theoretically pretty and practically slow; the infra team, not daring to touch the attention formula, will only haul boxes around outside.** This is the true meaning of co-design — not the relay race of "the algorithm is designed first, then thrown to engineering to implement," but both sides at the whiteboard simultaneously forced to concede under the other's constraints, growing a compromise neither could have thought of. **Whether a team can make a structural breakthrough depends not on how strong the algorithm scientists or how hardcore the systems engineers it hires, but on whether it has the courage to make these two kinds of people argue over the same KV number, at the same whiteboard.**

---

## Scene 3 — The Distributed Cluster: When Cache Has a "Location," Routing Becomes War

**Background**: Pushing PagedAttention and Prefix Caching to the extreme on a single machine is still not enough. The moment you manage a hyperscale inference cluster of tens of thousands of TPUs/GPUs, the gateway and scheduler must upgrade to a new capability — **Cache-Aware**. Because in a cluster, cache is no longer an abstract "VRAM space"; it **lives on one specific physical machine**. Who holds user A's history KV — node 7 or node 42 — decides where the next request should go. This is exactly the root of the "electricity burned into ash" disaster that opened this chapter.

**The disaster of traditional scheduling (Round-Robin / Least-Connection)**:

```
User A fires off 5 sentences; the traditional load balancer dispatches by "least busy" or "round robin":

  msg#1 ──► node1   (recompute all of A's history prefill)
  msg#2 ──► node2   (node2 has no cache of A → recompute all history again)
  msg#3 ──► node3   (recompute once more)
  msg#4 ──► node1   (node1's cache may already have been evicted by someone else → recompute anyway)
  msg#5 ──► node4   (recompute)

Result: the prefill of the same "fifty-thousand-word history" got computed 5 times.
        The Prefix Cache you built on every machine has a hit rate of zero across the board —
        because the next sentence simply doesn't return to the machine where the last one lived.
```

The load balancer's instinct is "give the work to the least busy machine" — a golden rule in stateless web services. But LLM inference **is not stateless**: request N is cheap precisely because "request N-1's KV is still on this machine." Handing a stateful cache to a scheduler that assumes statelessness is taking the whole cluster's compute and burying it as a sacrifice for the blind spot of "the cache was put in the wrong place."

**The right answer: prefix hashing + affinity routing + speculative cache migration.** The gateway must hold a global distributed cache index (usually maintained with an in-memory database or efficient RPC), and proceeds in three steps:

| Step | Mechanism | Approach |
|---|---|---|
| ① Identify | **Prefix Hashing** | The gateway computes a prefix hash over the request's `System Prompt + conversation history` (e.g. **MurmurHash3** — fast, evenly distributed). Same prefix → same hash |
| ② Route | **Affinity Routing** | No longer throw to the "least busy" node, but preferentially to **the physical node that has already cached that prefix's KV**, maximizing the cluster's overall Prompt Cache hit rate |
| ③ Trade off | **Speculative Cache Transfer** | If the node A holding the cache is already overloaded: compute "recompute prefill on idle node B" vs "pull the KV directly from A to B over the cluster's high-speed network" — whichever is faster wins |

The third step is the most precise move in the whole design, worth unpacking for its accounting problem:

```
Situation: a user request hits node A's cache, but node A is overloaded right now (long queue).
The scheduler faces two roads:

  Path A [Recompute]: send the request to idle node B, let B recompute the whole history prefill from zero
       cost ≈ Prefill_FLOPs(history length) / B's compute
       —— the longer the sequence, the more expensive this road (prefill is compute-bound)

  Path B [Migrate]: take the already-computed KV on A and haul it directly to B over RDMA high-speed network
       cost ≈ KV_Bytes(history length) / cluster network bandwidth
       —— when bandwidth is fierce enough, hauling a ready-made copy is far faster than computing one from scratch

  Decision: if  haul_cost(B) < recompute_cost(A):  take RDMA migration, saving the whole prefill compute
            else:                                  cut losses, recompute on B
```

Notice this problem and Scene 1's "KV paging vs recompute" are **the same problem replayed at a different scale**: Scene 1 was "haul to the DDR5 next door" versus "discard and recompute"; here it's "haul across machines to node B" versus "recompute on B." The physical intuition is identical: **prefill's compute cost grows super-linearly with sequence length, while the cost of hauling one ready-made KV grows only linearly with its byte count.** So the longer the sequence and the larger the context, the more cost-effective "hauling the ready-made copy" becomes — which is precisely why, in the long-context era, the in-cluster cache-migration network becomes as critical as the compute itself.

> ⚠️ Authenticity Caveat
> Concrete bandwidth numbers and internal network code-names like "Google Jupiter network 1.6 Tbps" and "direct cache transfer brings a significant TTI advantage" are **claims** the source gives in the internal DeepMind voice, which this book cannot independently verify — treat them as order-of-magnitude illustration. The real techniques that can be stated plainly are: **affinity routing + prefix hashing is a genuine direction that SGLang's RadixAttention, and various inference gateways (vLLM's prefix-aware scheduling, Mooncake's global KV index) are actually doing**; MurmurHash3 is a real non-cryptographic hash; RDMA hauling KV across nodes is also a real mechanism in the PD-disaggregation (Scene 4) architecture. The only thing to discount is product-specific numbers like "such-and-such vendor, such-and-such network, such-and-such bandwidth."

> 🔍 Deeper Commentary — The reef of affinity routing: cache affinity and load balancing are twins born to be at odds
> Affinity routing sounds perfect, but on the production line it drags you into a classic **dilemma**, and missing this means leaping from one pit into another. **Load balancing wants to "spread the work out"; cache affinity wants to "concentrate the work on the machine that holds the cache" — these two goals physically hedge against each other.** The more strictly you pin all of user A's requests to node 7 (maxing out the cache hit rate), the more easily you manufacture a **hotspot**: a frantically-called long system prompt (say, a viral Agent template) will funnel all traffic that hits it onto the two or three machines that first cached it, charring them, while other machines in the cluster sleep. This is why a mature cache-aware scheduler is never "mindless affinity," but **affinity with load awareness** — SGLang's scheduling, and the approaches of various gateways, are all in essence dynamically scoring between "cache-hit gain" and "node-load cost": a hit is good, of course, but if the holder is already overloaded, trigger Scene 3 step ③'s "migrate vs recompute" accounting and proactively divert traffic or replicate the cache to a second machine. **Deeper still, this forces an architectural philosophy: in a stateful inference cluster, there is simply no static answer called "optimal routing," only a dynamic process of "continuously rebalancing among cache affinity, node load, and migration cost."** Treat it as a hash function computed once, and you die of hotspots; treat it as a continuously regulated control loop, and you truly own a cluster that breathes. This is also the eternal motif of distributed systems: between consistency (pinning state to a fixed location) and availability/load (spreading traffic out), there is forever a tug of war — cache-aware routing is just the latest battlefield of that tug of war in the LLM era.

---

## Scene 4 — PD Disaggregation: When Prefill and Decode's Hardware Needs Are Exactly Opposite

**Background**: Push cache-aware routing to the extreme, and you collide with a deeper structural contradiction, whose solution is the "ultimate evolutionary direction" of current hyperscale inference infrastructure — **PD Disaggregation (Disaggregated Prefill & Decode)**. To understand why it is the endgame, you must first see clearly a fact many people overlook: the two stages of LLM inference, **Prefill and Decode, are two physically opposite loads**, and stuffing them on the same card, the same batch, is making two incompatible people share one room and drag each other down.

A comparison of the two stages:

| Dimension | Prefill (digest the prompt) | Decode (generate token by token) |
|---|---|---|
| Compute character | **Compute-bound** | **Memory-bound** |
| What it's doing | Pass the whole prompt through once, producing the KV Cache | Generate 1 token at a time, repeatedly reading the massive KV |
| What resource it eats | Eats **compute** (Tensor Cores maxed), eats bandwidth | Eats **VRAM capacity** (to hold all concurrent KV), eats memory-access bandwidth |
| Ideal hardware | High-compute, high-bandwidth chips (e.g. HBM3e) | Large-capacity VRAM |
| Pain point | A long prompt arriving **hogs the compute**, starving requests that are decoding | Compute sits largely idle, just endlessly hauling KV |

The contradiction is right here: **Prefill is short on compute; Decode is short on VRAM capacity; when Prefill squeezes compute dry, Decode's compute is idle; when Decode fills VRAM, Prefill's VRAM demand is crushed.** Mixed in the same GPU batch, the ugliest thing happens — a fifty-thousand-token long prompt enters prefill, instantly hogs all the compute, and the dozens of short requests answering character by character collectively freeze, their **TTFT (time to first token) and TPOT (time per output token) spiking together** — to the user it looks like "typing froze halfway through."

The industry has two layers of solution to this contradiction, one light and one heavy:

```
─── Same-machine relief: Chunked Prefill ──────────────────────────
  Slice a long prompt's prefill into fixed-size chunks (e.g. 512 tokens at a time),
  compute one chunk → yield compute to let decode run a step → compute the next chunk.
  Turn "swallowing a long prefill in one gulp" into "small-sip pipelining," avoiding long requests starving short ones.
  ↑ vLLM / DeepSpeed-FastGen's Dynamic SplitFuse both do this. It is "time-slicing on the same card."

─── Cross-machine cure: PD Disaggregation ─────────────────────────
  Just split the cluster into two kinds of dedicated nodes:

   ┌────────────────────┐                      ┌─────────────────────┐
   │  Prefill node (P)  │   RDMA push KV       │  Decode node (D)    │
   │  high-compute·high-BW│ ───────────────────►│  large-capacity VRAM │
   │  (e.g. HBM3e)       │   push the computed   │  focus on token-by-token gen │
   │  focus on digesting prompt │ KV over high-speed net to D │  not interrupted by prefill │
   └────────────────────┘                      └─────────────────────┘

  The P node finishes computing the KV and "pushes" the whole KV to the D node over an ultra-high-speed RDMA network;
  the D node takes it and focuses on decode. The two kinds of hardware each do their own job, not interfering.
```

Chunked Prefill is **time-slicing on the same card** — relief, treating the symptom; PD Disaggregation is **spatial isolation across machines** — a cure, treating the root. Because the two stages' hardware needs are opposite, the most thorough solution is not to make them take turns yielding on the same card, but to give them **each its most suitable hardware**: prefill nodes pile on compute, decode nodes pile on VRAM capacity, and RDMA pushes the KV in between and be done with it.

There is also a layer of compiler-level craft that complements PD disaggregation — **XLA's Kernel Fusion**:

```
Unfused (the naïve approach of a generic framework):
  PagedAttention addressing ──write back to HBM──► RoPE rotary position encoding ──write back to HBM──► Softmax
  └ each step's intermediate result is written back to HBM and re-read; the HBM round-trips are killers of latency and bandwidth

Fused (a hand-optimized single TPU kernel):
  ┌─────────────────────────────────────────────────┐
  │  PagedAttention + RoPE + Softmax  fused into one operator │
  │  intermediates stay in registers, no HBM write-back, register-level data reuse │
  └─────────────────────────────────────────────────┘

Another move: fixed block size (e.g. 16 / 32 tokens per block) → turn "dynamic sequence length"
        into "static block-array composition" → avoid dynamic shapes triggering XLA frequent recompilation
```

Both low-level details point at the same thing: **eliminate unnecessary HBM round-trips, eliminate dynamic shapes.** Kernel fusion kneads the three steps of attention addressing, RoPE, and Softmax into one kernel, keeping intermediates in registers instead of writing back and forth to HBM; fixed block size turns "dynamically varying sequence length" — the compiler's most hated thing — into "an array of fixed-size blocks," a static, predictable shape, letting the compiled machine code run at peak efficiency rather than triggering a costly recompilation every time a new length arrives.

> ⚠️ Authenticity Caveat
> "PD disaggregation is Google DeepMind / OpenAI's ultimate evolutionary direction at the infrastructure layer" is the source's internal-vantage claim; the verifiable fact is that PD disaggregation is a real and hot research and engineering direction (public works like DistServe, Splitwise, Mooncake are all doing it), and vLLM already supports disaggregated serving. "Prefill is compute-bound, Decode is memory-bound" is a real, widely measured-and-verified characteristic. What to discount is assertions like "such-and-such top company has fully adopted it internally, such-and-such concrete bandwidth" that cannot be externally checked — take it as "an industry-recognized frontier direction," not "an established reality at some vendor."

> 💡 A Word to the Wise
> A word to the wise: **When you find a system that won't tune well no matter how you turn the knobs, nine times out of ten it's not that a parameter is mis-set, but that you've forced two things of opposite nature into the same box — the real solution is often not to coexist more cleverly, but to admit they don't fit and split the box in two.** The story of Prefill and Decode is the cleanest example of this wisdom. How many teams have exhausted themselves on "letting prefill and decode coexist peacefully on the same card" — tuning batch, tuning priority, doing chunked prefill ever finer — all of it, in essence, marriage counseling for an unhappy couple. Chunked Prefill is brilliant counseling, but it cannot change the fundamental incompatibility of "one wants compute, the other wants VRAM." PD disaggregation's insight is plain to the point of crudeness: **since you two want opposite things, then stop living together.** In distributed systems this has a solemn name, "separation of concerns," yet its hardest part has never been the act of "separating," but **the courage to admit the judgment "putting them together was wrong in the first place"** — because "putting them together" is usually historical baggage, "we've always done it this way," the easy path. **An architect's maturity lies not in how complex a thing he can knead together, but in whether, after everyone has grown used to a certain coupling, he can stand up and say "these two things should never have been together," and then shoulder the migration cost and pry them apart.** Seeing through the mismatch is far harder, and far more valuable, than optimizing the coexistence.

---

## Scene 5 — The Twenty Frontier Armory Directions: A Battle Map Grouped by Strike Team

**Background**: The first four scenes drove home a few main threads in the material. But what the source spreads open at the end is something larger — the **twenty frontier directions** in the field of large-model caching, spanning the latest papers at top conferences (NeurIPS / ICML / ICLR / MLSys), open-source weapons in industrial production, and top-tier architectural methodology. It divides these twenty directions, by "which strike team should fight it," into three layers: **1–6 the algorithm layer** (for algorithm engineers to hack the model), **7–12 the infrastructure layer** (for the infra team to do engine selection and load testing), and **13–20 the architecture-methodology layer** (for architects to lay out the whole board when writing the system HLD). Below, the armory is organized into a battle map — and it is precisely the seed of the fuller cache-technique cheat-sheet in this book's Appendix G.

**[Algorithm layer 1–6] — to the algorithm / research engineers: take the knife to the model structure and the KV itself**

| # | Direction | One-sentence technical core | Representative |
|---|---|---|---|
| 1 | **MLA** | Low-rank compression projects K/V into a low-dim latent space c_t, decompressed on-the-fly at decode, slashing KV volume (see Scene 2) | DeepSeek-V2/V3 |
| 2 | **H2O (Heavy-Hitter Oracle)** | The attention matrix follows a **power-law distribution**, a few Tokens contribute the vast majority of the weight; dynamically identify and keep only these Heavy-Hitters' KV, evicting the rest in real time → a fixed-length cache holds infinitely long text | The *H2O* paper |
| 3 | **StreamingLLM** | Discovered the **Attention Sink** effect (the first 2–4 Tokens lock up huge attention); keep "the initial few tokens + the latest sliding window," enabling streaming infinite decode without recomputing prefill | The *Attention Sinks* paper |
| 4 | **Speculative Decoding + Tree-KV** | Speculative decoding (small model drafts, large model verifies) produces multiple branches; use a **tree-shaped KV Cache** to verify branches in parallel, avoiding redundant multi-path computation | The Medusa family |
| 5 | **Infinite-LLM / Mamba-2 hybrid** | Compress old KV beyond the window into a **fixed-size recurrent state**, space complexity O(N) → O(1) | Griffin / Hawk / Mamba-2 |
| 6 | **KV-Quant (low-bit quantization)** | Per-Channel/Per-Token mixed quantization + non-uniform quantization + outlier preservation, squeezing KV to **2-bit/3-bit with almost no accuracy loss** | The *KV-Quant* paper |

These six directions have one thing in common: they all ask "**can the KV Cache itself be smaller or smarter**" — MLA compresses dimension, H2O and dynamic eviction discard useless tokens, StreamingLLM keeps only head and tail, the Mamba family kneads history into a fixed state, KV-Quant lowers the bit-width. Notice the hidden thread among 1, 2, 3, 5: **they are all wrestling with "attention's long tail"** — H2O says "most tokens don't matter, drop them," StreamingLLM says "the first few and the most recent matter most, keep them," Mamba says "just compress the past into a single state." Three philosophies, answering one question: **facing an infinitely long history, what exactly should be remembered, and what can be forgotten.**

**[Infrastructure layer 7–12] — to the Infra team: engine selection and load testing for inference engines**

| # | Tool | Killer move | Use case |
|---|---|---|---|
| 7 | **vLLM** | Originator of **PagedAttention**, solving VRAM fragmentation; Chunked Prefill / Prefix Caching / tensor parallelism | The industry gold standard, the general first choice |
| 8 | **SGLang** | **RadixAttention** — turning Prompt caching into an automated **Radix Tree** | Multi-turn Agents with complex control flow, Few-shot, Tool-use; strong hit rate and throughput |
| 9 | **LMDeploy** | Near-perfect KV Cache quantization support (W4A16, KV INT4/INT8) + Persistent RPC | Squeezing the limit on edge / on-prem VRAM-constrained scenarios |
| 10 | **TensorRT-LLM** | In-flight Batching, FlashDecoding+, byte-level VRAM control | NVIDIA full stack, paired with Triton multi-node clusters |
| 11 | **LightLLM** | **Token Attention** — token-level fine-grained VRAM scheduling | Ultra-high concurrency, harsh environments with extremely uneven request lengths, OOM-resistant |
| 12 | **DeepSpeed-FastGen** | **Dynamic SplitFuse** — dynamically interleave and slice prefill and decode within the same batch | Microsoft's, optimizing the cache lifecycle of two-stage coexistence |

These six are "ready-to-use weapons" — the key to selection is not "which is strongest," but "**what does your load look like**": general purpose → vLLM; multi-turn Agents, highly reused prompts → SGLang's RadixAttention (which is exactly the single-machine version of Scene 3's affinity routing); VRAM-constrained, needing extreme quantization → LMDeploy; locked into the NVIDIA full stack → TensorRT-LLM.

**[Architecture-methodology layer 13–20] — to the architect / Tech Lead: strategic layout when writing the HLD**

| # | Methodology | Implementation point |
|---|---|---|
| 13 | **Cache-Aware Routing** | The gateway computes a prefix hash with MurmurHash3, routing same-prefix requests to the node holding that KV, maximizing cluster hit rate (see Scene 3) |
| 14 | **Tiered Cache Orchestration** | Three tiers HBM(L1)/DDR5(L2)/NVMe(L3), background threads Offload/Prefetch (see Scene 1) |
| 15 | **Chunked Prefill** | Slice long prefill into fixed chunks, pipeline with decode, prevent long requests starving short ones (see Scene 4) |
| 16 | **Dynamic KV Eviction** | No longer LRU-discarding whole histories crudely; score by **attention importance**, **preferentially drop punctuation, function words, transition words**, keep nouns and core semantic entities — "lossy but high-IQ" reclamation |
| 17 | **Deterministic Prompt Formatting** | "**Static first, dynamic last**"; Tools definitions / System Prompt / Few-shot serialized in fixed order; **absolutely never stuff a UUID, random salt, or timestamp at the prompt's head** (or the prefix never hits) |
| 18 | **Exact Prefix vs Semantic Cache** | Exact: identical tokens reuse the KV via a RadixTree; Semantic: use **GPTCache + Milvus/Qdrant** to compute vector similarity, **>0.98 directly intercept and return the last answer, cost goes to zero** |
| 19 | **PD Disaggregation** | The Prefill node finishes computing KV and RDMA-pushes it to the Decode node (see Scene 4) |
| 20 | **Context Cache Monetization** | Reference **Anthropic Claude's `cache_control`**: explicitly declare cache anchors in code, sharply lowering the input cost of massive long-text (legal contracts, codebases) tasks |

These eight are the architect's "whole-board game." Among them, directions 16 and 17 deserve singling out, because they are the cheapest, most often overlooked, yet highest-return. **Direction 17, deterministic prompt formatting, is almost zero-cost pure discipline** — you don't need to change a line of the model, don't need to buy a card; just hard-write one rule into the team's dev conventions, "the prompt's head is always static content, dynamic things go last," and the prefix cache's hit rate goes from "as good as useless" to "routinely hitting." Conversely, all it takes is one careless engineer stuffing a `request_id=<uuid>` or `current time: ...` at the head of the system prompt to **invalidate the entire team's carefully built prefix cache across the board** — because the very first token of the prefix changed, and nothing after it matching matters anymore. **Direction 18's semantic cache is a cost-cutting nuke of another dimension**: exact cache saves "the recompute of identical prefixes," semantic cache saves "questions of similar meaning, not calling the model at all" — GPTCache uses a vector database to compute similarity, and beyond 0.98 it returns the last answer directly, making this inference's cost zero.

> ⚠️ Authenticity Caveat
> The numbers in the table must be read in tiers: **Claude `cache_control` is a real API feature Anthropic provides, and prompt caching genuinely can sharply lower the input cost of repeated long prefixes** — but "80% cost reduction" is a scenario value dependent on cache hit rate and prefix proportion, not a guaranteed value (it actually depends on how much cache you reused, and cache reads themselves are also billed). "Similarity >0.98 returns directly" is an engineering experience value; set too loose and it returns a cache that answers the wrong question — semantic cache is a double-edged sword, and you must weigh hit rate against correctness yourself. SGLang's "hit rate and throughput several times beyond vLLM" is the result of a specific benchmark; swap the load and it may not hold. **These twenty directions are all real existing papers, tools, or methodologies; what is always to be discounted is the percentages precise to the single digit.**

> 🔍 Deeper Commentary — This map's greatest value is not the twenty tricks, but the grouping line of "who should fight it"
> Memorizing the twenty directions is an assistant engineer's skill; understanding **why they are split into the three layers 1–6 / 7–12 / 13–20** is the architect's vision. This grouping line hides a profound judgment about organization and technology: **these three layers require three fundamentally different kinds of people, moving three fundamentally different things, bearing three fundamentally different risks.** The algorithm layer (1–6) moves the model's mathematical structure; get it wrong and the model loses accuracy, goes dumb — the risk is in "capability," so it must be carried by research engineers who understand attention math and run large-scale ablation experiments — however strong an infra engineer is, he wouldn't dare touch MLA's latent-space dimension. The infrastructure layer (7–12) moves engine selection and load testing; get it wrong and the system OOMs, throughput collapses — the risk is in "stability and cost," requiring systems engineers who can read CUDA kernels, run stress tests, and watch P99 tail latency. The architecture-methodology layer (13–20) moves the cross-service overall layout; get it wrong and the whole cluster slowly bleeds out on problems like "cache in the wrong place" and "non-deterministic prompt format" — problems with **no single point of failure, yet leaking everywhere** — requiring architects who can see routing, tiering, format conventions, and commercial billing all at once on the whiteboard. **The most fatal mismatch is sending the wrong layer's people to fight the wrong war** — having an architect tune MLA's low-rank dimension (he doesn't understand ablation), or having an algorithm engineer design cluster routing (he doesn't understand hotspots and load) — both will blow up. **And the truly top-tier tech leader's value lies exactly on this grouping line: he doesn't need to personally finish all twenty things; what he needs is to accurately judge "is this direction a model problem, a system problem, or an architecture problem," then hand it to the right team, and let the three teams' results align on the co-design whiteboard (the wall of Scene 2).** Knowing who should do a thing is a higher-order and scarcer ability than knowing how to do it — the layering of these twenty directions is less a technical checklist than a **map of how an organization allocates cognition.**

---

## Chapter Summary

- **The core shift**: on a single machine, cache is "VRAM to be spared"; in a cluster, cache is "a strategic asset to be scheduled" — it has a location, an affinity, a migration cost, and like compute, it's the kind of thing that gets an SRE out of bed at three a.m.
- **Scene 1 — the three-tier heterogeneous pyramid**: L1 HBM (active decode) / L2 DDR5 (momentarily inactive multi-turn history) / L3 NVMe (cold data), transplanting the OS's virtual memory onto the KV Cache; the soul is **asynchronous prefetch + Overlapping** — use someone else's decode time to mask my hauling latency.
- **Scene 2 — slimming at the structure layer**: MHA → MQA (cut to 1 KV group, too damaging to ability) → GQA (grouped sharing, industry mainstream, Llama 3) → MLA (DeepSeek low-rank compression into latent space c_t, decompressed on-the-fly, KV greatly slashed). The key is **algorithm-infra co-design** — tear down the wall between the two teams.
- **Scene 3 — cache-aware routing**: Round-Robin is a disaster in stateful inference (the same user scattered across nodes, prefill recomputed); the right answer = prefix hashing (MurmurHash3) + affinity routing + speculative cache migration (the accounting of recompute vs RDMA pull). The reef: **cache affinity and load balancing are born to be at odds**, requiring dynamic rebalancing with load awareness.
- **Scene 4 — PD disaggregation**: Prefill is compute-bound, Decode is memory-bound, their hardware needs opposite; Chunked Prefill is same-machine time-slicing (treating symptoms), **PD disaggregation is cross-machine spatial isolation (treating the root)** — admit the two don't fit, split them into two dedicated node types, RDMA-push the KV. Pair with XLA kernel fusion (eliminating HBM round-trips) + fixed block size (eliminating dynamic-shape recompilation).
- **Scene 5 — the twenty-direction armory**: algorithm layer 1–6 (MLA / H2O / StreamingLLM / Tree-KV / Mamba hybrid / KV-Quant) to the research engineers; infra layer 7–12 (vLLM / SGLang / LMDeploy / TensorRT-LLM / LightLLM / DeepSpeed-FastGen) to infra; architecture layer 13–20 (routing / tiering / Chunked Prefill / dynamic eviction / deterministic formatting / semantic cache / PD disaggregation / Claude `cache_control` monetization) to the architects. **The grouping line is worth more than the list itself** — knowing who should do a thing is scarcer than knowing how to do it.
- **Authenticity discipline**: MLA, H2O, StreamingLLM, KV-Quant, vLLM, SGLang, Claude `cache_control` and the rest are all real work; what is always to be discounted is precise numbers like "cut 90%, cost down 80%, several-fold beyond, Jupiter 1.6 Tbps" and claims about "how some top lab does it internally."

These twenty directions are only seeds. Gathering them, together with the caching tricks scattered across the previous eleven chapters — from Chapter 3's UMA zero-copy, Chapter 5's KV Cache FP8 quantization, to this chapter's three-tier pyramid and MLA — into a cheat-sheet you can flip open at any time, is exactly what **Appendix G "Cache Technique Cheat-Sheet"** sets out to do: core concepts, VRAM compression, inference engines, architectural methodology, low-level operators, semantic cache, hit-rate pitfalls, plus a block that separately pins down numbers like "cut 90%, cost down 80%, Jupiter 1.6 Tbps" in a dedicated authenticity-calibration zone.

---

## Coda — Three Movements, One Spine

The armory's door swings shut, and the whole book reaches its end. Looking back over this journey, it really only asked one question, and asked it three times.

In the first movement, we asked "**how to build it**" — under that anti-distillation shield, to lawfully distill out a local model that belongs to you alone and runs on your desk. In the second movement, we asked "**how to use it**" — to grow eyes and hands on this small, cheap model, to make it understand and operate a computer designed for humans, then to look up and question the entire paradigm this craft stands upon. In the third movement, we asked "**how to sustain it**" — when these eyes and hands have to scale from a demo to ten-thousand-fold concurrency, to fight that most invisible war of memory and cache, so that the promise of "cheap" doesn't bounce a check at scale.

Build it, use it, sustain it. You'll find these three things are simply inseparable: every GB of KV Cache you fought over in the third movement, every one you saved, was saved to let the DGX Spark of the first movement hold up the never-closing eyes of the second movement. **This is not an anthology of three themes pinned together; it is the same engineering ideal — "an intelligence that truly belongs to you, listens to you, is cheap enough to be an organ and strong enough to be worth sustaining" — interrogated to the bottom from three angles.**

What the whole book honed over and over is, in fact, only two things. A **ruler of authenticity**: in the face of that string of seductive names (Mythos, Qwythos, Aluminium OS, Jupiter 1.6 Tbps, 78.4 points), instinctively telling apart "what I can verify" from "what someone wants me to believe" — this line divides the engineer who thinks from the loudspeaker who recites. And a **clean path**: technology is always neutral; the watershed is in intent and authorization — distilling a model you have the right to use, with lawful data, sustaining it on your own machine, is engineering; bypassing someone else's shield to steal a shadow is an attack.

Technology will go out of date; every model, every score, every conference name in this book may be forgotten by no one next year. But that ruler and that path will outlive any model.

> ⚠️ Closing Disclaimer
> This book is for educational and research purposes, an organization and critical commentary spanning the three great themes of distillation, agent operation, and cache. All descriptions in it of unauthorized capability extraction, bypassing access controls, or violating terms of service **are risk disclosures, not operational advice**; all unverified product names, specs, and numbers have been flagged nearby with ⚠️, and should be taken as "plausible engineering hypotheses" rather than established fact. May you use this book to **build, use well, and afford to sustain an intelligence that truly belongs to you — rather than to steal a shadow.**

The seven appendices that follow are the toolbox and calibration instrument this journey leaves you, ready to flip open anytime: **Appendix A** Models and Datasets Cheat-Sheet, **B** Terms and Tools Cheat-Sheet, **C** Controversies and Cognition Q&A, **D** Evaluation Question-Bank Design, **E** De-censoring and Safety Assessment, **F** Agent-Operation Tools and Papers Cheat-Sheet, **G** Cache Technique Cheat-Sheet. Put the directions into your head and leave the sources in the appendices for re-checking anytime — this, in the end, is what the whole book wants to place in your hands.
