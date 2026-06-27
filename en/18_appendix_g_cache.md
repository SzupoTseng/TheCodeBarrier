# Appendix G — LLM Cache Techniques Quick Reference

> A set of cross-reference tables that gathers the "KV Cache / Prompt Cache / inference engine / architecture methodology" ecosystem from Chapters 10–11 by topic. **Most entries are real, publicly available papers or open-source projects** (you can verify them yourself), marked ✅; a few "internal-architecture claims / specific specs / savings percentages" that appear in the book's narrative are unverified, marked ⚠️, and explained together in the calibration block at the end. Before citing, still check the version and license yourself (see the six iron rules of Chapter 6).

## G.1　Core Concepts

| Term | One-line explanation | Note |
|---|---|---|
| **KV Cache** | Store the already-computed Key/Value matrices from autoregressive generation in VRAM to avoid recomputation, reducing attention complexity from O(N²) to O(N) | Inference-layer cache ✅ |
| **Prompt Cache / Context Cache** | The API/service layer caches the KV state of an identical **prefix** server-side, skipping the repeated Prefill; once it hits, input-token cost is often discounted by 80–90% | Service-layer cache ✅ |
| **Prefill vs Decode** | Prefill = process the whole input in parallel in one pass (compute-bound); Decode = generate token by token (memory-bound); the two phases have opposite characteristics | ✅ |
| **Memory-Bound vs Compute-Bound** | A long-context KV Cache easily runs to several GB, pushing the GPU from "compute-bound" toward "memory-bandwidth-bound" — reading the cache becomes slower than recomputing | ✅ |
| **Attention Sink** | The model's first 2–4 tokens lock onto abnormally large attention weights — the key discovery behind StreamingLLM's streaming decode | ✅ |
| **Fragmentation** | A traditional KV Cache must pre-allocate the maximum contiguous VRAM per request, causing large amounts of idle waste → solved by PagedAttention | ✅ |

## G.2　VRAM Compression / Attention Structures

| Name | What it does | Origin / source |
|---|---|---|
| **MHA (Multi-Head Attention)** | Classic multi-head attention, KV heads = Q heads, the largest KV Cache footprint | Vaswani et al., 2017 ✅ |
| **MQA (Multi-Query Attention)** | All Q heads share a single K/V pair, cutting the KV Cache to 1/H but hurting expressiveness | Shazeer, 2019 ✅ |
| **GQA (Grouped-Query Attention)** | A compromise: Q heads are grouped, each group sharing one KV pair (e.g. Llama 3 uses 8 pairs), the industry mainstream | Ainslie et al., 2023 (adopted by Llama) ✅ |
| **MLA (Multi-head Latent Attention)** | Low-rank compression projects K/V into a very-low-dimensional latent vector, decompressed online at decode time; cuts the KV Cache by 90%+ | Proposed in DeepSeek-V2/V3 ✅; "DeepMind following up in depth" ⚠️ |
| **KV-Quant (2-bit quantization)** | Mixed per-channel/per-token quantization + outlier preservation, compressing KV to 2–3 bit with almost no quality loss | Paper *KVQuant…* (NeurIPS 2024) ✅; the "100B-token window" headline ⚠️ |
| **H2O (Heavy-Hitter Oracle)** | Finds that attention follows a power-law distribution, keeps only the KV of a few Heavy-Hitter tokens and dynamically evicts the rest → fixed-length cache | Paper *H2O…*, NeurIPS 2023 ✅ |
| **StreamingLLM** | Keeps "the initial few tokens (Attention Sink)" + "a sliding window," avoiding Prefill recomputation to stream effectively unbounded text | Paper *Efficient Streaming LM with Attention Sinks*, ICLR 2024 ✅ |
| **Speculative Decoding + Cache** | Speculative decoding uses a tree-shaped KV Cache to verify draft branches in parallel, avoiding duplicate computation across paths | Medusa / tree-KV family ✅ |
| **Infinite-LLM / hybrid RNN-Transformer** | Compresses over-window KV into a fixed-size recurrent state, reducing space complexity from O(N) to O(1) | Griffin/Hawk, Mamba-2 line ✅ |

## G.3　Inference Engines (Open / Semi-Open Source)

| Engine | Killer feature | Origin / source |
|---|---|---|
| **vLLM** | **PagedAttention**: borrows OS virtual-memory paging to manage KV, near-100% VRAM utilization; Chunked Prefill / Prefix Caching | UC Berkeley, open source ✅ |
| **SGLang** | **RadixAttention**: uses a radix tree to automatically manage prompt-prefix caching, with high hit rates for multi-turn / tool-use | UC Berkeley, open source ✅ |
| **LMDeploy** | Very complete KV Cache quantization (W4A16, KV INT4/INT8) + persistent RPC, well suited to edge / on-prem | Shanghai AI Lab (InternLM), open source ✅ |
| **TensorRT-LLM** | In-flight Batching, FlashDecoding+; byte-level VRAM control, integrated with Triton | NVIDIA, semi-open source ✅ |
| **LightLLM** | **Token Attention**: token-level fine-grained VRAM scheduling, resists OOM when request lengths are highly uneven | Open source ✅ |
| **DeepSpeed-FastGen** | **Split-Fused / Dynamic SplitFuse**: Prefill and Decode are dynamically interleaved and split within the same batch | Microsoft, open source ✅ |
| **Hugging Face TGI** | Production-grade inference serving (continuous batching, quantization, tensor parallelism), including Prefix Caching | Hugging Face, open source ✅ |

## G.4　Architecture Methodology

| Method | One line | Note |
|---|---|---|
| **Cache-Aware Routing** | The gateway computes a prefix hash of the System Prompt and routes requests with the same prefix to the same node that holds that physical cache | ✅ (concept/practice) |
| **Tiered Cache Orchestration** | HBM = L1, DDR5 = L2, NVMe SSD = L3; idle KV is offloaded/prefetched in the background, swapping out and back in | ✅ (the PCIe Gen5 rate is an illustrative example ⚠️) |
| **Chunked Prefill** | Split a long-text Prefill into fixed-size chunks (e.g. 512 tokens), pipelined and interleaved with Decode, reducing TTFT jitter | ✅ |
| **Dynamic KV Eviction** | Combines attention weights to score token importance, preferentially releasing punctuation/function words while keeping entity nouns (lossy but highly intelligent) | ✅ |
| **Deterministic Prompt Formatting** | "Static first, dynamic last"; serialize Tools/System/Few-shot in fixed alphabetical order, and never stuff a UUID/timestamp at the start | ✅ |
| **Exact vs Semantic Cache** | Exact = an identical token sequence reuses the KV via the RadixTree; Semantic = vector similarity > 0.98 directly returns the previous answer | ✅ |
| **PD Disaggregation (prefill/decode separation)** | The Prefill node (high compute) computes the KV and pushes it via RDMA to the Decode node (large VRAM) to continue | ✅ (a trend); specific infra details ⚠️ |
| **Context Cache Monetization** | Use `cache_control` to explicitly declare cache anchors, sharply lowering input cost for long-context tasks | Anthropic mechanism ✅; "an 80% plunge" is a scenario figure ⚠️ |

## G.5　Low-Level Operators / System Primitives

| Term | One line | Source |
|---|---|---|
| **FlashAttention** | An IO-aware, tiling-fused attention kernel that saves HBM reads/writes | Dao et al., 2022 (v2/v3 follow-ups) ✅ |
| **FlashDecoding** | Parallelizes along the sequence dimension specifically for the Decode phase, speeding up single-token generation in long contexts | Stanford/FlashAttention team ✅ |
| **PagedAttention** | Stores KV discretely in fixed-size "memory pages," eliminating fragmentation and supporting sharing / Copy-on-Write | vLLM paper, SOSP 2023 ✅ |
| **Radix Tree** | A compressed prefix tree that SGLang uses to automatically share/reuse the prompt-KV prefix across requests | Data structure (SGLang application) ✅ |
| **RoPE (Rotary Position Embedding)** | Rotary relative position encoding, related to KV Cache offsets and length extrapolation | Su et al., 2021 ✅ |
| **MurmurHash3 prefix hashing** | A non-cryptographic fast hash, used by Cache-Aware Routing to compute prefix fingerprints for routing | General-purpose hash (Appleby) ✅ |
| **RDMA KV transfer** | "Pushes" the KV Cache across nodes via remote direct memory access — the transport foundation of PD separation | InfiniBand/RoCE ✅; "Jupiter 1.6 Tbps" ⚠️ |

## G.6　Semantic-Cache Ecosystem

| Name | Role | Source |
|---|---|---|
| **GPTCache** | A semantic-cache middleware layer that maps similar inputs to existing answers, skipping the LLM call on a hit | Open source (Zilliz) ✅ |
| **Milvus** | A large-scale vector database, the similarity-retrieval backend for semantic caching | Open source (Zilliz) ✅ |
| **Qdrant** | A Rust vector database, a common retrieval backend for semantic caching / RAG | Open source ✅ |
| **ANN similarity retrieval** | Approximate nearest neighbor (HNSW/IVF, etc.) for embedding-similarity matching, with > 0.98 treated as a hit | General method ✅ |

## G.7　Hit-Rate Pitfalls (Chapter 11's "Golden Rules")

| Common bad practice (causes cache misses) | Correct strategy (maximizes hits) |
|---|---|
| Stuffing dynamic variables (timestamps, user IDs) into the System Prompt | Move dynamic user info to the User field at the very **end** of the message; keep the start fixed |
| Regenerating the history summary every turn | History is **append-only, never modified**; truncate from the **oldest** end to keep the prefix consistent |
| Randomizing the order of tool definitions (Tools) | Strictly fix the internal ordering, field structure, and sequence of the JSON tool list |
| Inconsistent details like spaces/newlines | Unify text-cleaning logic to prevent a single `\n` from changing the token sequence |
| Inserting a random salt / UUID at the start | Use `cache_control` as an explicit anchor; deterministic serialization (Deterministic Ordering) |

> **Golden Rule**: "**Put stable, static content first; put dynamic, high-variance content last**" — prefix matching is the root of every Prompt Cache.

---

> ⚠️ **Authenticity Calibration — claims in the Chapter 10–11 narrative to treat with care**
>
> Every entry marked ✅ in tables G.1–G.7 is a **real paper or open-source tool** verifiable on arXiv / GitHub / official docs (MLA, H2O, StreamingLLM, KV-Quant, vLLM, SGLang, LMDeploy, TensorRT-LLM, LightLLM, DeepSpeed, TGI, FlashAttention, PagedAttention, GPTCache, Milvus, Qdrant, Anthropic `cache_control`, etc.).
>
> The following content that appears in the book's narrative **cannot be verified by this book** and should be treated as **reasonable engineering extrapolation rather than established fact**:
> - **"DeepMind internal architecture" claims** ⚠️ (MLA's "DeepMind is also following up in depth," PD separation as "Google DeepMind/OpenAI's ultimate direction") — plausible in direction but with no public origin.
> - **Specific Gemini context-window sizes / concrete specs** ⚠️.
> - **"Google Jupiter 1.6 Tbps RDMA"** ⚠️ (the specific bandwidth figure is unverified).
> - **Exact savings percentages** ("KV Cache cut by 90%+," "input cost plunges 80%," "similarity > 0.98," "2-bit with almost no quality loss") ⚠️ — the order of magnitude is a useful reference, but the **exact numbers vary by model/workload** and should not be cited as guaranteed values.
> - **Hardware model numbers like PCIe Gen5 / HBM3e** are illustrative examples; defer to official specs for actual deployment.
