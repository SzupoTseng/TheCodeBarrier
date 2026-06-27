# 附录G　大模型缓存（LLM Cache）技术速查

> 把第 10–11 章出现的「KV Cache / Prompt Cache / 推理引擎 / 架构方法论」生态，按主题收拢成对照表。**多数条目为真实、公开的论文或开源项目**（可自行查证），标 ✅；少数出现在书中叙事的「内部架构声称 / 具体规格 / 节省百分比」属未经核实，标 ⚠️，并于文末校准区集中说明。引用前仍请自行核对版本与授权（见第 6 章六条铁律）。

## G.1　内核概念

| 名词 | 一句话解释 | 备注 |
|---|---|---|
| **KV Cache（键值缓存）** | 把自回归生成时已算好的 Key/Value 矩阵存在显存，避免重算，将注意力复杂度从 O(N²) 降到 O(N) | 推理层缓存 ✅ |
| **Prompt Cache / Context Cache** | API/服务层把相同**前缀**的 KV 状态缓存在伺服端，省掉重复 Prefill；命中后输入 Token 费用常打 1–2 折 | 服务层缓存 ✅ |
| **Prefill vs Decode** | Prefill＝一次平行处理整段输入（算力受限）；Decode＝逐 Token 生成（访存受限）；两阶段特性相反 | ✅ |
| **Memory-Bound vs Compute-Bound** | 长文本 KV Cache 动辄数 GB，把 GPU 从「算力受限」推向「访存带宽受限」，读缓存比重算还慢 | ✅ |
| **Attention Sink** | 模型最初 2–4 个 Token 锁定异常大的注意力权重，是 StreamingLLM 流式解码的关键发现 | ✅ |
| **显存碎片化（Fragmentation）** | 传统 KV Cache 须为每请求预分配连续最大显存，造成大量闲置浪费 → PagedAttention 解 | ✅ |

## G.2　显存压缩 / 注意力结构

| 名称 | 做什么 | 出处 / 来源 |
|---|---|---|
| **MHA（Multi-Head Attention）** | 经典多头注意力，KV 头数＝Q 头数，KV Cache 开销最大 | Vaswani et al., 2017 ✅ |
| **MQA（Multi-Query Attention）** | 所有 Q 头共享一对 K/V，KV Cache 降为 1/H，但表达力受损 | Shazeer, 2019 ✅ |
| **GQA（Grouped-Query Attention）** | 折中：Q 头分组各共享一对 KV（如 Llama 3 用 8 对），业界主流 | Ainslie et al., 2023（Llama 采用）✅ |
| **MLA（Multi-head Latent Attention）** | 低秩压缩把 K/V 投影到极低维隐空间矢量，解码时在线解压；KV Cache 砍 90%+ | DeepSeek-V2/V3 提出 ✅；「DeepMind 深度跟进」⚠️ |
| **KV-Quant（2-bit 量化）** | Per-Channel/Per-Token 混合量化＋离群值保留，把 KV 压到 2–3 bit 几乎不掉点 | 论文《KVQuant…》（NeurIPS 2024）✅；「100B Token 窗口」标题 ⚠️ |
| **H2O（Heavy-Hitter Oracle）** | 发现注意力呈幂律分布，只保留少数 Heavy-Hitter Token 的 KV，动态剔除其余 → 定长缓存 | 论文《H2O…》NeurIPS 2023 ✅ |
| **StreamingLLM** | 保留「初始几个 Token（Attention Sink）」＋「滑动窗口」，免重算 Prefill 达流式无限长文本 | 论文《Efficient Streaming LM with Attention Sinks》ICLR 2024 ✅ |
| **Speculative Decoding + Cache** | 投机解码用树状 KV Cache 平行校验草稿分支，避免多路重复计算 | Medusa / 树状 KV 系列 ✅ |
| **Infinite-LLM / 混合 RNN-Transformer** | 把超窗 KV 压进固定大小循环状态，空间复杂度 O(N)→O(1) | Griffin/Hawk、Mamba-2 路线 ✅ |

## G.3　推理引擎（开源 / 半开源）

| 引擎 | 内核杀手锏 | 出处 / 来源 |
|---|---|---|
| **vLLM** | **PagedAttention**：借操作系统虚拟内存分页管理 KV，显存利用率近 100%；Chunked Prefill / Prefix Caching | UC Berkeley，开源 ✅ |
| **SGLang** | **RadixAttention**：用基数树（Radix Tree）自动管理 Prompt 前缀缓存，多轮/Tool-use 命中率高 | UC Berkeley，开源 ✅ |
| **LMDeploy** | KV Cache 量化（W4A16、KV INT4/INT8）极完善＋Persistent RPC，适合边缘/私有化 | 上海 AI Lab（书生·浦语），开源 ✅ |
| **TensorRT-LLM** | In-flight Batching、FlashDecoding+；字节级显存控制，与 Triton 集成 | NVIDIA，半开源 ✅ |
| **LightLLM** | **Token Attention**：Token 级细粒度显存调度，请求长度极不均时抗 OOM | 开源 ✅ |
| **DeepSpeed-FastGen** | **Split-Fused / Dynamic SplitFuse**：Prefill 与 Decode 在同 Batch 动态交织切分 | Microsoft，开源 ✅ |
| **Hugging Face TGI** | 生产级推理服务（连续批量、量化、Tensor 平行），含 Prefix Caching | Hugging Face，开源 ✅ |

## G.4　架构方法论

| 方法 | 一句话 | 备注 |
|---|---|---|
| **Cache-Aware Routing** | 网关对 System Prompt 算前缀哈希，把相同前缀请求路由到持有该物理缓存的同一节点 | ✅（概念/实践）|
| **Tiered Cache Orchestration** | HBM＝L1、DDR5＝L2、NVMe SSD＝L3；闲置 KV 背景 Offload/Prefetch 换出换回 | ✅（PCIe Gen5 速率为情境举例 ⚠️）|
| **Chunked Prefill** | 把长文 Prefill 拆成固定大小 Chunk（如 512 Token），与 Decode 流水线交错，降 TTFT 抖动 | ✅ |
| **Dynamic KV Eviction** | 结合注意力权重算 Token 重要度，优先释放标点/虚词，保留实体名词（有损但高智能）| ✅ |
| **Deterministic Prompt Formatting** | 「静态在前、动态在后」；Tools/System/Few-shot 固定字母序串行化，禁开头塞 UUID/时间戳 | ✅ |
| **Exact vs Semantic Cache** | Exact＝Token 完全一致走 RadixTree 复用 KV；Semantic＝矢量相似度 >0.98 直接回上次答案 | ✅ |
| **PD Disaggregation（预填/解码分离）** | Prefill 节点（高算力）算完 KV，经 RDMA 推给 Decode 节点（大显存）续算 | ✅（趋势）；具体基建细节 ⚠️ |
| **Context Cache Monetization** | 用 `cache_control` 显式声明缓存锚点，长文任务输入费用大幅下降 | Anthropic 机制 ✅；「暴跌 80%」为情境数字 ⚠️ |

## G.5　底层算子 / 系统原语

| 名词 | 一句话 | 来源 |
|---|---|---|
| **FlashAttention** | IO 感知、分块（tiling）融合的注意力核，省 HBM 读写 | Dao et al., 2022（v2/v3 后续）✅ |
| **FlashDecoding** | 针对 Decode 阶段沿串行维度平行化，加速长上下文单 Token 生成 | Stanford/FlashAttention 团队 ✅ |
| **PagedAttention** | 把 KV 离散存固定大小「记忆页」，消碎片、支持共享/Copy-on-Write | vLLM 论文 SOSP 2023 ✅ |
| **Radix Tree** | 压缩前缀树，SGLang 用来自动共享/复用多请求的 Prompt KV 前缀 | 数据结构（SGLang 应用）✅ |
| **RoPE（旋转位置编码）** | 旋转式相对位置编码，与 KV Cache 偏移、长度外推相关 | Su et al., 2021 ✅ |
| **MurmurHash3 前缀哈希** | 非加密快速哈希，Cache-Aware Routing 用来算前缀指纹分流 | 通用哈希（Appleby）✅ |
| **RDMA KV transfer** | 跨节点以远程直接内存访问「推」KV Cache，PD 分离的传输基础 | InfiniBand/RoCE ✅；「Jupiter 1.6 Tbps」⚠️ |

## G.6　语意缓存生态

| 名称 | 角色 | 来源 |
|---|---|---|
| **GPTCache** | 语意缓存中介层：把相似输入映射到既有回答，命中即免调用 LLM | 开源（Zilliz）✅ |
| **Milvus** | 大规模矢量数据库，语意缓存的相似度检索后端 | 开源（Zilliz）✅ |
| **Qdrant** | Rust 矢量数据库，语意缓存/RAG 常用检索后端 | 开源 ✅ |
| **ANN 相似度检索** | 近似最近邻（HNSW/IVF 等）做 Embedding 相似度匹配，>0.98 视为命中 | 通用方法 ✅ |

## G.7　命中率避坑（第 11 章「黄金法则」）

| 常见错误实践（导致 Cache 失效）| 正确策略（最大化命中）|
|---|---|
| System Prompt 塞动态变量（时间戳、用户 ID）| 动态用户信息移到消息**最末尾**的 User 栏，开头固定 |
| 每轮重新生成历史摘要（Summary）| 历史**只追加不修改**；截断从**最旧**开始删，保前缀一致 |
| 工具定义（Tools）顺序随机化 | 严格固定 JSON 工具列表内部排序、字段结构与前后顺序 |
| 空格/换行等细节不统一 | 统一文本清洗逻辑，防一个 `\n` 改变 Token 串行 |
| 开头插入随机盐（Salt）/UUID | 用 `cache_control` 显式锚点；确定性串行化（Deterministic Ordering）|

> **黄金法则**：「**稳定的静态内容放最前、动态高变内容放最后**」——前缀匹配是一切 Prompt Cache 的根。

---

> ⚠️ **真伪校准 — 第 10–11 章叙事中需谨慎看待的声称**
>
> G.1–G.7 表中标 ✅ 者，均为可在 arXiv / GitHub / 官方文档查证的**真实论文或开源工具**（MLA、H2O、StreamingLLM、KV-Quant、vLLM、SGLang、LMDeploy、TensorRT-LLM、LightLLM、DeepSpeed、TGI、FlashAttention、PagedAttention、GPTCache、Milvus、Qdrant、Anthropic `cache_control` 等）。
>
> 下列在书中叙事出现的内容**本书无法核实**，应视为**合理工程推演而非既成事实**：
> - **「DeepMind 内部架构」相关声称** ⚠️（MLA「DeepMind 也深度跟进」、PD 分离「Google DeepMind/OpenAI 终极方向」）——方向合理但无公开出处。
> - **特定 Gemini 上下文窗口大小 / 具体规格** ⚠️。
> - **「Google Jupiter 1.6 Tbps RDMA」** ⚠️（具体带宽数字未经查证）。
> - **精确节省百分比**（「KV Cache 砍 90%+」「输入费用暴跌 80%」「相似度 >0.98」「2-bit 几乎不掉点」）⚠️——量级方向可参考，**确切数字依模型/负载而异**，不应当作保证值引用。
> - **PCIe Gen5 / HBM3e 等硬件型号**为情境举例，实际部署以官方规格为准。
