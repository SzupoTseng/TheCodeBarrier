# 附錄G　大模型快取（LLM Cache）技術速查

> 把第 10–11 章出現的「KV Cache / Prompt Cache / 推理引擎 / 架構方法論」生態，按主題收攏成對照表。**多數條目為真實、公開的論文或開源專案**（可自行查證），標 ✅；少數出現在書中敘事的「內部架構聲稱 / 具體規格 / 節省百分比」屬未經核實，標 ⚠️，並於文末校準區集中說明。引用前仍請自行核對版本與授權（見第 6 章六條鐵律）。

## G.1　核心概念

| 名詞 | 一句話解釋 | 備註 |
|---|---|---|
| **KV Cache（鍵值快取）** | 把自回歸生成時已算好的 Key/Value 矩陣存在顯存，避免重算，將注意力複雜度從 O(N²) 降到 O(N) | 推理層快取 ✅ |
| **Prompt Cache / Context Cache** | API/服務層把相同**前綴**的 KV 狀態快取在伺服端，省掉重複 Prefill；命中後輸入 Token 費用常打 1–2 折 | 服務層快取 ✅ |
| **Prefill vs Decode** | Prefill＝一次平行處理整段輸入（算力受限）；Decode＝逐 Token 生成（訪存受限）；兩階段特性相反 | ✅ |
| **Memory-Bound vs Compute-Bound** | 長文本 KV Cache 動輒數 GB，把 GPU 從「算力受限」推向「訪存頻寬受限」，讀快取比重算還慢 | ✅ |
| **Attention Sink** | 模型最初 2–4 個 Token 鎖定異常大的注意力權重，是 StreamingLLM 流式解碼的關鍵發現 | ✅ |
| **顯存碎片化（Fragmentation）** | 傳統 KV Cache 須為每請求預分配連續最大顯存，造成大量閒置浪費 → PagedAttention 解 | ✅ |

## G.2　顯存壓縮 / 注意力結構

| 名稱 | 做什麼 | 出處 / 來源 |
|---|---|---|
| **MHA（Multi-Head Attention）** | 經典多頭注意力，KV 頭數＝Q 頭數，KV Cache 開銷最大 | Vaswani et al., 2017 ✅ |
| **MQA（Multi-Query Attention）** | 所有 Q 頭共享一對 K/V，KV Cache 降為 1/H，但表達力受損 | Shazeer, 2019 ✅ |
| **GQA（Grouped-Query Attention）** | 折中：Q 頭分組各共享一對 KV（如 Llama 3 用 8 對），業界主流 | Ainslie et al., 2023（Llama 採用）✅ |
| **MLA（Multi-head Latent Attention）** | 低秩壓縮把 K/V 投影到極低維隱空間向量，解碼時線上解壓；KV Cache 砍 90%+ | DeepSeek-V2/V3 提出 ✅；「DeepMind 深度跟進」⚠️ |
| **KV-Quant（2-bit 量化）** | Per-Channel/Per-Token 混合量化＋離群值保留，把 KV 壓到 2–3 bit 幾乎不掉點 | 論文《KVQuant…》（NeurIPS 2024）✅；「100B Token 視窗」標題 ⚠️ |
| **H2O（Heavy-Hitter Oracle）** | 發現注意力呈冪律分佈，只保留少數 Heavy-Hitter Token 的 KV，動態剔除其餘 → 定長快取 | 論文《H2O…》NeurIPS 2023 ✅ |
| **StreamingLLM** | 保留「初始幾個 Token（Attention Sink）」＋「滑動窗口」，免重算 Prefill 達流式無限長文本 | 論文《Efficient Streaming LM with Attention Sinks》ICLR 2024 ✅ |
| **Speculative Decoding + Cache** | 投機解碼用樹狀 KV Cache 平行校驗草稿分支，避免多路重複計算 | Medusa / 樹狀 KV 系列 ✅ |
| **Infinite-LLM / 混合 RNN-Transformer** | 把超窗 KV 壓進固定大小循環狀態，空間複雜度 O(N)→O(1) | Griffin/Hawk、Mamba-2 路線 ✅ |

## G.3　推理引擎（開源 / 半開源）

| 引擎 | 核心殺手鐧 | 出處 / 來源 |
|---|---|---|
| **vLLM** | **PagedAttention**：借作業系統虛擬記憶體分頁管理 KV，顯存利用率近 100%；Chunked Prefill / Prefix Caching | UC Berkeley，開源 ✅ |
| **SGLang** | **RadixAttention**：用基數樹（Radix Tree）自動管理 Prompt 前綴快取，多輪/Tool-use 命中率高 | UC Berkeley，開源 ✅ |
| **LMDeploy** | KV Cache 量化（W4A16、KV INT4/INT8）極完善＋Persistent RPC，適合邊緣/私有化 | 上海 AI Lab（書生·浦語），開源 ✅ |
| **TensorRT-LLM** | In-flight Batching、FlashDecoding+；位元組級顯存控制，與 Triton 整合 | NVIDIA，半開源 ✅ |
| **LightLLM** | **Token Attention**：Token 級細粒度顯存調度，請求長度極不均時抗 OOM | 開源 ✅ |
| **DeepSpeed-FastGen** | **Split-Fused / Dynamic SplitFuse**：Prefill 與 Decode 在同 Batch 動態交織切分 | Microsoft，開源 ✅ |
| **Hugging Face TGI** | 生產級推理服務（連續批次、量化、Tensor 平行），含 Prefix Caching | Hugging Face，開源 ✅ |

## G.4　架構方法論

| 方法 | 一句話 | 備註 |
|---|---|---|
| **Cache-Aware Routing** | 網關對 System Prompt 算前綴雜湊，把相同前綴請求路由到持有該物理快取的同一節點 | ✅（概念/實踐）|
| **Tiered Cache Orchestration** | HBM＝L1、DDR5＝L2、NVMe SSD＝L3；閒置 KV 背景 Offload/Prefetch 換出換回 | ✅（PCIe Gen5 速率為情境舉例 ⚠️）|
| **Chunked Prefill** | 把長文 Prefill 拆成固定大小 Chunk（如 512 Token），與 Decode 流水線交錯，降 TTFT 抖動 | ✅ |
| **Dynamic KV Eviction** | 結合注意力權重算 Token 重要度，優先釋放標點/虛詞，保留實體名詞（有損但高智慧）| ✅ |
| **Deterministic Prompt Formatting** | 「靜態在前、動態在後」；Tools/System/Few-shot 固定字母序序列化，禁開頭塞 UUID/時間戳 | ✅ |
| **Exact vs Semantic Cache** | Exact＝Token 完全一致走 RadixTree 復用 KV；Semantic＝向量相似度 >0.98 直接回上次答案 | ✅ |
| **PD Disaggregation（預填/解碼分離）** | Prefill 節點（高算力）算完 KV，經 RDMA 推給 Decode 節點（大顯存）續算 | ✅（趨勢）；具體基建細節 ⚠️ |
| **Context Cache Monetization** | 用 `cache_control` 顯式宣告快取錨點，長文任務輸入費用大幅下降 | Anthropic 機制 ✅；「暴跌 80%」為情境數字 ⚠️ |

## G.5　底層算子 / 系統原語

| 名詞 | 一句話 | 來源 |
|---|---|---|
| **FlashAttention** | IO 感知、分塊（tiling）融合的注意力核，省 HBM 讀寫 | Dao et al., 2022（v2/v3 後續）✅ |
| **FlashDecoding** | 針對 Decode 階段沿序列維度平行化，加速長上下文單 Token 生成 | Stanford/FlashAttention 團隊 ✅ |
| **PagedAttention** | 把 KV 離散存固定大小「記憶頁」，消碎片、支援共享/Copy-on-Write | vLLM 論文 SOSP 2023 ✅ |
| **Radix Tree** | 壓縮前綴樹，SGLang 用來自動共享/復用多請求的 Prompt KV 前綴 | 資料結構（SGLang 應用）✅ |
| **RoPE（旋轉位置編碼）** | 旋轉式相對位置編碼，與 KV Cache 偏移、長度外推相關 | Su et al., 2021 ✅ |
| **MurmurHash3 前綴雜湊** | 非加密快速雜湊，Cache-Aware Routing 用來算前綴指紋分流 | 通用雜湊（Appleby）✅ |
| **RDMA KV transfer** | 跨節點以遠端直接記憶體存取「推」KV Cache，PD 分離的傳輸基礎 | InfiniBand/RoCE ✅；「Jupiter 1.6 Tbps」⚠️ |

## G.6　語意快取生態

| 名稱 | 角色 | 來源 |
|---|---|---|
| **GPTCache** | 語意快取中介層：把相似輸入映射到既有回答，命中即免調用 LLM | 開源（Zilliz）✅ |
| **Milvus** | 大規模向量資料庫，語意快取的相似度檢索後端 | 開源（Zilliz）✅ |
| **Qdrant** | Rust 向量資料庫，語意快取/RAG 常用檢索後端 | 開源 ✅ |
| **ANN 相似度檢索** | 近似最近鄰（HNSW/IVF 等）做 Embedding 相似度匹配，>0.98 視為命中 | 通用方法 ✅ |

## G.7　命中率避坑（第 11 章「黃金法則」）

| 常見錯誤實踐（導致 Cache 失效）| 正確策略（最大化命中）|
|---|---|
| System Prompt 塞動態變量（時間戳、用戶 ID）| 動態用戶資訊移到訊息**最末尾**的 User 欄，開頭固定 |
| 每輪重新生成歷史摘要（Summary）| 歷史**只追加不修改**；截斷從**最舊**開始刪，保前綴一致 |
| 工具定義（Tools）順序隨機化 | 嚴格固定 JSON 工具列表內部排序、欄位結構與前後順序 |
| 空格/換行等細節不統一 | 統一文本清洗邏輯，防一個 `\n` 改變 Token 序列 |
| 開頭插入隨機鹽（Salt）/UUID | 用 `cache_control` 顯式錨點；確定性序列化（Deterministic Ordering）|

> **黃金法則**：「**穩定的靜態內容放最前、動態高變內容放最後**」——前綴匹配是一切 Prompt Cache 的根。

---

> ⚠️ **真偽校準 — 第 10–11 章敘事中需謹慎看待的聲稱**
>
> G.1–G.7 表中標 ✅ 者，均為可在 arXiv / GitHub / 官方文件查證的**真實論文或開源工具**（MLA、H2O、StreamingLLM、KV-Quant、vLLM、SGLang、LMDeploy、TensorRT-LLM、LightLLM、DeepSpeed、TGI、FlashAttention、PagedAttention、GPTCache、Milvus、Qdrant、Anthropic `cache_control` 等）。
>
> 下列在書中敘事出現的內容**本書無法核實**，應視為**合理工程推演而非既成事實**：
> - **「DeepMind 內部架構」相關聲稱** ⚠️（MLA「DeepMind 也深度跟進」、PD 分離「Google DeepMind/OpenAI 終極方向」）——方向合理但無公開出處。
> - **特定 Gemini 上下文視窗大小 / 具體規格** ⚠️。
> - **「Google Jupiter 1.6 Tbps RDMA」** ⚠️（具體頻寬數字未經查證）。
> - **精確節省百分比**（「KV Cache 砍 90%+」「輸入費用暴跌 80%」「相似度 >0.98」「2-bit 幾乎不掉點」）⚠️——量級方向可參考，**確切數字依模型/負載而異**，不應當作保證值引用。
> - **PCIe Gen5 / HBM3e 等硬體型號**為情境舉例，實際部署以官方規格為準。
