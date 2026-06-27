# 附錄B　名詞與工具速查

> 把全書出現的技術名詞，按主題收攏成一張「解碼器」。這些**全是真實、公開的技術**，與附錄 A 那些「待核實的模型名」不同——掌握它們，你就能聽懂任何蒸餾／本地部署的宣傳到底在說什麼。

## B.1　蒸餾核心概念

| 名詞 | 一句話解釋 | 出現章節 |
|---|---|---|
| **知識蒸餾（Knowledge Distillation, KD）** | 讓大教師模型把能力傳給小學生模型 | 全書 |
| **教師 / 學生（Teacher / Student）** | 被學的大模型 / 來學的小模型 | 第 4、5 章 |
| **硬標籤（Hard Target）** | 只學最終答案 | 第 4 章 |
| **軟標籤 / Logits（Soft Target）** | 學教師的完整機率分佈（「暗知識」）| 第 4 章 |
| **KL 散度（KL-Divergence）** | 衡量兩個機率分佈差多遠的數學工具 | 第 4、5 章 |
| **思維鏈（Chain-of-Thought, CoT）** | 模型一步步推理的過程 | 第 2、4、5 章 |
| **CoT 對齊損失** | 給思考過程加權，逼學生「想清楚再答」 | 第 4、5 章 |
| **軌跡反轉（Trace Inversion）** | 打亂/倒序推理步驟，防學生死背 | 第 2、4 章 |
| **語義改寫（Synthetic Paraphrasing）** | 高溫採樣改寫數據，擴泛化、防過擬合 | 第 4、5 章 |
| **過度擬合（Overfitting）** | 死背訓練數據、換場景就崩 | 第 4 章 |
| **abliteration** | 消除權重中「拒絕」方向，做去審查 | 第 2 章 |

## B.2　訓練與微調

| 名詞 | 解釋 |
|---|---|
| **SFT（指令微調）** | 用「指令→回應」對監督微調 |
| **QLoRA** | 量化 + 低秩適配的高效微調法（省顯存）|
| **全參數微調（Full FT）** | 更新模型所有權重（最徹底、最耗資源）|
| **DeepSpeed ZeRO-3** | 把權重/梯度/優化器狀態分片，省記憶體跑大模型 |
| **混合精度（BF16/FP16）** | 用較低精度加速訓練、兼顧穩定 |
| **梯度裁剪 / 餘弦退火 / Early Stopping** | 三種防訓練崩壞、穩定收斂的護欄 |
| **YaRN + RoPE Base 擴展** | 把模型上下文視窗從 8K 漸進拉到 1M 的技術 |
| **FlashAttention-3** | 高效注意力計算，省記憶體、加速長序列 |

## B.3　量化與部署

| 名詞 | 解釋 |
|---|---|
| **量化（Quantization）** | 把權重從高精度壓到低精度，換取體積/速度 |
| **GGUF** | 本地部署主流模型格式（llama.cpp/Ollama/LM Studio）|
| **Q4_K_M / Q5_K_M / Q8_0** | 常見量化等級；數字大=精確/大，K_M=混合精度 |
| **混合量化** | 注意力層保高精度、MLP 層壓低精度 |
| **KV Cache** | 上下文記憶體；長文本的真正瓶頸 |
| **KV Cache 量化（FP8/INT8）** | 壓縮上下文記憶體，釋放長文本潛能 |
| **MoE（混合專家）** | 多專家、每次只激活少數；頻寬受限機器的救星 |
| **激活參數 vs 總參數** | 前者決定速度，後者決定智商上限 |

## B.4　硬體（NVIDIA DGX Spark）

| 名詞 | 解釋 |
|---|---|
| **DGX Spark** | NVIDIA 的桌上型「個人 AI 超級電腦」|
| **GB10（Grace Blackwell）** | DGX Spark 搭載的超級晶片 |
| **UMA（統一記憶體架構）** | CPU 與 iGPU 共享同一池記憶體 |
| **128GB LPDDR5X（~600 GB/s）** | 容量巨大、但頻寬不及獨顯 HBM/GDDR |
| **容量 vs 頻寬鐵律** | 容量決定「能不能跑」，頻寬決定「跑多快」|
| **numactl --interleave=all / mlock** | 鎖記憶體、交錯通道，榨乾 UMA 頻寬 |

## B.5　訓練、微調與蒸餾工具

| 工具 | 角色 | 章節 |
|---|---|---|
| **Hugging Face `transformers` / `datasets`** | 模型與資料的基礎庫 | 第 2、4、5 章 |
| **`peft`** | LoRA / QLoRA 等參數高效微調 | 第 2、5 章 |
| **`trl`（SFTTrainer / GKDTrainer / DPO）** | 監督微調、在線蒸餾、偏好對齊 | 第 2、4 章 |
| **Axolotl / LLaMA-Factory / Unsloth** | 開箱即用的微調框架（Unsloth 省顯存）| 第 1、2 章 |
| **DeepSpeed（ZeRO-3）/ FSDP / `accelerate`** | 分散式訓練、權重分片 | 第 4、5 章 |
| **`bitsandbytes`（NF4）** | QLoRA 量化基座、paged optimizer | 第 2、4 章 |
| **FlashAttention-3** | 高效注意力，長序列省記憶體 | 第 3、5 章 |
| **distilabel（Argilla）** | 合成數據 / 語義改寫 / CoT 生成 | 第 4、5 章 |
| **mergekit** | LoRA/權重融合與模型合併 | 第 5 章 |
| **MiniLLM / GKD（實作）** | on-policy / reverse-KL 蒸餾 | 第 4 章 |
| **TransformerLens / heretic / refusal_direction** | abliteration（去審查）激活分析 | 第 2 章 |
| **Microsoft Presidio / spaCy NER** | 資料去隱私（PII 去識別）| 第 5 章 |
| **Redis 叢集 + Apache Arrow** | 高吞吐零複製資料快取層 | 第 5 章 |
| **Prometheus + Grafana** | 硬體/訓練遙測監控 | 第 5 章 |

## B.6　量化工具

| 工具 | 角色 |
|---|---|
| **llama.cpp（GGUF / K-quants / `quantize`）** | 主流本地量化與推理 |
| **imatrix（importance matrix）** | 量化前校準，同體積更高品質 |
| **AutoGPTQ / AutoAWQ** | GPTQ / AWQ 訓練後量化 |
| **ExLlamaV2** | 高速量化推理引擎 |
| **FP8 / INT8 KV-Cache 量化** | 壓縮上下文記憶體（vLLM、llama.cpp）|

## B.7　本地推理與部署

| 工具 | 角色 |
|---|---|
| **Ollama** | 本地模型管理/推理後端，支援多模型動態載入 |
| **LM Studio** | 圖形化本地推理前端 |
| **llama.cpp / vLLM / SGLang** | 推理引擎（vLLM/SGLang 高吞吐、支援投機解碼）|
| **NVIDIA NIM / DGX OS stack** | DGX Spark 內建推理堆疊 |
| **Open WebUI / Cursor / Continue** | 前端介面，支援模型切換與「雙模對決 Arena」|
| **OpenHands（OpenDevin）/ Aider** | 開源 Coding Agent，可對接本地模型 |

## B.8　評估基準

| 基準 | 測什麼 |
|---|---|
| **MMLU-Pro** | 多領域知識與推理 |
| **HumanEval** | 程式碼生成能力 |
| **GSM8K** | 數學應用題推理 |
| **lm-evaluation-harness（EleutherAI）** | 統一跑各基準的工具 |
| **RULER / Needle in a Haystack** | 長上下文檢索能力（驗證「讀得懂多長」）|
