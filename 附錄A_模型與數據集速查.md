# 附錄A　模型、數據集與來源速查

> ⚠️ 全表真偽提醒
> 下列**模型／數據集／團隊／報導名稱，均為坊間流傳、未經證實的傳聞名單**，本書**無法核實其真實存在、規格或授權狀態**。本表僅作為「據傳有這些」的索引，**不構成下載或使用建議**。任何使用前，務必自行查證來源、授權與合法性（見第 6 章六條鐵律）。

## A.1　蒸餾模型（坊間流傳，未經證實）

| 模型名稱 | 基底 | 據稱規格 / 定位 | 據稱出處 |
|---|---|---|---|
| **Qwable-v1** | Qwen3.6-35B-A3B | SFT，繼承 Fable 5 工具調用；~24GB(Q4_K_M)，~102 tok/s | r/huggingface、r/LocalLLaMA、r/opencodeCLI |
| **GPT-OSS 120B Fable-5 Distilled** | GPT-OSS 120B | MoE 128 專家/激活 4；~72GB，~35–50 tok/s；GGUF(Q8_0/Q5_0)；AutoTrust AI Lab 算力 | autotrust/gpt-oss-120b-Fable-5-Distilled-GGUF（HF）|
| **Qwythos-9B-Claude-Mythos-5-1M** | Qwen3.5-9B | 全參數微調；1M 上下文、無審查、4GB 可跑；~6.5GB，~180 tok/s | Empero AI、Interfaze、Threads、零度解说 |
| **richardyoung/qwythos-9b-abliterated** | Qwythos-9B | abliteration 二次去審查變體 | Ollama、HF |
| **Qwythos-27B**（預告）| 未明 | Empero Roadmap 宣稱的下一代「Mythos 級」27B | Empero |
| **Qwen3.6-14B-A3B-FableVibes** | Qwen3.6-35B-A3B「heretic」剪枝 | Pruning + QLoRA MoE；保留「Claude 感」；~11GB，~140 tok/s | tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF（HF）|
| **clzoro/Qwen3.5-27B-Claude-distill** | Qwen3.5-27B（密集）| 餵 Opus CoT，文學/角色扮演強；~19GB，~110 tok/s | HF |
| **Dzluck/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled** | Qwen3.5-2B | 邊緣裝置極致輕量；~2.1GB，~250 tok/s；含 GGUF | HF |
| **Jackrong 系列**（如 27B Coder）| Qwen 系 | 軌跡反轉處理 CoT，社群「最受信任」| LocalLLaMA |

## A.2　數據集（坊間流傳，未經證實）

| 數據集 | 內容 | 用途 | 風險 |
|---|---|---|---|
| **armand0e/claude-fable-5-claude-code** | Fable 5 原始 Agent 執行軌跡（去隱私化，與 Glint-Research 合作），可轉 OpenAI 格式 | 程式/工具調用蒸餾 | ⚠️ 商業模型軌跡，ToS/法律高風險 |
| **ox-ox/mythos-character-distillation** | Claude「說話風格與心理特質」：自發元認知、拒用罐頭回覆、自我修正不找藉口 | 人格/文風蒸餾 | ⚠️ 同上 |

## A.3　Fable 5 / 蒸餾相關報導與討論（標題，未核實）

- 「Anthropic 在 Claude Fable 5 加入蒸餾偵測功能」— 動區動趨（蒸餾偵測 → 自動退回 Opus）
- 「Anthropic Fable 5 近乎橫掃所有 AI 基準，首度將模型蒸餾攻擊列入封鎖範圍」— inside.com.tw
- 「Fable 5 Not Available? What Actually Happened to Claude's Newest Model」（提及美國出口管制指令）
- 「Anthropic Accuses Alibaba of Distilling Claude AI Model Capabilities」— Global Banking & Finance Review
- 「Model Distillation: The Mechanics of Stealing an AI's Intelligence」
- 「Claude 被开源了？Qwythos-9B 突然发布！无审查、104万上下文、4GB 显存可跑」— 零度解说
- 「又有一個強力開源推理模型出來了！Empero 釋出 Qwythos-9B-Claude-Mythos-5」— Threads
- 「Qwythos-9B-Claude-Mythos-5-1M GPU Requirements: VRAM & Cheapest GPU」— Interfaze
- Reddit 討論串：r/LocalLLaMA（mynamasteph：「I would not trust any Claude distill…」）、r/opencodeCLI（TomLucidor：「we kind need SFT/RL…」）、r/huggingface（Qwable-v1 release）

## A.4　DGX Spark 硬體來源（標題，未核實）

- 「採用 Blackwell 架構的 AI 個人超級電腦 | NVIDIA DGX Spark」— NVIDIA（GB10 Grace Blackwell）
- 「採用 GB10 晶片，輝達與多家系統廠商推出新型態 AI 工作站」（iGPU 與 GB100 同源、第五代…）
- 「NVIDIA GB10 Grace Blackwell 超级芯片——桌面 AI 时代」（統一內存架構 UMA）
- 「Solved the DGX Spark, 102 stable tok/s Qwen3.5-35B-A3B」— Reddit（實測參考點）
- 「DGX Spark, what models are you running?」— Reddit
- 「麗臺 NVIDIA DGX Spark 桌上型 AI 超級電腦（GB10/128G/4TB SSD/DGX OS）」、「DGX Spark Founders Edition」

## A.5　提示工程 / 蒸餾方法來源（Q6 附帶，標題，未核實）

- platform.claude.com — Prompting best practices（clarity、examples、XML、thinking、agentic）
- GitHub — Claude Code 各版系統提示與 token 數彙整
- arXiv — 「Training Small Critic Agents」（Opus-4.6 teacher / detailed prompt）
- systemprompt.io — Daily Development Workflows（.md best practices、settings.json）
- claudemarketplaces.com/skills/.../knowledge-distillation — KD 實作指南
- Drew Breunig（dbreunig.com）、LinkedIn「Lessons from Leaked Source」、youmind（NotebookLM）、uxplanet（Prompting Best Practices）

## A.6　真實可用的開源基底與資料集（對應第 2 章）

> 與 A.1–A.2 那些「坊間流傳、未經證實」的雪花名不同，下列為**確實公開、可在 Hugging Face 查證**的基底家族與蒸餾常用資料集。第 2 章的可替換管線實際上就建在這些之上——任何宣稱「Claude distill」的成品，底下幾乎都是這些基底加這些（或自製）資料。

| 真實基底家族 | 出處 | 蒸餾定位 |
|---|---|---|
| **Qwen（Qwen2.5 / Qwen3）** | Alibaba | 中英雙語強、尺寸齊全，社群微調首選 ✅ |
| **Llama 3.x** | Meta | 生態最廣、GQA 推理友善 ✅ |
| **Mistral / Mixtral** | Mistral AI | 密集與 MoE 皆有，歐系開放權重 ✅ |
| **Gemma 2 / 3** | Google | 輕量端側、授權相對寬鬆 ✅ |
| **GPT-OSS（20B / 120B）** | OpenAI | MoE 開放權重，A.1 的 120B 條目即據此 ✅ |
| **Phi-3 / Phi-4** | Microsoft | 小模型、合成資料導向 ✅ |
| **DeepSeek-V2/V3 / R1** | DeepSeek | MLA 省 KV，R1 推理軌跡常作蒸餾教材 ✅ |

| 真實公開資料集 | 內容 | 用途 |
|---|---|---|
| **OpenHermes 2.5** | 百萬級多源指令／對話 | 通用 SFT ✅ |
| **UltraChat / UltraFeedback** | 大規模多輪對話／偏好標註 | SFT + 偏好對齊 ✅ |
| **OpenOrca / SlimOrca** | GPT 生成 CoT 解題軌跡 | 推理蒸餾 ✅ |
| **Magpie** | 由對齊模型自我生成的指令對 | 零人工合成 SFT ✅ |
| **Tülu 3（含 mixtures）** | AllenAI 開源後訓練配方與資料 | 完整 post-training 參考 ✅ |
| **distilabel pipelines** | Argilla 的合成／改寫／評分管線 | 自製蒸餾資料 ✅ |

> 真偽校準一覽
> **較可信（對應真實技術/產品）**：Anthropic、Claude、Fable 5、Qwen、Llama、Mistral、Gemma、GPT-OSS、Phi、DeepSeek、Hugging Face、Ollama、GGUF、QLoRA、abliteration、OpenHermes / UltraChat / OpenOrca / Magpie / Tülu / distilabel、NVIDIA DGX Spark / GB10 / LPDDR5X UMA。
> **無法核實（極可能為傳聞或幻覺）**：Mythos 模型、Qwythos-9B、Qwable-v1、FableVibes、Empero AI 的具體存在與規格、「Fable 5 上線數天遭出口管制下架」「4,659 條軌跡」「蒸餾偵測自動降轉至 Opus 4.8」。A.1 表中的具體 tok/s、GB 數與「繼承 Fable 5 工具調用」均為傳聞聲稱，未經實測。
