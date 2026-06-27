# 附录A　模型、数据集与来源速查

> ⚠️ 全表真伪提醒
> 下列**模型／数据集／团队／报导名称，均为坊间流传、未经证实的传闻名单**，本书**无法核实其真实存在、规格或授权状态**。本表仅作为「据传有这些」的索引，**不构成下载或使用建议**。任何使用前，务必自行查证来源、授权与合法性（见第 6 章六条铁律）。

## A.1　蒸馏模型（坊间流传，未经证实）

| 模型名称 | 基底 | 据称规格 / 定位 | 据称出处 |
|---|---|---|---|
| **Qwable-v1** | Qwen3.6-35B-A3B | SFT，继承 Fable 5 工具调用；~24GB(Q4_K_M)，~102 tok/s | r/huggingface、r/LocalLLaMA、r/opencodeCLI |
| **GPT-OSS 120B Fable-5 Distilled** | GPT-OSS 120B | MoE 128 专家/激活 4；~72GB，~35–50 tok/s；GGUF(Q8_0/Q5_0)；AutoTrust AI Lab 算力 | autotrust/gpt-oss-120b-Fable-5-Distilled-GGUF（HF）|
| **Qwythos-9B-Claude-Mythos-5-1M** | Qwen3.5-9B | 全参数微调；1M 上下文、无审查、4GB 可跑；~6.5GB，~180 tok/s | Empero AI、Interfaze、Threads、零度解说 |
| **richardyoung/qwythos-9b-abliterated** | Qwythos-9B | abliteration 二次去审查变体 | Ollama、HF |
| **Qwythos-27B**（预告）| 未明 | Empero Roadmap 宣称的下一代「Mythos 级」27B | Empero |
| **Qwen3.6-14B-A3B-FableVibes** | Qwen3.6-35B-A3B「heretic」剪枝 | Pruning + QLoRA MoE；保留「Claude 感」；~11GB，~140 tok/s | tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF（HF）|
| **clzoro/Qwen3.5-27B-Claude-distill** | Qwen3.5-27B（密集）| 喂 Opus CoT，文学/角色扮演强；~19GB，~110 tok/s | HF |
| **Dzluck/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled** | Qwen3.5-2B | 边缘设备极致轻量；~2.1GB，~250 tok/s；含 GGUF | HF |
| **Jackrong 系列**（如 27B Coder）| Qwen 系 | 轨迹反转处理 CoT，社群「最受信任」| LocalLLaMA |

## A.2　数据集（坊间流传，未经证实）

| 数据集 | 内容 | 用途 | 风险 |
|---|---|---|---|
| **armand0e/claude-fable-5-claude-code** | Fable 5 原始 Agent 运行轨迹（去隐私化，与 Glint-Research 合作），可转 OpenAI 格式 | 程序/工具调用蒸馏 | ⚠️ 商业模型轨迹，ToS/法律高风险 |
| **ox-ox/mythos-character-distillation** | Claude「说话风格与心理特质」：自发元认知、拒用罐头回复、自我修正不找借口 | 人格/文风蒸馏 | ⚠️ 同上 |

## A.3　Fable 5 / 蒸馏相关报导与讨论（标题，未核实）

- 「Anthropic 在 Claude Fable 5 加入蒸馏侦测功能」— 动区动趋（蒸馏侦测 → 自动退回 Opus）
- 「Anthropic Fable 5 近乎横扫所有 AI 基准，首度将模型蒸馏攻击列入封锁范围」— inside.com.tw
- 「Fable 5 Not Available? What Actually Happened to Claude's Newest Model」（提及美国出口管制指令）
- 「Anthropic Accuses Alibaba of Distilling Claude AI Model Capabilities」— Global Banking & Finance Review
- 「Model Distillation: The Mechanics of Stealing an AI's Intelligence」
- 「Claude 被开源了？Qwythos-9B 突然发布！无审查、104万上下文、4GB 显存可跑」— 零度解说
- 「又有一个强力开源推理模型出来了！Empero 发布 Qwythos-9B-Claude-Mythos-5」— Threads
- 「Qwythos-9B-Claude-Mythos-5-1M GPU Requirements: VRAM & Cheapest GPU」— Interfaze
- Reddit 讨论串：r/LocalLLaMA（mynamasteph：「I would not trust any Claude distill…」）、r/opencodeCLI（TomLucidor：「we kind need SFT/RL…」）、r/huggingface（Qwable-v1 release）

## A.4　DGX Spark 硬件来源（标题，未核实）

- 「采用 Blackwell 架构的 AI 个人超级电脑 | NVIDIA DGX Spark」— NVIDIA（GB10 Grace Blackwell）
- 「采用 GB10 芯片，辉达与多家系统厂商推出新型态 AI 工作站」（iGPU 与 GB100 同源、第五代…）
- 「NVIDIA GB10 Grace Blackwell 超级芯片——桌面 AI 时代」（统一内存架构 UMA）
- 「Solved the DGX Spark, 102 stable tok/s Qwen3.5-35B-A3B」— Reddit（实测参考点）
- 「DGX Spark, what models are you running?」— Reddit
- 「丽台 NVIDIA DGX Spark 桌面型 AI 超级电脑（GB10/128G/4TB SSD/DGX OS）」、「DGX Spark Founders Edition」

## A.5　提示工程 / 蒸馏方法来源（Q6 附带，标题，未核实）

- platform.claude.com — Prompting best practices（clarity、examples、XML、thinking、agentic）
- GitHub — Claude Code 各版系统提示与 token 数汇整
- arXiv — 「Training Small Critic Agents」（Opus-4.6 teacher / detailed prompt）
- systemprompt.io — Daily Development Workflows（.md best practices、settings.json）
- claudemarketplaces.com/skills/.../knowledge-distillation — KD 实作指南
- Drew Breunig（dbreunig.com）、LinkedIn「Lessons from Leaked Source」、youmind（NotebookLM）、uxplanet（Prompting Best Practices）

## A.6　真实可用的开源基底与数据集（对应第 2 章）

> 与 A.1–A.2 那些「坊间流传、未经证实」的雪花名不同，下列为**确实公开、可在 Hugging Face 查证**的基底家族与蒸馏常用数据集。第 2 章的可替换管线实际上就建在这些之上——任何宣称「Claude distill」的成品，底下几乎都是这些基底加这些（或自制）数据。

| 真实基底家族 | 出处 | 蒸馏定位 |
|---|---|---|
| **Qwen（Qwen2.5 / Qwen3）** | Alibaba | 中英双语强、尺寸齐全，社群微调首选 ✅ |
| **Llama 3.x** | Meta | 生态最广、GQA 推理友善 ✅ |
| **Mistral / Mixtral** | Mistral AI | 密集与 MoE 皆有，欧系开放权重 ✅ |
| **Gemma 2 / 3** | Google | 轻量端侧、授权相对宽松 ✅ |
| **GPT-OSS（20B / 120B）** | OpenAI | MoE 开放权重，A.1 的 120B 条目即据此 ✅ |
| **Phi-3 / Phi-4** | Microsoft | 小模型、合成数据导向 ✅ |
| **DeepSeek-V2/V3 / R1** | DeepSeek | MLA 省 KV，R1 推理轨迹常作蒸馏教材 ✅ |

| 真实公开数据集 | 内容 | 用途 |
|---|---|---|
| **OpenHermes 2.5** | 百万级多源指令／对话 | 通用 SFT ✅ |
| **UltraChat / UltraFeedback** | 大规模多轮对话／偏好标注 | SFT + 偏好对齐 ✅ |
| **OpenOrca / SlimOrca** | GPT 生成 CoT 解题轨迹 | 推理蒸馏 ✅ |
| **Magpie** | 由对齐模型自我生成的指令对 | 零人工合成 SFT ✅ |
| **Tülu 3（含 mixtures）** | AllenAI 开源后训练配方与数据 | 完整 post-training 参考 ✅ |
| **distilabel pipelines** | Argilla 的合成／改写／评分管线 | 自制蒸馏数据 ✅ |

> 真伪校准一览
> **较可信（对应真实技术/产品）**：Anthropic、Claude、Fable 5、Qwen、Llama、Mistral、Gemma、GPT-OSS、Phi、DeepSeek、Hugging Face、Ollama、GGUF、QLoRA、abliteration、OpenHermes / UltraChat / OpenOrca / Magpie / Tülu / distilabel、NVIDIA DGX Spark / GB10 / LPDDR5X UMA。
> **无法核实（极可能为传闻或幻觉）**：Mythos 模型、Qwythos-9B、Qwable-v1、FableVibes、Empero AI 的具体存在与规格、「Fable 5 上线数天遭出口管制下架」「4,659 条轨迹」「蒸馏侦测自动降转至 Opus 4.8」。A.1 表中的具体 tok/s、GB 数与「继承 Fable 5 工具调用」均为传闻声称，未经实测。
