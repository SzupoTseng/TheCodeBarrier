# 附录F　代理操作（Computer Use）工具与论文速查

> 把第 7–9 章出现的「Computer Use / GUI Agent」生态，按主题收拢成一张对照表。**本附录区分两类**：标准的视觉定位模型、开源代理框架、评测基准与沙箱基础设施，多为**真实、公开的论文或项目**（可自行查证）；而第 7–9 章叙事中出现的若干「Google I/O 2026 产品名」属**未经核实**，已于文末 ⚠️ 区明确标记，请勿当作既成事实引用。

## F.1　视觉定位 / 屏幕理解模型

| 名称 | 一句话定位 | 出处 / 来源 |
|---|---|---|
| **Set-of-Mark（SoM）** | 截屏前先分割 UI 组件并叠加数字标签，把「座标预测」变「选择题」 | Microsoft Research，2023（GPT-4V 视觉定位）✅ |
| **SAM 2** | 影像/视频通用分割模型，常作 SoM 的前置 UI 组件侦测器 | Meta，2024 ✅ |
| **CogAgent** | 18B 专为 GUI 操作微调的 VLM，高/低解析双信道 | THUDM，CVPR 2024 ✅ |
| **ScreenAI** | 同时看懂 UI 布局图与底层语意树（Accessibility/DOM）的 VLM | Google，2024 ✅ |
| **Ferret-UI** | 行动设备（iOS/Android）UI 理解小模型，组件分类/OCR/定位 | Apple，2024 ✅ |
| **NaViT** | 任意长宽比/分辨率的 ViT 打包训练法（多尺度视觉金字塔概念来源） | Google，2023（概念）✅ |
| **ViT（Vision Transformer）** | 把影像切 patch 当 token 的视觉骨干，上述模型的共同基础 | Dosovitskiy et al., 2020 ✅ |

## F.2　网页 / GUI 代理框架（开源）

| 名称 | 角色 | 出处 / 来源 |
|---|---|---|
| **BrowserUse** | 以 Playwright + LLM 驱动、抽取页面可交互元素再决策的浏览器代理，重试/回溯韧性强 | GitHub `browser-use`（开源）✅ |
| **Skyvern** | 以 VLM 驱动的浏览器自动化，免写死 XPath/选择器 | GitHub `Skyvern-AI`（开源）✅ |
| **WebVoyager** | 端到端多模态 Web Agent（截屏+SoM），同时是评测集 | 论文，2024 ✅ |
| **Mind2Web** | 通用 Web Agent 数据集/基准（亦见 F.3）| OSU NLP，2023 ✅ |
| **Code-as-Action / CodeAct** | 让 LLM 直接输出可运行代码（JS/Python）当动作，免疫视觉歪斜 | 论文 *Executable Code Actions…*，UIUC 2024 ✅ |

## F.3　评测基准（Benchmarks）

| 基准 | 测什么 | 规模 / 来源 |
|---|---|---|
| **OSWorld** | 真实 OS 级端到端操作（Office/终端/GIMP 等） | 369 任务，2024，业界公认最难 ✅ |
| **WebArena** | 自托管网站（电商/论坛/Git/CMS）的功能性网页任务 | 812 任务，CMU 2023 ✅ |
| **VisualWebArena** | WebArena 的视觉强化版，需理解图像内容 | 2024 ✅ |
| **Mind2Web** | 137 个真实网站的通用网页操作 | 2,000+ 任务，2023 ✅ |
| **AndroidWorld** | Android 真机环境动态任务，含参数化奖励 | 116 任务，Google 2024 ✅ |

## F.4　基础设施 / 沙箱

| 名称 | 角色 | 来源 |
|---|---|---|
| **Browserbase** | 托管式无头浏览器集群，处理 Session/Proxy/反爬墙（亦推开源 Stagehand 框架） | 商业（YC 新创）✅ |
| **gVisor** | 应用层用户态内核沙箱，拦截 syscall 隔离不可信代码 | Google，开源 ✅ |
| **Firecracker MicroVM** | <125ms 启动的轻量 KVM，每 Agent 一个隔离环境 | AWS，开源 ✅ |
| **Playwright** | 跨浏览器自动化（DOM 操作 / JS 注入 / 截屏） | Microsoft，开源 ✅ |

## F.5　协定 / 安全 / 架构概念

| 名词 | 一句话解释 | 来源 |
|---|---|---|
| **MCP（Model Context Protocol）** | 让模型以标准协定调用外部工具/数据的开放接口 | Anthropic，2024 ✅ |
| **Constitutional AI** | 用一套「宪法」原则做自我批判/对齐；引申为 OS 动作拦截护栏 | Anthropic ✅ |
| **State Space Models / Mamba** | 线性复杂度的串行模型，长串行记忆的替代架构 | Gu & Dao，2023 ✅ |
| **ReAct** | 推理（Reason）与行动（Act）交错的代理范式 | Yao et al., 2022 ✅ |
| **Chain-of-Thought（CoT）** | 让模型一步步推理再答（规划层内核） | Wei et al., 2022 ✅ |
| **[0,1000] 座标范式** | 把点击座标归一到 0–1000 整数区间，分辨率无关 | CogAgent/Qwen-VL 等通用做法 ✅ |
| **Location Tokens（位置 token）** | 用特殊 token 直接表示座标/边界框，视觉定位常见手法 | Pix2Seq/CogAgent/Ferret ✅ |
| **Accessibility Tree 双流对齐** | 视觉截屏 + 无障碍树交叉注意力，幻觉时靠语意树修正座标 | ScreenAI 路线 ✅ |
| **HITL / MFA Break Protocol** | 遇验证码/MFA 时冻结 VM、转人工接管再无缝解冻 | 工程实践（概念）✅ |

---

> ⚠️ **真伪校准 — 第 7–9 章叙事中的「未核实名称」**
>
> 以下名称出现在书中「Google I/O 2026」相关段落，**本书无法核实其真实存在、规格或时程**，应视为情境推演而非事实：
> - **Gemini 3.5 Flash「内置 Computer Use」** ⚠️、**OSWorld 78.4 分** ⚠️（具体分数未经查证）
> - **Aluminium OS**（AI 原生笔电系统）⚠️、**Antigravity**（云端 Agent Harness）⚠️
> - **Project Jarvis**（Chrome 代理服务）⚠️、**AppFunctions / Android MCP** ⚠️
> - **Hydra Memory Layer** ⚠️（内存架构，无公开出处）
>
> 相对地，**Project Astra**（Google DeepMind 通用助理）、**MCP**（Anthropic 开放协定）、**Constitutional AI** 为**真实已公开**项目。F.1–F.5 各表中标 ✅ 者，均为可在 arXiv / GitHub / 官方博客查证的真实论文或开源项目；引用前仍请自行核对版本、授权与合法性（见第 6 章六条铁律）。
