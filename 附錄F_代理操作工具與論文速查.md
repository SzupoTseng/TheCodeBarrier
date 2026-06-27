# 附錄F　代理操作（Computer Use）工具與論文速查

> 把第 7–9 章出現的「Computer Use / GUI Agent」生態，按主題收攏成一張對照表。**本附錄區分兩類**：標準的視覺定位模型、開源代理框架、評測基準與沙箱基礎設施，多為**真實、公開的論文或專案**（可自行查證）；而第 7–9 章敘事中出現的若干「Google I/O 2026 產品名」屬**未經核實**，已於文末 ⚠️ 區明確標記，請勿當作既成事實引用。

## F.1　視覺定位 / 螢幕理解模型

| 名稱 | 一句話定位 | 出處 / 來源 |
|---|---|---|
| **Set-of-Mark（SoM）** | 截圖前先分割 UI 元件並疊加數字標籤，把「座標預測」變「選擇題」 | Microsoft Research，2023（GPT-4V 視覺定位）✅ |
| **SAM 2** | 影像/影片通用分割模型，常作 SoM 的前置 UI 元件偵測器 | Meta，2024 ✅ |
| **CogAgent** | 18B 專為 GUI 操作微調的 VLM，高/低解析雙通道 | THUDM，CVPR 2024 ✅ |
| **ScreenAI** | 同時看懂 UI 佈局圖與底層語意樹（Accessibility/DOM）的 VLM | Google，2024 ✅ |
| **Ferret-UI** | 行動裝置（iOS/Android）UI 理解小模型，元件分類/OCR/定位 | Apple，2024 ✅ |
| **NaViT** | 任意長寬比/解析度的 ViT 打包訓練法（多尺度視覺金字塔概念來源） | Google，2023（概念）✅ |
| **ViT（Vision Transformer）** | 把影像切 patch 當 token 的視覺骨幹，上述模型的共同基礎 | Dosovitskiy et al., 2020 ✅ |

## F.2　網頁 / GUI 代理框架（開源）

| 名稱 | 角色 | 出處 / 來源 |
|---|---|---|
| **BrowserUse** | 以 Playwright + LLM 驅動、抽取頁面可互動元素再決策的瀏覽器代理，重試/回溯韌性強 | GitHub `browser-use`（開源）✅ |
| **Skyvern** | 以 VLM 驅動的瀏覽器自動化，免寫死 XPath/選擇器 | GitHub `Skyvern-AI`（開源）✅ |
| **WebVoyager** | 端到端多模態 Web Agent（截圖+SoM），同時是評測集 | 論文，2024 ✅ |
| **Mind2Web** | 通用 Web Agent 資料集/基準（亦見 F.3）| OSU NLP，2023 ✅ |
| **Code-as-Action / CodeAct** | 讓 LLM 直接輸出可執行程式碼（JS/Python）當動作，免疫視覺歪斜 | 論文 *Executable Code Actions…*，UIUC 2024 ✅ |

## F.3　評測基準（Benchmarks）

| 基準 | 測什麼 | 規模 / 來源 |
|---|---|---|
| **OSWorld** | 真實 OS 級端到端操作（Office/終端/GIMP 等） | 369 任務，2024，業界公認最難 ✅ |
| **WebArena** | 自託管網站（電商/論壇/Git/CMS）的功能性網頁任務 | 812 任務，CMU 2023 ✅ |
| **VisualWebArena** | WebArena 的視覺強化版，需理解圖像內容 | 2024 ✅ |
| **Mind2Web** | 137 個真實網站的通用網頁操作 | 2,000+ 任務，2023 ✅ |
| **AndroidWorld** | Android 真機環境動態任務，含參數化獎勵 | 116 任務，Google 2024 ✅ |

## F.4　基礎設施 / 沙箱

| 名稱 | 角色 | 來源 |
|---|---|---|
| **Browserbase** | 託管式無頭瀏覽器叢集，處理 Session/Proxy/反爬牆（亦推開源 Stagehand 框架） | 商業（YC 新創）✅ |
| **gVisor** | 應用層使用者態核心沙箱，攔截 syscall 隔離不可信代碼 | Google，開源 ✅ |
| **Firecracker MicroVM** | <125ms 啟動的輕量 KVM，每 Agent 一個隔離環境 | AWS，開源 ✅ |
| **Playwright** | 跨瀏覽器自動化（DOM 操作 / JS 注入 / 截圖） | Microsoft，開源 ✅ |

## F.5　協定 / 安全 / 架構概念

| 名詞 | 一句話解釋 | 來源 |
|---|---|---|
| **MCP（Model Context Protocol）** | 讓模型以標準協定呼叫外部工具/資料的開放介面 | Anthropic，2024 ✅ |
| **Constitutional AI** | 用一套「憲法」原則做自我批判/對齊；引申為 OS 動作攔截護欄 | Anthropic ✅ |
| **State Space Models / Mamba** | 線性複雜度的序列模型，長序列記憶的替代架構 | Gu & Dao，2023 ✅ |
| **ReAct** | 推理（Reason）與行動（Act）交錯的代理範式 | Yao et al., 2022 ✅ |
| **Chain-of-Thought（CoT）** | 讓模型一步步推理再答（規劃層核心） | Wei et al., 2022 ✅ |
| **[0,1000] 座標正規化** | 把點擊座標歸一到 0–1000 整數區間，解析度無關 | CogAgent/Qwen-VL 等通用做法 ✅ |
| **Location Tokens（位置 token）** | 用特殊 token 直接表示座標/邊界框，視覺定位常見手法 | Pix2Seq/CogAgent/Ferret ✅ |
| **Accessibility Tree 雙流對齊** | 視覺截圖 + 無障礙樹交叉注意力，幻覺時靠語意樹修正座標 | ScreenAI 路線 ✅ |
| **HITL / MFA Break Protocol** | 遇驗證碼/MFA 時凍結 VM、轉人工接管再無縫解凍 | 工程實踐（概念）✅ |

---

> ⚠️ **真偽校準 — 第 7–9 章敘事中的「未核實名稱」**
>
> 以下名稱出現在書中「Google I/O 2026」相關段落，**本書無法核實其真實存在、規格或時程**，應視為情境推演而非事實：
> - **Gemini 3.5 Flash「內建 Computer Use」** ⚠️、**OSWorld 78.4 分** ⚠️（具體分數未經查證）
> - **Aluminium OS**（AI 原生筆電系統）⚠️、**Antigravity**（雲端 Agent Harness）⚠️
> - **Project Jarvis**（Chrome 代理服務）⚠️、**AppFunctions / Android MCP** ⚠️
> - **Hydra Memory Layer** ⚠️（記憶體架構，無公開出處）
>
> 相對地，**Project Astra**（Google DeepMind 通用助理）、**MCP**（Anthropic 開放協定）、**Constitutional AI** 為**真實已公開**項目。F.1–F.5 各表中標 ✅ 者，均為可在 arXiv / GitHub / 官方部落格查證的真實論文或開源專案；引用前仍請自行核對版本、授權與合法性（見第 6 章六條鐵律）。
