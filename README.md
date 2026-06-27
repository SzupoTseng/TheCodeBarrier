# 代碼壁壘 · The Code Barrier

<p align="center">
  <img src="TheCodeBarrier.png" alt="代碼壁壘 · The Code Barrier — 封面海報 / cover poster" width="620">
</p>

**🌐 Language / 語言：[繁體中文](#繁體中文) · [English](#english) · [简体中文](#简体中文) · [日本語](#日本語)**

---

<a id="繁體中文"></a>
## 繁體中文

> **蒸餾、代理操作與快取——造出、用好、養得起一個真正屬於你的本地智慧。**

二〇二六年那個夏天，**Claude Fable 5** 橫掃基準、卻又因一紙出口管制在數天內全球停用；社群只搶到一個極窄的窗口。一時間，掛著「Claude distilled」名號的開源模型洗版而出。這本書，就是那場調查的紀錄——它從「怎麼自己蒸一個」一路追問到「蒸出來之後怎麼用它、怎麼養它」，於是長成了三個樂章。

> 💡 **全書的核心命題**
> 技術自由的真諦，不是「免費取得最強的能力」，而是「**完全掌控一個夠用的能力**」——能掌控的平庸，勝過不能擁有的卓越。

### 三個樂章

| 樂章 | 動詞 | 內容 |
|------|------|------|
| **第一樂章（第 1–6 章）** | **造它** | 反蒸餾的盾、開源影子清單、DGX Spark 選型、蒸餾的科學、企業級流水線、法律與現實校準 |
| **第二樂章（第 7–9 章）** | **用它** | 推理—行動迴圈、二十道難關與類人操控、為 AI 重新設計電腦（從像素到語意總線） |
| **第三樂章（第 10–11 章）** | **養它** | 快取的物理學（瓶頸是記憶體不是算力）、超大規模快取與前沿軍械庫 |

造它、用它、養它——這三個動詞是貫穿全書十一章的那根脊椎：你蒸出的廉價小模型，正是第二樂章那雙「眼和手」；而第三樂章的記憶體戰爭，又回頭決定它能不能撐住萬級並發。

### 結構

- **正文 11 章**，每章採故事化情境拆解，附「💡 君之一席話」與「🔍 進階點評」。
- **附錄 A–R（18 份）**：A 模型/數據集 · B 名詞工具 · C 爭議問答 · D 評測題庫 · E 去審查與安全評估 · F 代理操作工具與論文 · G 快取技術速查 · **H–J 快取三百式（300 式逐項深拆）** · **K–M 蒸餾必考三百題（300 題逐題深拆）** · N 大模型 16 大研究方向 · O 智能層次論（AGI/ASI/UAI） · P 開源蒸餾與對齊工具鏈 · Q 蒸餾流水線實作指引（合規工程四十步） · **R 核心實作源碼參考**。

### 真偽紀律與紅線

書中橫跨真實技術與未證實傳聞：凡未經查證的產品名、規格、分數、延遲，一律標「⚠️ 真偽提醒／校準」，請信其方向、疑其細節。涉及未授權能力提取、繞過存取控制、破除安全對齊的內容，**均為風險揭露，不構成操作建議**；越獄題庫、去審查配方、繞過防護腳本一律不收錄（詳第 6 章與附錄 E）。

### 怎麼讀 / 怎麼建置

照三樂章順序讀，或直接翻到你最關心的那一章。建置為純 Python、無第三方依賴：

```bash
python _build.py        # 合併單檔 .md + 帶側欄目錄的 .html（PDF 由 HTML 列印）
python _convert_cn.py   # 繁體 → 简体（需 opencc），輸出到 cn/
```

### 語言版本

| 語言 | 狀態 | 位置 |
|------|------|------|
| 繁體中文 | ✅ 完成（11 章 + 附錄 A–R） | 倉庫根目錄 |
| 简体中文 | ✅ 完成（opencc tw2sp 轉換） | [`cn/`](cn/) |
| English | 🚧 核心完成（第 1–11 章 + 附錄 A–G）；附錄 H–R 翻譯中 | [`en/`](en/) |
| 日本語 | 🚧 規劃中 | [`ja/`](ja/) |

---

<a id="english"></a>
## English

> **Distillation, computer-use agents, and caching — build, wield, and sustain a local intelligence that is truly yours.**

In the summer of 2026, **Claude Fable 5** swept the benchmarks — then vanished worldwide within days under an export-control order, leaving the community a razor-thin window. Overnight, open models wearing the "Claude distilled" badge flooded the feeds. This book is the record of that investigation: it starts from *how do I distill one myself* and follows the question all the way to *once you have it, how do you use it, and how do you keep it running* — and so it grew into three movements.

> 💡 **The book's central thesis**
> Technical freedom isn't "getting the strongest capability for free" — it's "**fully controlling a capability that's good enough**." A mediocrity you own beats an excellence you can't.

### Three movements

| Movement | Verb | Contents |
|----------|------|----------|
| **One (Ch. 1–6)** | **Build it** | The anti-distillation shield · the open-source "shadows" · DGX Spark selection · the science of distillation · an enterprise pipeline · law & reality check |
| **Two (Ch. 7–9)** | **Use it** | The reason–act loop · twenty hurdles & human-like control · redesigning the computer for AI (pixels → semantic state bus) |
| **Three (Ch. 10–11)** | **Sustain it** | The physics of cache (the bottleneck is memory, not compute) · cache at hyperscale & the frontier armory |

### Structure

- **11 chapters**, each a story-driven scenario breakdown, with "💡 A Word to the Wise" and "🔍 Deeper Commentary" boxes.
- **Appendices A–R (18)**, including the two deep catalogs: **H–J "300 Cache Techniques"** and **K–M "300 Distillation Exam Questions"**, plus N (16 research directions), O (AGI/ASI/UAI), P (toolchain), Q (compliant 40-step pipeline), and **R (core reference implementations / source code)**.

### Authenticity discipline & red lines

Real engineering and unverified rumor sit side by side; every unverified product name, spec, score, or latency is flagged "⚠️ Authenticity Caveat" — trust the direction, doubt the detail. Anything touching unauthorized extraction, bypassing access controls, or removing safety alignment is **risk disclosure only, not operational advice**; jailbreak banks, de-censorship recipes, and vendor-defense bypass scripts are deliberately omitted (see Ch. 6 and Appendix E).

### Build

Pure Python, no third-party deps: `python _build.py` (merged `.md` + sidebar-TOC `.html`; PDF printed from HTML).

| Language | Status | Path |
|----------|--------|------|
| 繁體中文 | ✅ Complete (11 ch + App. A–R) | repo root |
| 简体中文 | ✅ Complete (opencc tw2sp) | [`cn/`](cn/) |
| English | 🚧 Core complete (Ch. 1–11 + App. A–G); App. H–R in translation | [`en/`](en/) |
| 日本語 | 🚧 Planned | [`ja/`](ja/) |

---

<a id="简体中文"></a>
## 简体中文

> **蒸馏、代理操作与缓存——造出、用好、养得起一个真正属于你的本地智能。**

二〇二六年那个夏天，**Claude Fable 5** 横扫基准、却又因一纸出口管制在数天内全球停用，社群只抢到一个极窄的窗口。一时间，挂着「Claude distilled」名号的开源模型刷屏而出。这本书，就是那场调查的纪录——从「怎么自己蒸一个」一路追问到「蒸出来之后怎么用它、怎么养它」，于是长成了三个乐章。

> 💡 **全书的核心命题**
> 技术自由的真谛，不是「免费取得最强的能力」，而是「**完全掌控一个够用的能力**」。

### 三个乐章

| 乐章 | 动词 | 内容 |
|------|------|------|
| **第一乐章（第 1–6 章）** | **造它** | 反蒸馏的盾、开源影子清单、DGX Spark 选型、蒸馏的科学、企业级流水线、法律与现实校准 |
| **第二乐章（第 7–9 章）** | **用它** | 推理—行动回路、二十道难关与类人操控、为 AI 重新设计电脑 |
| **第三乐章（第 10–11 章）** | **养它** | 缓存的物理学（瓶颈是内存不是算力）、超大规模缓存与前沿军械库 |

完整 11 章 + 附录 A–R（含「缓存三百式」与「蒸馏必考三百题」两份逐项深拆目录、核心源码参考）。在线阅读见 [`cn/`](cn/)。

> 真伪纪律与红线：未经查证的产品名/规格/分数一律标「⚠️ 真伪提醒」；越狱题库、去审查配方、绕过防护脚本一律不收录（详第 6 章与附录 E）。

---

<a id="日本語"></a>
## 日本語

> **蒸留・コンピュータ操作エージェント・キャッシュ——自分だけのローカル知能を「作り、使い、養う」。**

二〇二六年の夏、**Claude Fable 5** はベンチマークを席巻した直後、輸出規制により数日で世界的に停止された。コミュニティが確保できたのはごく狭い窓だけだった。本書はその調査の記録であり、「どう自分で蒸留するか」から「蒸留した後、どう使い、どう運用し続けるか」までを追い、三つの楽章へと育った。

> 💡 **本書の中心命題**
> 技術的自由とは「最強の能力を無料で得ること」ではなく、「**十分な能力を完全に掌握すること**」である。

- **第一楽章（第 1–6 章）作る**：蒸留対策の盾、オープンソースの「影」、DGX Spark 選定、蒸留の科学、企業級パイプライン、法と現実。
- **第二楽章（第 7–9 章）使う**：推論—行動ループ、二十の難関と人間的操作、AI のためのコンピュータ再設計。
- **第三楽章（第 10–11 章）養う**：キャッシュの物理学、超大規模キャッシュと最前線の武器庫。

> 🚧 日本語版は準備中です。本文は繁体中文版・英語版を参照してください。
> The Japanese edition is in progress; please refer to the Traditional Chinese or English editions for now.

---

> 📜 本書僅供**教育與研究**用途，詳見 [LICENSE](LICENSE)。書中框架名稱與未證實宣稱為敘事示意。
> For **educational & research** use only — see [LICENSE](LICENSE).
