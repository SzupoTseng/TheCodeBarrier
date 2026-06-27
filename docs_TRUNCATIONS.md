# 源檔截斷 · 重建記錄 (Source-Truncation Reconstruction Record)

> **狀態：✅ 已重建並跨版本傳播（繁中正典 → 簡中 cn/ → 英文 en/）。日文 ja/ 待建立時一併套用。**

二〇二六年六月，附錄 I／J／M 三份目錄在繁中正典原始素材即有**部分欄位截斷**（以 `（…截斷…）` placeholder 標示）。
依用戶決定「**依標題推論補寫繁中，再傳播**」，已完成重建——所有重建欄位以固定標記昭示，符合全書「真偽紀律：可推演、不假冒」。

## 標記約定
- 繁中／簡中：`〔⚠️ 原素材截斷·依標題推演重建〕<內容>`
- 英文：`[⚠️ Source truncated; reconstructed by inference from the title] <content>`

## 重建統計

| 附錄 | 重建項目 | 重建欄位 |
|---|---|---|
| I — 快取三百式（中）101–200 | 4（110、160、190、200） | 24 |
| J — 快取三百式（下）201–300 | 4（210、230、250、260） | 22 |
| M — 蒸餾必考三百題（下）201–300 | 10（291–300） | 60 |
| **合計** | **18 項** | **106 欄位** |

## ⚠️ 偵測誤判（false positive，未動）
初掃以「含『截斷』字樣」偵測，誤含 3 項——其「截斷」是正常技術詞彙、非 placeholder，已保持原樣不動：

| 項目 | 誤判處 | 真實語意 |
|---|---|---|
| 附錄 I · 116 Token-Level Continuous Cache Compaction | 背景「分支**截斷**致物理地址錯亂」 | branch truncation（分支裁切） |
| 附錄 I · 151 Context-Aware Dynamic Quantization Threshold | 背景「數值**截斷**噪音」 | numeric-truncation noise |
| 附錄 J · 243 Dynamic Synaptic-Gated Low-Rank Latent Cache | 技術難點「最小粗粒度**截斷**」 | coarse-grained cutoff |

## 重建明細（繁中正典欄位）
- **附錄 I**：110（原理尾 + 執行方向/應用情境/技術難點/可能效益/開源可行性/地端可行性/真偽）、160（執行方向起 7 欄）、190（執行方向尾 + 5 欄）、200（技術難點/可能效益/真偽）
- **附錄 J**（命名式構想，真偽欄誠實標高度推測）：210（全 8 欄）、230（背景/真偽）、250（原理起 7 欄）、260（執行方向 + 還原 5 欄骨架）
- **附錄 M**：291–300 各補 背景／優點／缺點／沒問的缺點／點評／以往經驗（每題 6 欄，循 `（與第 X 題同源）` 錨點比對深度）

## 傳播驗證
- 繁中：`附錄I/J/M`，0 殘留 placeholder；25/22/60 重建標記。已重建 `代碼壁壘_全書.md` + `.html`。
- 簡中 `cn/`：opencc tw2sp 重轉，0 殘留 placeholder；25/22/60 標記。已重建 `代码壁垒_全书.md` + `.html`。
- 英文 `en/`：`20/21/24` 重譯，24/22/60 `reconstructed by inference` 標記，0 bare truncation 欄位。已重建 `TheCodeBarrier_full.md` + `.html`。
- ⚠️ 三版 PDF 均需由各自最新 HTML **手動列印**後才含本次重建內容。
