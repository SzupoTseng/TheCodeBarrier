# 附錄P　開源蒸餾與對齊工具鏈速查

> 把第 4、5 章談的「軟標籤對齊、偏好優化、分頁 KV、模型合併」等理論，對應到**真實、公開、可在 GitHub / Hugging Face 查證**的開源工具。本附錄的蒸餾、推理、合併、評測四節（P.1–P.3）為**完全合規**的工程工具，給足技術深度；P.4 的「去審查（abliteration）」工具屬**雙用途（dual-use）**，本書只做**防禦性風險揭露**，不提供任何操作步驟、CLI、設定檔——紅線一律回指**附錄E**與**第 6 章**。

---

## P.1　知識蒸餾與微調工具

把教師（大模型）的能力壓進學生（小模型），實務上分三條路線：**SFT（學硬標籤）→ 在線蒸餾（學 logits／on-policy）→ 偏好對齊（DPO/ORPO/GRPO）**。下表的工具多可互相對接（同一份 `transformers` 權重 + `datasets` 語料）。

| 工具 | 核心技術 | 對接模組 | 硬體開銷 | 優點 | 風險 / 注意 |
|---|---|---|---|---|---|
| **LLaMA-Factory** | 一體化微調：SFT / DPO / ORPO / **GRPO**、LoRA/QLoRA/Full FT、Sequence Packing | `transformers`+`peft`+`trl`+`deepspeed`，WebUI/YAML | 中→極高（Full FT 需 ZeRO-3 分片） | 配置化、演算法覆蓋最全、上手快 | 抽象層厚，出錯時難下沉除錯；學習率失控易 NaN |
| **axolotl** | YAML 驅動微調框架，支援多資料格式、packing、RoPE 擴展 | `transformers`+`accelerate`+`deepspeed` | 中→高 | 社群配方多、可複現性好 | 版本與依賴漂移快，需鎖版本 |
| **Unsloth** | 手寫 Triton kernel + 融合算子，QLoRA 省顯存提速 | `transformers`+`peft`+`bitsandbytes` | 低（單卡友善） | 顯存砍半、速度 2× 級，消費級顯卡可跑 | 支援的模型架構有限；多卡擴展弱於 DeepSpeed |
| **HF `trl`** | `SFTTrainer` / `DPOTrainer` / **`GKDTrainer`**（廣義知識蒸餾、on-policy reverse-KL） | `transformers`+`accelerate`+`peft` | 中 | 官方維護、與生態無縫、`GKDTrainer` 是「真蒸餾」入口 | 偏底層，需自己組 pipeline |
| **DeepSpeed（ZeRO-3）/ FSDP** | 把權重／梯度／優化器狀態**分片**到多卡或 CPU/NVMe | 上述框架的後端 | 解鎖大模型 Full FT | 用有限顯存跑超大模型；ZeRO-Offload 救小顯存 | 通訊開銷大，頻寬不足時被網路綁死；配置敏感 |
| **distilabel（+ Argilla）** | 合成資料 / CoT 生成 / 語義改寫 / 偏好對標註 pipeline | LLM API 或本地推理引擎 | 低（資料端） | 把「造蒸餾語料」標準化、可審計、可去重 | 合成資料品質＝下游天花板，需人工抽查＋去汙染 |

**點評**

- **LLaMA-Factory**：開源微調生態目前事實上的「全家桶」。它的價值在於把 DPO/ORPO/GRPO 這些公式凶險的對齊演算法封成 YAML，讓你專注在**資料偏好本身**而非梯度數值穩定性。代價是抽象層厚——一旦 loss 飛了，往往得繞過框架去看 `trl` 底層。**第一次做蒸餾選它，要深入除錯時下沉到 `trl`。**
- **`trl` 的 `GKDTrainer`**：全書談「學生學教師完整機率分佈（軟標籤／暗知識）」，落到程式碼就是它。相較純 SFT 只學硬標籤，GKD 是 **on-policy + reverse-KL**，讓學生在自己生成的軌跡上對齊教師，能顯著緩解過度擬合（死背）。**想做「真蒸餾」而非「拿教師輸出當 SFT 語料」，這是正門。**
- **Unsloth**：消費級顯卡的救星，靠融合 kernel 把 QLoRA 顯存砍半。但它**不是分散式方案**——要 Full FT 70B 級還是得回到 DeepSpeed/FSDP。定位是「單卡/雙卡把 LoRA 微調做到極致」。
- **distilabel**：蒸餾成敗的隱形勝負手在資料。它把合成語料生成做成可重現 pipeline，配 Argilla 做人工審查。**記住：合成資料的品質就是學生的智商上限，且務必對訓練集做基準洩漏（benchmark contamination）檢查。**

---

## P.2　推理與部署工具

蒸餾／微調完成後，要把模型「跑起來」並榨乾吞吐。瓶頸通常不是算力而是**記憶體頻寬與 KV-Cache 碎片**（見附錄 B.4「容量 vs 頻寬鐵律」）。

| 工具 | 核心技術 | 適用場景 | 硬體開銷 | 優點 | 風險 / 注意 |
|---|---|---|---|---|---|
| **vLLM** | **PagedAttention**（KV-Cache 分頁、零碎片）+ continuous batching + **Speculative Decoding** | 高併發 GPU 服務 | 高（須常駐權重；投機解碼還要載 draft 模型） | 吞吐業界標竿、零 padding、OpenAI 相容 API | 綁 CUDA kernel，跨硬體可攜性差；長 context 仍吃顯存 |
| **TGI（Text-Generation-Inference）** | 連續批次 + 張量並行 + 量化推理，生產級服務化 | 雲端/企業端點部署 | 高 | 開箱即用的 HTTP 服務、可觀測性好、HF 官方 | 功能演進與授權條款變動需留意；客製 kernel 不如 vLLM 多 |
| **llama.cpp** | **GGUF** 格式 + K-quants + CPU/Metal/CUDA 全平台、imatrix 校準 | 本地端／邊緣／Apple Silicon | 低（量化後可純 CPU） | 跨平台、易部署、量化生態最成熟 | 單請求吞吐不如 vLLM；高併發非其強項 |
| **SGLang** | RadixAttention（前綴 KV 重用）+ 結構化輸出快速解碼 | 多輪/共享前綴/Agent 工作流 | 高 | 前綴重用對 few-shot/Agent 提速明顯 | 較新、生態與文件仍在追趕 vLLM |
| **TensorRT-LLM** | NVIDIA 官方，AOT 編譯 engine + in-flight batching + FP8/INT4 | 榨乾單一 NVIDIA GPU 極致延遲/吞吐 | 高 | NVIDIA 硬體上常為效能上限 | 須預編譯 engine、綁死 NVIDIA、可攜性最差、上手陡 |
| **Ollama** | llama.cpp 上層封裝 + Modelfile + 一鍵拉模型 | 本地快速試玩/桌面整合 | 低 | 安裝即用、API 友善、生態廣 | 底層即 llama.cpp，高併發/精調控不如 vLLM 與 llama.cpp 原生 |

**點評**

- **vLLM**：本附錄「部署」一節的核心。PagedAttention 借鑑作業系統的虛擬分頁，把 KV-Cache 切成非連續實體 block，消滅傳統預分配的記憶體空洞（可省下大量浪費），這是它高併發的根本。**Speculative Decoding**（小 draft 模型猜、大 target 模型一次驗證多 token）能進一步壓低延遲。代價是綁死 CUDA kernel，跨平台遷移痛苦。
- **llama.cpp**：與 vLLM 互補。vLLM 主打**伺服器高併發**，llama.cpp 主打**本地單機可攜性**——GGUF + K-quants 讓你在筆電甚至純 CPU 上跑量化模型，配 `imatrix`（重要性矩陣校準）能在同等體積下換更高品質。**個人本地端首選；要扛流量上 vLLM。**
- **SGLang**：當工作流是「大量共享前綴」（同一份 system prompt 反覆呼叫、Agent 多輪）時，RadixAttention 的前綴 KV 重用優勢最大。屬於針對特定負載型態的優化。
- **TensorRT-LLM vs Ollama（兩個極端）**：同樣不在「通用首選」之列，但代表光譜兩端。TensorRT-LLM 用 **AOT 編譯 engine** 在 NVIDIA 卡上換取最後一截延遲/吞吐——代價是預編譯成本與**徹底綁死 NVIDIA**，是「已知硬體、要榨乾最後 10%」時才動用的重武器。Ollama 則是另一端：它本質是 **llama.cpp 的上層封裝**，賣的是「`ollama run` 即用」的桌面體驗，適合本地試玩與整合，但別把它的便利誤當成新的推理引擎——**要調 batch/量化/併發，仍得回到 llama.cpp 或 vLLM 本體。**

> 注意：源文件宣稱「本地工作站轟出 100+／102 tok/s」屬具體效能宣傳，**與硬體、模型大小、量化等級、batch 強相關，無法一概而論**，請以自己機器實測為準。

---

## P.3　模型合併與評測工具

不訓練、只**算術合併**多個權重，已是低成本「縫合」能力的主流手法；而任何蒸餾/合併的成果都必須過**統一評測**才算數。

| 工具 / 方法 | 核心技術 | 用途 | 注意 |
|---|---|---|---|
| **mergekit** | 多種合併演算法的統一框架 | 把多個微調模型融成一個 | 合併≠保證更強，需評測驗證 |
| └ **SLERP** | 球面線性插值，兩模型權重平滑插值 | 兩模型風格/能力折中 | 一次僅兩模型 |
| └ **TIES-Merging** | 修剪微小差異 + 解決符號衝突再合併 | 多模型融合、減少互相干擾 | 需調稀疏比例 |
| └ **DARE** | 隨機丟棄並重縮放任務向量後再合併 | 降低參數冗餘、與 TIES 搭配 | 隨機性需固定種子 |
| **lm-evaluation-harness（EleutherAI）** | 統一跑 MMLU-Pro / GSM8K / HumanEval 等基準 | 橫向評測、防自我感覺良好 | 須查資料洩漏；prompt 模板影響分數 |
| **AlpacaEval 2 / MT-Bench / Arena-Hard** | LLM-as-judge 評指令遵循與對話品質 | 補靜態基準測不到的「好不好用」 | judge 有偏好偏差（長度/風格）、需固定 judge 與基準對手 |
| **RULER / Needle-in-a-Haystack** | 長上下文檢索/推理壓力測試 | 驗證「宣稱的 context 真讀得懂」 | 配合 YaRN/RoPE 擴展一起驗 |

**點評**

- **mergekit**：用幾乎零算力把多個專長模型「縫」成一個——例如把一個強推理、一個強中文的微調體合併。**SLERP** 適合兩模型平滑折中；**TIES/DARE** 處理多模型時的參數符號衝突與冗餘。但要牢記：**合併是經驗性操作，常常合出更差的模型，必須用 P.3 的評測工具回歸驗證，不能憑感覺發版。**
- **lm-evaluation-harness**：學生模型蒸餾完到底有沒有掉智商，靠它一把尺量。**最大陷阱是基準洩漏（test set 混進訓練資料）導致虛高**——務必做去汙染檢查，並注意 prompt 模板差異會明顯影響分數。
- **AlpacaEval / MT-Bench / Arena-Hard**：靜態基準（MMLU/GSM8K）量「會不會」，judge 類評測量「好不好用」——對齊/蒸餾後的對話模型往往基準沒掉、體感卻變差，靠 LLM-as-judge 才抓得到。但**陷阱比靜態基準更隱蔽**：judge 有系統性偏好（偏好更長、更客套、與自己同源的輸出），且分數會隨 judge 模型版本漂移。**用它做相對比較、固定 judge 與基準對手，別把絕對分數當真理。**
- **RULER**：搭配附錄 B.2 的 YaRN+RoPE 擴展使用。模型「宣稱」支援 1M context 不等於「讀得懂」，RULER 與 Needle 測的就是有效長度而非標稱長度。

---

## P.4　對齊與「去審查」工具（防禦性說明）

> ⚠️ **本節為風險揭露，非操作指南。** 以下工具**本身公開、合法、且有正當研究用途**（機械可解釋性／安全研究），但同一批技術被用於**移除模型的安全對齊（去審查 / abliteration）時即構成濫用**。本書**不提供、不轉錄**任何 abliteration 的步驟、CLI、Python 膠水腳本或 YAML 設定——對應的紅線判準一律見**附錄E** 與**第 6 章六條鐵律**。

| 工具 | 它是什麼 | 正當用途 | 雙用途風險 ⚠️ |
|---|---|---|---|
| **TransformerLens** | 機械可解釋性（Mechanistic Interpretability）函式庫，提供 HookPoint 讀取／介入 Transformer 內部激活 | 學術上分析注意力頭、神經元功能、做安全與對齊研究 | 同樣的激活介入能力可被用來定位並削弱安全相關電路 ⚠️ |
| **llm-abliteration 類腳本** | 將「拒絕方向（refusal direction）」從殘差流移除的去審查工具 | （研究語境）量化安全對齊的脆弱性、評估防禦 | **設計目的即移除安全對齊**，產出無拒答的權重，屬高風險濫用 ⚠️⚠️ |

**這些工具在做什麼（事實層面）**

學界 2024 年的 *refusal-direction* 研究（**Arditi et al., 2024**，「Refusal in LLMs is mediated by a single direction」）發現：許多開源模型的「拒絕回答」行為，在高維激活空間中由**單一方向**主導。這是一項**真實、可查證**的可解釋性發現。其防禦面價值在於：它揭露了當前安全對齊的脆弱性，提醒部署方**安全不能只靠模型自身權重**。

**為什麼有風險（防禦意識）**

同一個發現一旦反過來用——把那個方向從權重中抹除——就能在幾乎不重訓的情況下**移除模型的安全護欄**，產出會配合有害請求的權重。這正是 abliteration 工具的爭議所在：技術門檻低、算力需求小、且**繞過了開發者投入的對齊工作**。本書立場明確：

- **不轉錄**源文件中的「三步驟戰役對接 / 一鍵消融」配方、`pip install`、對照提示詞、層數範圍、`save_pretrained` 流程或 DPO 去審查 YAML——這些屬操作性內容，**一律省略**。
- 去除安全對齊的模型可能生成惡意軟體、攻擊代碼、違法內容，**法律與倫理責任由使用者自負**；散布此類權重在多數司法管轄區與平台條款下皆受限。
- 任何「為了能力而拆掉安全」的訴求，**先回到附錄E 的紅線清單與第 6 章六條鐵律**核對是否越界，再決定做不做——通常答案是不做。

> **一句話**：TransformerLens 是把可以救命也可以傷人的手術刀；本書教你**認得它、防得住它被濫用**，不教你**怎麼揮**。操作邊界 → 附錄E ／第 6 章。

---

> ⚠️ **真偽校準**
>
> - P.1–P.3 列出的工具——LLaMA-Factory、axolotl、Unsloth、`trl`（含 `GKDTrainer`）、DeepSpeed/FSDP、distilabel/Argilla、vLLM、TGI、llama.cpp、SGLang、TensorRT-LLM、Ollama、mergekit（SLERP/TIES/DARE）、lm-evaluation-harness、AlpacaEval/MT-Bench/Arena-Hard、RULER——**均為真實、公開、可在 GitHub / Hugging Face / arXiv 查證**的專案，引用前請自行核對版本、授權與合法性。
> - P.4 的 **TransformerLens** 與 **refusal-direction（Arditi 2024）** 為真實已發表項目；**llm-abliteration** 泛指一類公開去審查腳本，**本書不背書、不指引其使用**。
> - 源文件中的**具體效能數字（102 / 100+ tok/s）、模型名（如「Qwable-72B」「120B Fable」「Claude-v5」）、層數與超參數**屬**未經核實**的情境宣傳，**不可當作事實或推薦配置引用**——一切以你自己硬體上的實測為準。
