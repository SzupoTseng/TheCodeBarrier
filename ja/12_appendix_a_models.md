# 付録A　モデル・データセット・出典クイックリファレンス

> ⚠️ 全表 真偽メモ
> 以下の**モデル／データセット／チーム／報道名は、いずれも巷で流布する未検証の伝聞リスト**であり、本書は**その実在・仕様・ライセンス状態を確認できない**。本表はあくまで「こういうものがあるらしい」という索引にすぎず、**ダウンロードや利用を推奨するものではない**。利用前には必ず出典・ライセンス・適法性を自ら確認すること（第6章の六つの鉄則を参照）。

## A.1　蒸留モデル（巷で流布、未検証）

| モデル名 | ベース | 称される仕様／位置づけ | 称される出典 |
|---|---|---|---|
| **Qwable-v1** | Qwen3.6-35B-A3B | SFT、Fable 5 のツール呼び出しを継承。約24GB(Q4_K_M)、約102 tok/s | r/huggingface、r/LocalLLaMA、r/opencodeCLI |
| **GPT-OSS 120B Fable-5 Distilled** | GPT-OSS 120B | MoE 128エキスパート／アクティベート4。約72GB、約35〜50 tok/s。GGUF(Q8_0/Q5_0)。AutoTrust AI Lab の演算力 | autotrust/gpt-oss-120b-Fable-5-Distilled-GGUF（HF）|
| **Qwythos-9B-Claude-Mythos-5-1M** | Qwen3.5-9B | 全パラメータ微調整。1M コンテキスト、脱検閲、4GBで動作可。約6.5GB、約180 tok/s | Empero AI、Interfaze、Threads、零度解说 |
| **richardyoung/qwythos-9b-abliterated** | Qwythos-9B | abliteration による二次脱検閲バリアント | Ollama、HF |
| **Qwythos-27B**（予告）| 不明 | Empero Roadmap が掲げる次世代「Mythos 級」27B | Empero |
| **Qwen3.6-14B-A3B-FableVibes** | Qwen3.6-35B-A3B「heretic」枝刈り | Pruning + QLoRA MoE。「Claude らしさ」を保持。約11GB、約140 tok/s | tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF（HF）|
| **clzoro/Qwen3.5-27B-Claude-distill** | Qwen3.5-27B（密）| Opus の CoT を学習、文学／ロールプレイに強い。約19GB、約110 tok/s | HF |
| **Dzluck/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled** | Qwen3.5-2B | エッジデバイス向け超軽量。約2.1GB、約250 tok/s。GGUF 同梱 | HF |
| **Jackrong シリーズ**（27B Coder など）| Qwen 系 | 軌跡反転で CoT を処理、コミュニティで「最も信頼される」| LocalLLaMA |

## A.2　データセット（巷で流布、未検証）

| データセット | 内容 | 用途 | リスク |
|---|---|---|---|
| **armand0e/claude-fable-5-claude-code** | Fable 5 の生のエージェント実行軌跡（プライバシー除去、Glint-Research と協業）、OpenAI 形式に変換可 | プログラム／ツール呼び出しの蒸留 | ⚠️ 商用モデルの軌跡、ToS／法的リスク大 |
| **ox-ox/mythos-character-distillation** | Claude の「話し方と心理特性」：自発的メタ認知、定型文の拒否、言い訳せず自己修正 | 人格／文体の蒸留 | ⚠️ 同上 |

## A.3　Fable 5／蒸留に関する報道・議論（タイトル、未検証）

- 「Anthropic、Claude Fable 5 に蒸留検知機能を追加」— 動區動趨（蒸留検知 → 自動的に Opus へ退避）
- 「Anthropic Fable 5 がほぼ全 AI ベンチマークを席巻、モデル蒸留攻撃を初めてブロック対象に」— inside.com.tw
- 「Fable 5 Not Available? What Actually Happened to Claude's Newest Model」（米国の輸出管理指令に言及）
- 「Anthropic Accuses Alibaba of Distilling Claude AI Model Capabilities」— Global Banking & Finance Review
- 「Model Distillation: The Mechanics of Stealing an AI's Intelligence」
- 「Claude がオープンソース化された？Qwythos-9B が突如リリース！脱検閲・104万コンテキスト・4GB VRAM で動作」— 零度解说
- 「またも強力なオープンソース推論モデル登場！Empero が Qwythos-9B-Claude-Mythos-5 を公開」— Threads
- 「Qwythos-9B-Claude-Mythos-5-1M GPU Requirements: VRAM & Cheapest GPU」— Interfaze
- Reddit スレッド：r/LocalLLaMA（mynamasteph：「I would not trust any Claude distill…」）、r/opencodeCLI（TomLucidor：「we kind need SFT/RL…」）、r/huggingface（Qwable-v1 release）

## A.4　DGX Spark ハードウェアの出典（タイトル、未検証）

- 「Blackwell アーキテクチャ採用の AI パーソナルスーパーコンピュータ | NVIDIA DGX Spark」— NVIDIA（GB10 Grace Blackwell）
- 「GB10 チップ採用、NVIDIA と複数のシステムメーカーが新形態の AI ワークステーションを投入」（iGPU は GB100 と同源、第5世代…）
- 「NVIDIA GB10 Grace Blackwell スーパーチップ——デスクトップ AI 時代」（統合メモリアーキテクチャ UMA）
- 「Solved the DGX Spark, 102 stable tok/s Qwen3.5-35B-A3B」— Reddit（実測の参照点）
- 「DGX Spark, what models are you running?」— Reddit
- 「麗臺 NVIDIA DGX Spark デスクトップ型 AI スーパーコンピュータ（GB10/128G/4TB SSD/DGX OS）」、「DGX Spark Founders Edition」

## A.5　プロンプトエンジニアリング／蒸留手法の出典（Q6 関連、タイトル、未検証）

- platform.claude.com — Prompting best practices（clarity、examples、XML、thinking、agentic）
- GitHub — Claude Code 各バージョンのシステムプロンプトとトークン数のまとめ
- arXiv — 「Training Small Critic Agents」（Opus-4.6 teacher／detailed prompt）
- systemprompt.io — Daily Development Workflows（.md best practices、settings.json）
- claudemarketplaces.com/skills/.../knowledge-distillation — KD 実装ガイド
- Drew Breunig（dbreunig.com）、LinkedIn「Lessons from Leaked Source」、youmind（NotebookLM）、uxplanet（Prompting Best Practices）

## A.6　実在し利用可能なオープンソースのベースとデータセット（第2章対応）

> A.1〜A.2 の「巷で流布、未検証」の雪のような名前とは異なり、以下は**確かに公開され、Hugging Face で検証できる**ベースファミリーと蒸留に常用されるデータセットである。第2章の差し替え可能なパイプラインは、実際にはこれらの上に構築されている——「Claude distill」を称する成果物も、その下層はほぼこれらのベースに、これら（または自作）のデータを加えたものだ。

| 実在するベースファミリー | 出典 | 蒸留での位置づけ |
|---|---|---|
| **Qwen（Qwen2.5 / Qwen3）** | Alibaba | 中英バイリンガルに強く、サイズが揃う。コミュニティ微調整の第一選択 ✅ |
| **Llama 3.x** | Meta | エコシステム最大、GQA で推論にやさしい ✅ |
| **Mistral / Mixtral** | Mistral AI | 密・MoE の両方あり、欧州系のオープンウェイト ✅ |
| **Gemma 2 / 3** | Google | 軽量な端末側向け、ライセンスが比較的寛容 ✅ |
| **GPT-OSS（20B / 120B）** | OpenAI | MoE オープンウェイト。A.1 の 120B 項目はこれが根拠 ✅ |
| **Phi-3 / Phi-4** | Microsoft | 小型モデル、合成データ志向 ✅ |
| **DeepSeek-V2/V3 / R1** | DeepSeek | MLA で KV を節約、R1 の推論軌跡はよく蒸留教材になる ✅ |

| 実在する公開データセット | 内容 | 用途 |
|---|---|---|
| **OpenHermes 2.5** | 百万級の多源指示／対話 | 汎用 SFT ✅ |
| **UltraChat / UltraFeedback** | 大規模なマルチターン対話／選好ラベル | SFT + 選好アライメント ✅ |
| **OpenOrca / SlimOrca** | GPT 生成の CoT 問題解決軌跡 | 推論の蒸留 ✅ |
| **Magpie** | アライメント済みモデルが自己生成した指示ペア | 人手ゼロの合成 SFT ✅ |
| **Tülu 3（mixtures 含む）** | AllenAI 公開のポストトレーニング・レシピとデータ | 完全な post-training の参考 ✅ |
| **distilabel pipelines** | Argilla の合成／書き換え／採点パイプライン | 蒸留データの自作 ✅ |

> 真偽校正 一覧
> **比較的信頼できる（実在の技術／製品に対応）**：Anthropic、Claude、Fable 5、Qwen、Llama、Mistral、Gemma、GPT-OSS、Phi、DeepSeek、Hugging Face、Ollama、GGUF、QLoRA、abliteration、OpenHermes / UltraChat / OpenOrca / Magpie / Tülu / distilabel、NVIDIA DGX Spark / GB10 / LPDDR5X UMA。
> **検証不能（伝聞またはハルシネーションの可能性が極めて高い）**：Mythos モデル、Qwythos-9B、Qwable-v1、FableVibes、Empero AI の具体的な実在と仕様、「Fable 5 が公開数日で輸出管理により撤去」「4,659 件の軌跡」「蒸留検知で自動的に Opus 4.8 へ降格」。A.1 表の具体的な tok/s、GB 数、「Fable 5 のツール呼び出しを継承」は、いずれも伝聞の主張であり実測されていない。
