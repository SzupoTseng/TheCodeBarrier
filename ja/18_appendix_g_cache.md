# 付録G　大規模モデルのキャッシュ（LLM Cache）技術早見表

> 第10〜11章に登場する「KV Cache / Prompt Cache / 推論エンジン / アーキテクチャ方法論」のエコシステムを、主題ごとに対照表へまとめる。**大半の項目は実在する公開論文またはオープンソースプロジェクト**（自分で検証可能）であり ✅ を付す。ごく一部、本書の語りに現れる「内部アーキテクチャの主張／具体的なスペック／削減率の数値」は未検証であり ⚠️ を付し、巻末の校正区にまとめて説明する。引用前にはバージョンとライセンスを必ず自分で確認すること（第6章の六つの鉄則を参照）。

## G.1　中核概念

| 名称 | 一言での説明 | 備考 |
|---|---|---|
| **KV Cache（キーバリューキャッシュ）** | 自己回帰生成時に計算済みの Key/Value 行列を VRAM に保存し、再計算を避けることで注意機構の計算量を O(N²) から O(N) へ削減する | 推論層キャッシュ ✅ |
| **Prompt Cache / Context Cache** | API・サービス層が同一**プレフィックス**の KV 状態をサーバ側にキャッシュし、重複する Prefill を省く。ヒット後の入力トークン料金はしばしば1〜2割まで下がる | サービス層キャッシュ ✅ |
| **Prefill vs Decode** | Prefill＝入力全体を一度に並列処理（計算律速）；Decode＝トークンを逐次生成（メモリ律速）。両段階の特性は正反対 | ✅ |
| **Memory-Bound vs Compute-Bound** | 長文の KV Cache は容易に数 GB に達し、GPU を「計算律速」から「メモリ帯域律速」へ追い込む。キャッシュの読み出しが再計算より遅くなることもある | ✅ |
| **Attention Sink** | モデル最初の2〜4トークンが異常に大きな注意重みを占有する現象。StreamingLLM のストリーミング復号における鍵となる発見 | ✅ |
| **VRAM 断片化（Fragmentation）** | 従来の KV Cache はリクエストごとに連続した最大 VRAM を事前確保する必要があり、大量の遊休領域を浪費する → PagedAttention が解決 | ✅ |

## G.2　VRAM 圧縮／注意機構の構造

| 名称 | 何をするか | 出典／ソース |
|---|---|---|
| **MHA（Multi-Head Attention）** | 古典的なマルチヘッド注意機構。KV ヘッド数＝Q ヘッド数で、KV Cache のオーバーヘッドが最大 | Vaswani et al., 2017 ✅ |
| **MQA（Multi-Query Attention）** | 全 Q ヘッドが1組の K/V を共有し、KV Cache を 1/H に削減するが、表現力は損なわれる | Shazeer, 2019 ✅ |
| **GQA（Grouped-Query Attention）** | 折衷案：Q ヘッドをグループ分けし各グループが1組の KV を共有（例：Llama 3 は8組を使用）。業界の主流 | Ainslie et al., 2023（Llama が採用）✅ |
| **MLA（Multi-head Latent Attention）** | 低ランク圧縮で K/V を極めて低次元の潜在空間ベクトルへ射影し、復号時にオンラインで解凍する。KV Cache を90%以上削減 | DeepSeek-V2/V3 が提唱 ✅；「DeepMind も深く追随」⚠️ |
| **KV-Quant（2-bit 量子化）** | Per-Channel/Per-Token 混合量子化＋外れ値の保持により、KV を 2〜3 bit まで圧縮してもほぼ精度劣化なし | 論文『KVQuant…』（NeurIPS 2024）✅；「100B トークンの窓」という表題 ⚠️ |
| **H2O（Heavy-Hitter Oracle）** | 注意がべき乗則分布を示すことを発見し、少数の Heavy-Hitter トークンの KV のみを保持し残りを動的に除去 → 定長キャッシュ | 論文『H2O…』NeurIPS 2023 ✅ |
| **StreamingLLM** | 「最初の数トークン（Attention Sink）」＋「スライディングウィンドウ」を保持し、Prefill の再計算なしにストリーミングで無限長文を扱う | 論文『Efficient Streaming LM with Attention Sinks』ICLR 2024 ✅ |
| **Speculative Decoding + Cache** | 投機的復号で木構造の KV Cache を用いて草案分岐を並列検証し、多経路の重複計算を回避 | Medusa／木構造 KV 系列 ✅ |
| **Infinite-LLM / ハイブリッド RNN-Transformer** | 窓を超えた KV を固定サイズの循環状態へ圧縮し、空間計算量を O(N)→O(1) に | Griffin/Hawk、Mamba-2 路線 ✅ |

## G.3　推論エンジン（オープンソース／セミオープンソース）

| エンジン | 中核の決め技 | 出典／ソース |
|---|---|---|
| **vLLM** | **PagedAttention**：OS の仮想記憶ページングを借りて KV を管理し、VRAM 利用率をほぼ100%に；Chunked Prefill / Prefix Caching | UC Berkeley、オープンソース ✅ |
| **SGLang** | **RadixAttention**：基数木（Radix Tree）で Prompt プレフィックスのキャッシュを自動管理し、マルチターン／Tool-use のヒット率が高い | UC Berkeley、オープンソース ✅ |
| **LMDeploy** | KV Cache 量子化（W4A16、KV INT4/INT8）が極めて充実＋Persistent RPC で、エッジ／オンプレに適する | 上海 AI Lab（書生・浦語）、オープンソース ✅ |
| **TensorRT-LLM** | In-flight Batching、FlashDecoding+；ビット単位の VRAM 制御と Triton 統合 | NVIDIA、セミオープンソース ✅ |
| **LightLLM** | **Token Attention**：トークン単位の細粒度 VRAM スケジューリングで、リクエスト長が極端に不均一でも OOM に強い | オープンソース ✅ |
| **DeepSpeed-FastGen** | **Split-Fused / Dynamic SplitFuse**：Prefill と Decode を同一バッチ内で動的に交織・分割 | Microsoft、オープンソース ✅ |
| **Hugging Face TGI** | 本番級の推論サービス（連続バッチ、量子化、テンソル並列）、Prefix Caching を含む | Hugging Face、オープンソース ✅ |

## G.4　アーキテクチャ方法論

| 手法 | 一言 | 備考 |
|---|---|---|
| **Cache-Aware Routing** | ゲートウェイが System Prompt のプレフィックスハッシュを計算し、同一プレフィックスのリクエストを当該物理キャッシュを保持する同一ノードへルーティング | ✅（概念／実践）|
| **Tiered Cache Orchestration** | HBM＝L1、DDR5＝L2、NVMe SSD＝L3；遊休 KV をバックグラウンドで Offload/Prefetch して退避・復帰 | ✅（PCIe Gen5 の速度は状況の一例 ⚠️）|
| **Chunked Prefill** | 長文の Prefill を固定サイズの Chunk（例：512トークン）に分割し、Decode とパイプラインで交錯させ、TTFT のばらつきを抑える | ✅ |
| **Dynamic KV Eviction** | 注意重みと組み合わせてトークン重要度を算出し、句読点・虚詞を優先的に解放、実体名詞を保持（損失ありだが高知能）| ✅ |
| **Deterministic Prompt Formatting** | 「静的を前、動的を後ろ」；Tools/System/Few-shot を固定のアルファベット順で直列化し、冒頭に UUID／タイムスタンプを挟むのを禁止 | ✅ |
| **Exact vs Semantic Cache** | Exact＝トークン完全一致で RadixTree を辿り KV を再利用；Semantic＝ベクトル類似度 >0.98 で前回の回答を直接返す | ✅ |
| **PD Disaggregation（プレフィル／デコード分離）** | Prefill ノード（高計算力）が KV を算出し、RDMA で Decode ノード（大 VRAM）へ送って続きを計算 | ✅（トレンド）；具体的なインフラ詳細 ⚠️ |
| **Context Cache Monetization** | `cache_control` でキャッシュアンカーを明示宣言し、長文タスクの入力費用を大幅に下げる | Anthropic の機構 ✅；「80%急落」は状況依存の数値 ⚠️ |

## G.5　低レイヤ演算子／システムプリミティブ

| 名称 | 一言 | ソース |
|---|---|---|
| **FlashAttention** | IO 感知・分割（tiling）融合の注意機構カーネルで、HBM の読み書きを節約 | Dao et al., 2022（v2/v3 が後続）✅ |
| **FlashDecoding** | Decode 段階に向けて系列次元に沿って並列化し、長文脈での単一トークン生成を高速化 | Stanford／FlashAttention チーム ✅ |
| **PagedAttention** | KV を固定サイズの「メモリページ」に離散保存し、断片化を解消、共有／Copy-on-Write をサポート | vLLM 論文 SOSP 2023 ✅ |
| **Radix Tree** | 圧縮プレフィックス木。SGLang が複数リクエストの Prompt KV プレフィックスを自動共有・再利用するために使用 | データ構造（SGLang での応用）✅ |
| **RoPE（回転位置エンコーディング）** | 回転式の相対位置エンコーディングで、KV Cache のオフセットや長さ外挿に関連 | Su et al., 2021 ✅ |
| **MurmurHash3 プレフィックスハッシュ** | 非暗号の高速ハッシュ。Cache-Aware Routing がプレフィックスの指紋を算出して振り分けに用いる | 汎用ハッシュ（Appleby）✅ |
| **RDMA KV transfer** | ノード間でリモートダイレクトメモリアクセスにより KV Cache を「押し出す」、PD 分離の転送基盤 | InfiniBand/RoCE ✅；「Jupiter 1.6 Tbps」⚠️ |

## G.6　セマンティックキャッシュのエコシステム

| 名称 | 役割 | ソース |
|---|---|---|
| **GPTCache** | セマンティックキャッシュの仲介層：類似入力を既存の回答へマッピングし、ヒットすれば LLM 呼び出しを省く | オープンソース（Zilliz）✅ |
| **Milvus** | 大規模ベクトルデータベース。セマンティックキャッシュの類似度検索バックエンド | オープンソース（Zilliz）✅ |
| **Qdrant** | Rust 製ベクトルデータベース。セマンティックキャッシュ／RAG でよく使われる検索バックエンド | オープンソース ✅ |
| **ANN 類似度検索** | 近似最近傍（HNSW/IVF など）で埋め込みの類似度マッチングを行い、>0.98 をヒットとみなす | 汎用手法 ✅ |

## G.7　ヒット率の落とし穴回避（第11章「黄金律」）

| よくある誤った実践（Cache 失効を招く）| 正しい戦略（ヒット最大化）|
|---|---|
| System Prompt に動的変数（タイムスタンプ、ユーザー ID）を詰める | 動的なユーザー情報はメッセージの**最末尾**の User 欄へ移し、冒頭は固定する |
| 毎ターン履歴の要約（Summary）を再生成する | 履歴は**追記のみで改変しない**；切り詰めは**最古**から削除し、プレフィックスの一貫性を保つ |
| 工具定義（Tools）の順序をランダム化する | JSON 工具リストの内部の並び・フィールド構造・前後順序を厳格に固定する |
| 空白・改行などの細部が不統一 | テキスト整形ロジックを統一し、ひとつの `\n` がトークン列を変えるのを防ぐ |
| 冒頭にランダムなソルト（Salt）／UUID を挿入する | `cache_control` で明示アンカーを置く；決定論的な直列化（Deterministic Ordering）|

> **黄金律**：「**安定した静的内容を最前に、動的で変化の激しい内容を最後に置く**」——プレフィックス一致こそ、あらゆる Prompt Cache の根である。

---

> ⚠️ **真偽校正 — 第10〜11章の語りで慎重に扱うべき主張**
>
> G.1〜G.7 の表で ✅ を付したものは、いずれも arXiv / GitHub / 公式ドキュメントで検証可能な**実在の論文またはオープンソースツール**である（MLA、H2O、StreamingLLM、KV-Quant、vLLM、SGLang、LMDeploy、TensorRT-LLM、LightLLM、DeepSpeed、TGI、FlashAttention、PagedAttention、GPTCache、Milvus、Qdrant、Anthropic の `cache_control` など）。
>
> 以下の、本書の語りに現れる内容は**本書では検証できない**ため、**既成事実ではなく妥当な工学的推論**とみなすべきである：
> - **「DeepMind の内部アーキテクチャ」に関する主張** ⚠️（MLA を「DeepMind も深く追随」、PD 分離を「Google DeepMind/OpenAI の究極の方向」）——方向性は妥当だが公開出典なし。
> - **特定の Gemini のコンテキストウィンドウサイズ／具体的なスペック** ⚠️。
> - **「Google Jupiter 1.6 Tbps RDMA」** ⚠️（具体的な帯域幅の数値は未検証）。
> - **正確な削減率**（「KV Cache を90%以上削減」「入力費用が80%急落」「類似度 >0.98」「2-bit でほぼ精度劣化なし」）⚠️——桁の方向性は参考になるが、**正確な数値はモデル／負荷により異なり**、保証値として引用すべきではない。
> - **PCIe Gen5 / HBM3e などのハードウェア型番**は状況の一例であり、実際の配備では公式スペックに従うこと。
