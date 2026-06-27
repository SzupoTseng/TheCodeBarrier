# 付録R　中核実装ソースコード参照

> 本付録は、本書全体の鍵となる技術の**参考実装**を一箇所にまとめたもの——蒸留損失、KV Cache/PagedAttention、量子化とデプロイ。コードは読みやすく、改変しやすく、実際のライブラリに近いことを目標とする（PyTorch / transformers / vLLM / llama.cpp / peft / distilabel）。各段に**目標**と**講評**を付す。すべての教師モデルは一律に**あなたが使用権を持つ対象**（オープンウェイトまたは自前モデル）と仮定する。本付録には防護回避・反フィンガープリント・脱検閲（abliteration）のコードは**一切含まない**——そのレッドラインは付録 E と第 6 章を参照。

> ⚠️ これらは**参考用の骨格**であり、本番環境にコピペでそのまま使えるものではない：実際のデプロイではエラー処理・形状チェック・分散同期・バージョン互換を自分で補う必要がある。「原理を理解した後の取っ掛かり」として扱うこと。

---

## R.1　知識蒸留の中核実装

本節では、知識蒸留（Knowledge Distillation, KD）を大規模言語モデルの訓練で用いる際の中核となるコード骨格をまとめる。いずれも PyTorch と Hugging Face `transformers` エコシステムを基礎とする。すべての教師モデルは、あなたが正当な使用権を持つオープンソースまたは自前のウェイトモデル（例えば自社の大型モデルから小型モデルを蒸留する）と仮定する。コードは読みやすく、そのまま移植できることを目標とし、実務ではデータパイプラインとデバイス数に応じて微調整が必要となる。

---

### R.1.1　Hinton のソフト・ハードラベル KD 損失

最も古典的な KD 損失：温度 `T` で教師と学生の確率分布を軟化し、KL ダイバージェンスで教師の「暗黙知（dark knowledge）」に近づけ、さらに `α` で真のラベルとの交差エントロピーと混合する。ソフトラベルの勾配は softmax の `1/T` スケーリングによって圧縮されるため、ハードラベルと同じオーダーを保つには `T²` を掛け戻す必要がある点に注意。

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5,
            ignore_index=-100):
    """Hinton (2015) のソフト＋ハードラベル蒸留。

    student_logits, teacher_logits: (B, V) または (B, L, V)
    labels: (B,) または (B, L) の正解 token id
    """
    # ソフトラベル：教師は逆伝播に関与しないため detach が必要
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)  # 1/T のスケーリングで圧縮された勾配の大きさを補正

    # ハードラベル：標準的な交差エントロピー（padding を自動的に無視）
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

**講評**：`reduction="batchmean"` こそが数学的定義上正しい KL の平均である（`"mean"` は語彙次元で余分に割ってしまい歪む）。`T*T` のスケーリングと教師の `detach()` は、最も見落とされがちな2つの細部である。

---

### R.1.2　Forward KL と Reverse KL（mode-covering vs mode-seeking）

KD の方向性が、学生がどう「妥協」するかを決める。Forward KL `KL(p_teacher ‖ p_student)` は mode-covering：学生は教師のすべての台集合（support）をカバーするよう強いられ、小型モデルでは過度に平滑で幻覚を含む分布を学びやすい。Reverse KL `KL(p_student ‖ p_teacher)` は mode-seeking：学生は教師の高確率モードに集中し、よりシャープで安定した出力を生む。これが MiniLLM や GKD が LLM 蒸留で reverse KL を好む理由である。

```python
import torch.nn.functional as F

def forward_kl(student_logits, teacher_logits, T=1.0):
    """KL(teacher || student) — mode-covering。学生が教師の全分布をカバーする。"""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def reverse_kl(student_logits, teacher_logits, T=1.0):
    """KL(student || teacher) — mode-seeking。学生が教師の主要モードに集中する。"""
    # 注意：第1引数は log-prob でなければならないため、teacher に log_softmax を適用；
    # target は student の確率だが、逆伝播を許す必要があるため student は detach しない。
    p_student = F.softmax(student_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits.detach() / T, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean() * (T * T)
```

**講評**：生成系 LLM の蒸留では通常 reverse KL（または GKD の Jensen-Shannon 一般化補間）を選び、forward KL が学生に「何でも少しずつ学ぶ」ことを強いて、流暢だが信頼できないロングテールを生むのを避ける。分類や語彙が制限されたタスクなら forward KL でも十分である。

---

### R.1.3　系列レベル KD／on-policy 蒸留 loop の骨格

系列レベル KD（GKD/on-policy distillation）の鍵：訓練データを**学生自身がサンプリング**し、学生が生成した系列を教師が採点することで、teacher-forcing 下の露出バイアス（exposure bias）を解消する。以下は単一訓練ステップの骨格。

```python
import torch

@torch.no_grad()
def sample_from_student(student, tokenizer, prompts, max_new_tokens=256):
    """On-policy：学生自身が系列を生成する（後で教師が採点する）。"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(student.device)
    gen = student.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=1.0, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].size(1)
    return gen, prompt_len  # gen: (B, prompt_len + new_len)

def on_policy_step(student, teacher, tokenizer, prompts, kl_fn, optimizer, T=1.0):
    # 1) 学生による自己サンプリング（on-policy データ）
    seqs, prompt_len = sample_from_student(student, tokenizer, prompts)
    attn = (seqs != tokenizer.pad_token_id).long()

    # 2) 学生が自身の系列を順伝播（勾配が必要）
    s_logits = student(seqs, attention_mask=attn).logits  # (B, L, V)

    # 3) 教師が同じ系列を採点（凍結、no_grad）
    with torch.no_grad():
        t_logits = teacher(seqs, attention_mask=attn).logits

    # 4) 生成部分（response）のみで loss を計算、shift して next-token に整合
    s_resp = s_logits[:, prompt_len - 1:-1, :]
    t_resp = t_logits[:, prompt_len - 1:-1, :]
    loss = kl_fn(s_resp.reshape(-1, s_resp.size(-1)),
                 t_resp.reshape(-1, t_resp.size(-1)), T=T)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    return loss.item()
```

**講評**：on-policy サンプリングはコストが高く、実務では `lambda`（GKD の `lmbda`）の確率で on-policy と固定データを混ぜることが多い。`trl.GKDTrainer` はこのロジックをすでにカプセル化しており、自前の骨格は主に理解とカスタマイズのために使う。

---

### R.1.4　CoT 加重損失（`<think>...</think>` 内 token の加重）

推論チェーン（chain-of-thought）を蒸留する際、思考部分は最終的な答えよりも教師の推論能力を多く担うことが多い。以下では系列中の `<think>` と `</think>` マーカーに基づいて token ごとの重みマスクを作り、思考部分により高い重みを与える。

```python
import torch

def build_cot_weight_mask(input_ids, tokenizer, think_weight=2.0, base_weight=1.0,
                          open_tag="<think>", close_tag="</think>"):
    """各 token に重みを生成；<think>...</think> 区間内では重みが高い。"""
    open_id = tokenizer.convert_tokens_to_ids(open_tag)
    close_id = tokenizer.convert_tokens_to_ids(close_tag)
    weights = torch.full_like(input_ids, base_weight, dtype=torch.float)
    for b in range(input_ids.size(0)):
        inside = False
        for t in range(input_ids.size(1)):
            tok = input_ids[b, t].item()
            if tok == open_id:
                inside = True
            weights[b, t] = think_weight if inside else base_weight
            if tok == close_id:
                inside = False
    return weights  # (B, L)

def weighted_token_loss(per_token_loss, weight_mask, pad_mask):
    """per_token_loss/(weight_mask/pad_mask): (B, L)；スカラーを返す。"""
    w = weight_mask * pad_mask  # padding の重みをゼロにする
    return (per_token_loss * w).sum() / w.sum().clamp_min(1.0)
```

**講評**：`per_token_loss` は `F.cross_entropy(..., reduction="none")` または token ごとの KL から得られる。ラベルの区切りに特殊 token を使う場合（`<think>`/`</think>` を tokenizer の special tokens に追加することを推奨）、BPE で複数のサブワードに分割されて境界を取りこぼすのを避けられる。

---

### R.1.5　語彙横断の整合（cross-tokenizer logit projection）

教師と学生が異なる tokenizer を使う場合、logit を位置ごとに整合できない。よくある方法：教師の top-k token を取り、文字列にデコードし、学生の tokenizer で再エンコードし、教師の確率質量を学生の語彙上の対応する（先頭）token に再配分する。以下は概略の骨格で、完全なサブワード結合戦略は省略している。

```python
import torch

@torch.no_grad()
def project_teacher_to_student(teacher_logits, teacher_tok, student_tok,
                               student_vocab_size, top_k=64, T=1.0):
    """教師の単ステップ分布を学生の語彙へ投影する（top-k 文字列の再エンコード + 質量の再配分）。

    teacher_logits: (V_t,) 単一位置の教師 logits
    戻り値: (V_s,) 学生語彙上の目標確率分布
    """
    probs = torch.softmax(teacher_logits / T, dim=-1)
    topk = torch.topk(probs, k=top_k)
    student_dist = torch.zeros(student_vocab_size, device=teacher_logits.device)

    for tok_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
        # 教師 token -> 文字列 -> 学生 tokenizer で再エンコード
        piece = teacher_tok.decode([tok_id]).strip()
        if not piece:
            continue
        s_ids = student_tok.encode(piece, add_special_tokens=False)
        if not s_ids:
            continue
        # 質量は学生がエンコードした先頭 token に与える（簡略戦略；サブワード鎖に沿った分配も可）
        student_dist[s_ids[0]] += p

    total = student_dist.sum()
    if total > 0:
        student_dist /= total  # 再正規化し、整合できない末尾を捨てる
    return student_dist
```

**講評**：これは概略であって本番級ではない——厳密な方法では、複数サブワードの token の確率をチェーンに沿って分配すること、空白プレフィックス（`▁`/`Ġ`）の差異、正規化バイアスを扱う必要がある。ULD（Universal Logit Distillation）は最適輸送（optimal transport）で2つの語彙を整合させ、より頑健な代替となる。

---

### R.1.6　完全な訓練ステップ（HF Trainer-style）

上記のモジュールを統合してカスタム `Trainer` を作る：教師は凍結して `eval()` に設定し、学生は PEFT LoRA と組み合わせて VRAM を削減できる。分散は `accelerate` 設定ファイルで DeepSpeed ZeRO-3 または FSDP を有効化し、勾配チェックポイントを開く。

```python
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

class DistillTrainer(Trainer):
    def __init__(self, teacher_model, T=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T, self.alpha = T, alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        s_out = model(input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"])
        with torch.no_grad():
            t_out = self.teacher(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"])
        # 蒸留損失（R.1.1）：next-token 整合のため shift が必要
        s_logits = s_out.logits[:, :-1, :].contiguous()
        t_logits = t_out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = kd_loss(s_logits, t_logits, shift_labels,
                       T=self.T, alpha=self.alpha)
        return (loss, s_out) if return_outputs else loss

# --- 組み立て ---
student = AutoModelForCausalLM.from_pretrained("your/small-base",
                                               torch_dtype=torch.bfloat16)
student.gradient_checkpointing_enable()         # VRAM 節約
student.config.use_cache = False                # 勾配チェックポイントと排他
student = get_peft_model(student, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM"))
teacher = AutoModelForCausalLM.from_pretrained("your/teacher",
                                               torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("your/small-base")

args = TrainingArguments(
    output_dir="./kd-out", per_device_train_batch_size=2,
    gradient_accumulation_steps=8, learning_rate=1e-4,
    bf16=True, gradient_checkpointing=True,
    deepspeed="ds_zero3.json",   # または accelerate config で FSDP を有効化
    logging_steps=10, num_train_epochs=3,
)
trainer = DistillTrainer(teacher_model=teacher, T=2.0, alpha=0.5,
                         model=student, args=args,
                         train_dataset=train_ds, tokenizer=tokenizer)
trainer.train()
```

起動時は `accelerate launch --config_file fsdp.yaml train.py` で、あるいは `TrainingArguments` に直接 `deepspeed="ds_zero3.json"` を掛ける（ZeRO-3 はパラメータ・勾配・オプティマイザ状態の三者を各 GPU にシャーディングする）。教師が大きすぎる場合は、別プロセスで／量子化（8-bit など）してロードし、前向き採点のみを行わせればよい。

**講評**：教師は必ず `eval()` ＋ `requires_grad_(False)` とし、`no_grad` で包むこと。さもないと更新しない勾配に VRAM と演算力が浪費される。`gradient_checkpointing` と `use_cache=True` は排他で、cache を手動でオフにしないと警告が出るか無効化される。


---

## R.2　KV Cache とキャッシュの中核実装

本節では、推論サービスにおいて KV Cache に関わる中核的な参考実装をまとめる。VRAM 見積もり、PagedAttention のページ管理、CUDA アドレッシング、RadixAttention のプレフィックス共有から、FlashDecoding、KV 量子化、多層キャッシュルーティングまでを扱う。すべてのコードは可読性を最優先し、意味的な正しさを本旨とする。命名は第 10・11 章と一致させた（vLLM / SGLang / FlashAttention / Triton）。

---

### R.2.1　KV Cache VRAM 計算機

**用途**：ある系列の KV Cache が占有するバイト数を見積もる。容量計画と batch 上限の導出の出発点である。

```python
def kv_cache_bytes(
    L: int,            # Transformer 層数 (num_layers)
    kv_heads: int,     # KV head 数 (GQA/MQA 後の num_key_value_heads)
    head_dim: int,     # 各 head の次元
    seq_len: int,      # 系列長 (prompt + generated)
    batch: int = 1,    # 並行系列数
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp8/int8 = 1
) -> int:
    """KV Cache の VRAM = 2 (K と V を各1つずつ) * dtype_bytes
       * L * kv_heads * head_dim * seq_len * batch"""
    return 2 * dtype_bytes * L * kv_heads * head_dim * seq_len * batch


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


if __name__ == "__main__":
    # Llama-3-8B: L=32, num_key_value_heads=8 (GQA), head_dim=128, bf16
    per_token = kv_cache_bytes(L=32, kv_heads=8, head_dim=128,
                               seq_len=1, batch=1, dtype_bytes=2)
    print("每 token:", human(per_token))                 # 128.00KB

    total = kv_cache_bytes(L=32, kv_heads=8, head_dim=128,
                           seq_len=8192, batch=128, dtype_bytes=2)
    print("batch=128 x 8K:", human(total))               # ~128.00GB
```

**講評**：token あたり 128KB は Llama-3-8B の鍵となる定数である——`2 * 2 * 32 * 8 * 128 = 131072 = 128KB`。GQA は KV head を 32 から 8 に削り、cache を 4 倍縮小するのに等しい。MHA（kv_heads=32）を使うと token あたり 512KB に跳ね上がり、batch=128×8K でいきなり 512GB を食い尽くす。実務では `2 * dtype_bytes * L * kv_heads * head_dim` を「token あたり定数」として先に計算し、`seq_len * batch` を掛けると、容量の天井を素早く暗算できて便利である。

---

### R.2.2　PagedAttention BlockManager / BlockTable

**用途**：vLLM 風の固定サイズのブロック（block）で KV Cache の VRAM を管理し、連続割り当てによる内部・外部の断片化を解消する。

```python
from collections import deque
from typing import Dict, List


class PhysicalTokenBlock:
    """block_size 個の token を固定で収容する物理 VRAM ブロック。"""
    def __init__(self, block_number: int, block_size: int):
        self.block_number = block_number   # 物理ブロックのインデックス
        self.block_size = block_size
        self.ref_count = 0                 # いくつの系列で共有されているか (copy-on-write 用)

    def __repr__(self):
        return f"PhysicalTokenBlock(#{self.block_number}, ref={self.ref_count})"


class BlockAllocator:
    """free-list で物理ブロックの割り当て/解放を管理、O(1) 割り当て。"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.all_blocks = [PhysicalTokenBlock(i, block_size)
                           for i in range(num_blocks)]
        self.free: deque = deque(self.all_blocks)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free:
            raise MemoryError("no free KV blocks (preemption required)")
        block = self.free.popleft()
        block.ref_count = 1
        return block

    def free_block(self, block: PhysicalTokenBlock) -> None:
        assert block.ref_count > 0
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free)


class BlockTable:
    """単一系列の logical block -> physical block マッピング。"""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.blocks: List[PhysicalTokenBlock] = []
        self.num_tokens = 0

    def append_token(self) -> None:
        block_size = self.allocator.block_size
        # 最後のブロックが満杯（またはまだブロックが無い）ときのみ、新しい物理ブロックを割り当てる
        if self.num_tokens % block_size == 0:
            self.blocks.append(self.allocator.allocate())
        self.num_tokens += 1

    def physical_index(self, logical_token_idx: int) -> int:
        """logical token 位置を (block_number, offset) に翻訳する。"""
        block_size = self.allocator.block_size
        block = self.blocks[logical_token_idx // block_size]
        offset = logical_token_idx % block_size
        return block.block_number, offset

    def fork(self) -> "BlockTable":
        """beam search / parallel sampling の copy-on-write 分岐：
           既存のブロックを共有し、ref_count を増やすだけ。"""
        child = BlockTable(self.allocator)
        child.blocks = list(self.blocks)
        child.num_tokens = self.num_tokens
        for b in child.blocks:
            b.ref_count += 1
        return child

    def free(self) -> None:
        for b in self.blocks:
            self.allocator.free_block(b)
        self.blocks.clear()
        self.num_tokens = 0
```

**講評**：PagedAttention の真髄は、KV Cache を OS のページング式の固定ブロックに切ることにある。`BlockTable` が「ページテーブル」、`BlockAllocator.free` が「空きページのリスト」に相当する。`ref_count` により parallel sampling / beam search が同一 prompt の KV を共有でき（copy-on-write）、ある分岐が共有済みブロックに書き込むときにのみ実際にコピーする。`num_tokens % block_size == 0` が新ブロックを割り当てる唯一の条件であり——これがページングでメモリ断片を均し、利用率を 90%+ に押し上げる鍵である。`num_free == 0` のときは必ず例外を投げてスケジューラに preemption（swap-out または recompute）を行わせ、サイレントに失敗させてはならない。

---

### R.2.3　paged_attention CUDA kernel のアドレッシング中核

**用途**：CUDA kernel の内部が block_table を介して logical token を物理 VRAM ポインタに翻訳し、K/V を読み取る様子を示す。

```cuda
// 単一の query head について、ある系列のすべての KV token を走査する。
// block_table はその系列の「ページテーブル」：logical block idx -> physical block number。
// kv_cache レイアウト: [num_blocks, num_kv_heads, head_dim, block_size]
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    scalar_t*       __restrict__ out,          // [num_heads, head_dim]
    const scalar_t* __restrict__ q,            // [num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,      // 物理 KV プール
    const scalar_t* __restrict__ v_cache,
    const int*      __restrict__ block_table,  // [max_blocks_per_seq]
    const int       context_len,               // この系列の現在の token 数
    const int       num_kv_heads) {

    const int kv_head = blockIdx.y;            // この thread block が担当する KV head
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int logical_block = 0; logical_block < num_blocks; ++logical_block) {
        // 1) ページテーブル参照：logical block -> 物理 block number
        const int physical_block = block_table[logical_block];

        // 2) このブロック内で処理すべき token 数（最後のブロックは満杯でないことがある）
        const int tokens_in_block =
            min(BLOCK_SIZE, context_len - logical_block * BLOCK_SIZE);

        for (int t = 0; t < tokens_in_block; ++t) {
            // 3) 物理オフセットを計算：(physical_block, kv_head) の head_dim 連続区間に位置決め
            const int64_t base =
                (((int64_t)physical_block * num_kv_heads + kv_head)
                 * HEAD_DIM) * BLOCK_SIZE;

            const scalar_t* k_ptr = k_cache + base + t;  // K[..., :, t]
            const scalar_t* v_ptr = v_cache + base + t;

            // 4) dot(q, k) -> score、online-softmax に累積（ここでは省略）
            //    実装では running max / running sum と組み合わせて数値的に安定なオンライン softmax を行い、
            //    score * v を out に加算していく。
            (void)k_ptr; (void)v_ptr;
        }
    }
}
```

**講評**：KV が VRAM 上で「連続」だと仮定する従来の attention に比べ、paged kernel は `block_table[logical_block]` という間接アドレッシングの層が1つ増える——これこそ PagedAttention が非連続の物理ページから論理的に連続な系列を組み立てられる代償であり真髄である。レイアウト `[num_blocks, num_kv_heads, head_dim, block_size]` は同一 head の連続 token を最内次元に置き、warp 内の thread が coalesced に読めるようにする。実際の vLLM kernel は内層で online-softmax（running max + running sum）を使い、完全な score 行列を保存せずに済ませる。本スケッチは「ページテーブル参照 → 物理オフセット」というアドレッシングの主筋に焦点を当てている。

---

### R.2.4　RadixAttention プレフィックスツリー

**用途**：SGLang 風の radix tree で、同じプレフィックス（system prompt、few-shot、複数ターンの履歴）の KV Cache を共有し、リクエストをまたいだ自動キャッシュ再利用を実現する。

```python
import time
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """エッジに一連の token 列を持ち、ノードにそのプレフィックスに対応する KV cache ポインタ（ブロック一覧）を掛ける。"""
    def __init__(self):
        self.children: Dict[int, "RadixNode"] = {}  # 先頭 token を key とする
        self.key: List[int] = []                    # このノードから親ノードまでの token 区間
        self.kv_blocks: List[int] = []              # 対応する KV 物理ブロック id
        self.parent: Optional["RadixNode"] = None
        self.last_access = time.monotonic()         # LRU 用


class RadixCache:
    def __init__(self):
        self.root = RadixNode()

    @staticmethod
    def _match_len(a: List[int], b: List[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], RadixNode]:
        """キャッシュ済みの最長プレフィックスの KV ブロックと、停止したノードを返す。"""
        node, matched, i = self.root, [], 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            m = self._match_len(child.key, tokens[i:])
            node_kv = child.kv_blocks[:m]
            matched.extend(node_kv)
            if m < len(child.key):     # 部分一致、エッジの途中で停止
                return matched, child
            node, i = child, i + m
            node.last_access = time.monotonic()
        return matched, node

    def insert(self, tokens: List[int], kv_blocks: List[int]) -> None:
        """完全な系列を1本挿入する；分岐点で自動的にノードを分裂させる。"""
        node, i = self.root, 0
        while i < len(tokens):
            first = tokens[i]
            child = node.children.get(first)
            if child is None:
                leaf = RadixNode()
                leaf.key = tokens[i:]
                leaf.kv_blocks = kv_blocks[i:]
                leaf.parent = node
                node.children[first] = leaf
                return
            m = self._match_len(child.key, tokens[i:])
            if m < len(child.key):     # エッジ上で分岐 -> 中間ノードを分裂させる
                self._split(node, child, m)
                child = node.children[first]
            node, i = child, i + m

    def _split(self, parent: RadixNode, child: RadixNode, m: int) -> None:
        mid = RadixNode()
        mid.key, mid.kv_blocks = child.key[:m], child.kv_blocks[:m]
        mid.parent = parent
        child.key, child.kv_blocks = child.key[m:], child.kv_blocks[m:]
        child.parent = mid
        mid.children[child.key[0]] = child
        parent.children[mid.key[0]] = mid

    def evict(self, num_blocks: int) -> List[int]:
        """LRU 逐出：葉ノードのみを逐出し、root を保持、回収した KV ブロックを allocator に返す。"""
        freed: List[int] = []
        while len(freed) < num_blocks:
            leaf = self._lru_leaf()
            if leaf is None or leaf is self.root:
                break
            freed.extend(leaf.kv_blocks)
            del leaf.parent.children[leaf.key[0]]
            leaf.parent = None
        return freed

    def _lru_leaf(self) -> Optional[RadixNode]:
        leaves = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.children:
                stack.extend(n.children.values())
            elif n is not self.root:
                leaves.append(n)
        return min(leaves, key=lambda x: x.last_access) if leaves else None
```

**講評**：RadixAttention は「prefix caching」を木に一般化したものである——任意の2つのリクエストはプレフィックスを共有しさえすれば（system prompt に限らない）、`match_prefix` で既存の KV ブロックにヒットし再計算を省ける。`_split` が radix tree の魂である：新しい系列があるエッジの途中で分岐するとき、エッジを「共有部分 ＋ 2つの分岐」に裂かなければならない。逐出戦略はあえて葉ノードのみを動かし root を保持することで、複数リクエストから参照されるホットなプレフィックス（通常は root に近い）が誤って逐出されないことを保証する。SGLang の計測では、複数ターン対話や few-shot のシナリオでヒット率が 50%+ に達し、TTFT が顕著に下がる。

---

### R.2.5　FlashDecoding split-KV reduction

**用途**：FlashDecoding が decode 段階で超長の KV を系列次元に沿って複数ブロックに切って並列計算し、log-sum-exp で部分結果を合併することで、長系列下の GPU 占有率を高める様子を示す。

```python
import numpy as np


def flash_decoding(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   num_splits: int = 4):
    """q: [d], K/V: [seq, d]。KV を seq に沿って num_splits ブロックに切り並列計算し、
       各ブロックが個別に数値的に安定な softmax を行い、最後に log-sum-exp で合併する。"""
    seq, d = K.shape
    scale = 1.0 / np.sqrt(d)
    chunk = (seq + num_splits - 1) // num_splits

    partial_out = []   # 各ブロックの加重 V 累積（ブロック内 sum で割り済み）
    partial_lse = []   # 各ブロックの log-sum-exp (running max 補正を含む)

    # --- 各 split を独立に計算（GPU 上では異なる thread block に対応、並列可能）---
    for s in range(num_splits):
        lo, hi = s * chunk, min((s + 1) * chunk, seq)
        if lo >= hi:
            continue
        scores = (K[lo:hi] @ q) * scale          # [chunk]
        m = scores.max()                          # ブロック内 running max
        p = np.exp(scores - m)                    # 安定化
        l = p.sum()                               # ブロック内の分母
        o = (p @ V[lo:hi]) / l                    # ブロック内の加重平均
        partial_out.append(o)
        partial_lse.append(m + np.log(l))         # log-sum-exp

    # --- reduction：split をまたいで log-sum-exp で重み付けし直して合併 ---
    lses = np.array(partial_lse)
    global_max = lses.max()
    weights = np.exp(lses - global_max)           # 各ブロックの重み (= 各ブロックの真の分母比)
    weights /= weights.sum()
    out = sum(w * o for w, o in zip(weights, partial_out))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, seq = 128, 8192
    q = rng.standard_normal(d)
    K = rng.standard_normal((seq, d))
    V = rng.standard_normal((seq, d))

    # 単一ブロックの reference softmax と照合し、数値の一致を検証
    ref_scores = (K @ q) / np.sqrt(d)
    ref = (np.exp(ref_scores - ref_scores.max()) @ V) / \
          np.exp(ref_scores - ref_scores.max()).sum()
    fd = flash_decoding(q, K, V, num_splits=8)
    print("max abs diff:", np.abs(ref - fd).max())   # ~1e-15
```

**講評**：decode 段階では batch がしばしば非常に小さく（=1 のことすらある）、単一 query が非常に長い KV と向き合うため、従来の kernel は少数の SM しか働かせられず、GPU が大きく遊ぶ。FlashDecoding は KV の seq 次元にもう一刀入れ、複数の SM が並列で別々の区間を走査できるようにし、最後に log-sum-exp で各区間の部分 softmax を正しく合併する——鍵は各区間が自身の `m`（running max）と `lse` を持ち、合併時に `exp(lse_i - global_max)` で重み付けし直すことにあり、数学的には単一ブロックの softmax と完全に等価である（上例では誤差 ~1e-15）。長 context 推論の decode スループット向上の多くはここから来る。

---

### R.2.6　KV Cache 量子化（FP8 / INT8 per-token）

**用途**：KV Cache への書き込み前に K/V を per-token 量子化（INT8 または FP8）し、読み取り時に逆量子化することで、精度損失を極小に抑えつつ cache の VRAM と帯域を半減する。

```python
import numpy as np


def quantize_kv_int8(x: np.ndarray):
    """per-token 対称量子化：x [num_tokens, head_dim] -> int8 + per-token scale。
       各 token が自身の absmax で scale を求め、外れ値 token が全体を引きずるのを避ける。"""
    absmax = np.abs(x).max(axis=-1, keepdims=True)     # [num_tokens, 1]
    scale = absmax / 127.0
    scale = np.maximum(scale, 1e-8)                    # ゼロ除算防止
    q = np.round(x / scale).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_kv_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """読み取り時に fp16/fp32 に戻し、attention 計算に供する。"""
    return q.astype(np.float32) * scale


def quantize_kv_fp8(x: np.ndarray):
    """FP8 (E4M3) per-token：scale で数値を E4M3 の動的範囲(±448)に縮めて格納する。
       ここでは numpy で模擬し、実際は Triton/cuda が torch.float8_e4m3fn に書き込む。"""
    FP8_MAX = 448.0
    absmax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(absmax / FP8_MAX, 1e-8)
    scaled = np.clip(x / scale, -FP8_MAX, FP8_MAX)
    # E4M3 の有限な仮数を模擬：round-to-nearest で仮数ビットを量子化
    q = _round_to_e4m3(scaled)
    return q, scale.astype(np.float32)


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    exp = np.floor(np.log2(np.maximum(ax, 1e-12)))
    mant_step = np.power(2.0, exp - 3)        # E4M3：仮数3ビット
    return sign * np.round(ax / mant_step) * mant_step


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = rng.standard_normal((16, 128)).astype(np.float32)

    q, s = quantize_kv_int8(k)
    err = np.abs(k - dequantize_kv_int8(q, s)).mean()
    print("INT8 mean abs err:", err)          # ~1e-3、VRAM 半減
```

**講評**：KV Cache 量子化は「わずかな精度で VRAM と帯域を稼ぐ」コスパの高い手段である——INT8 はそのまま半減、FP8（E4M3）は Hopper/Ada でネイティブなハードウェアサポートがある。per-token 量子化（token ごとに1つの scale）は per-tensor よりずっと頑健である。attention では少数の外れ値 token の数値が特に大きいことが多く、scale を共有すると他の token の量子化解像度が犠牲になるからだ。実務では K のほうが V より量子化に敏感で（K は softmax に入ると指数的に増幅される）、そのため「K は FP8、V は INT8」や K に高めの精度を残す混合戦略がよく見られる。

---

### R.2.7　Cache-aware routing（プレフィックスハッシュの一貫性ルーティング）

**用途**：複数ノードの推論クラスタで、リクエストのプレフィックス（system prompt + 履歴）からハッシュを計算し、一貫性ハッシュでノードを選ぶことで、同じプレフィックスのリクエストを同一ノードへ安定して振り分け、RadixAttention / prefix cache のヒットを最大化する。

```python
import bisect
import mmh3   # MurmurHash3、高速かつ分布が均一


class ConsistentHashRouter:
    """一貫性ハッシュリング + 仮想ノード、ノードの増減による大規模な cache 失効を避ける。"""
    def __init__(self, nodes, vnodes: int = 160):
        self.ring = {}           # hash -> node
        self.sorted_keys = []
        self.vnodes = vnodes
        for n in nodes:
            self.add_node(n)

    def _hash(self, key: str) -> int:
        # 32-bit 無符号ハッシュをリング上の座標とする
        return mmh3.hash(key, signed=False)

    def add_node(self, node: str) -> None:
        for v in range(self.vnodes):
            h = self._hash(f"{node}#{v}")
            self.ring[h] = node
            bisect.insort(self.sorted_keys, h)

    def remove_node(self, node: str) -> None:
        for v in range(self.vnodes):
            h = self._hash(f"{node}#{v}")
            self.ring.pop(h, None)
            idx = bisect.bisect_left(self.sorted_keys, h)
            if idx < len(self.sorted_keys) and self.sorted_keys[idx] == h:
                self.sorted_keys.pop(idx)

    def route(self, system_prompt: str, history: str) -> str:
        """プレフィックスハッシュでノードを選ぶ：同じプレフィックスは常にリング上の同じ位置 -> 同じノードに落ちる。"""
        prefix_key = system_prompt + "\x00" + history
        h = self._hash(prefix_key)
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


if __name__ == "__main__":
    router = ConsistentHashRouter(["gpu-0", "gpu-1", "gpu-2", "gpu-3"])
    sys_p = "You are a helpful assistant."
    print(router.route(sys_p, "user: hi"))      # 同じプレフィックスは安定して同じノードにヒット
    print(router.route(sys_p, "user: hi"))       # 上の行と同じノード
```

**講評**：prefix cache は「同じプレフィックスが同一ノードに当たる」ときにのみ意味を持ち、さもなければ各ノードがそれぞれ KV を計算してヒット率はゼロになる。一貫性ハッシュ（仮想ノードを含む）の価値は、ノードの増減時に再マッピングが必要な key が `1/N` だけで済み、全件を再シャッフルしない点にある——1台の機器が上下線するだけでクラスタ全体の cache が無効化されるのを避ける。`mmh3.hash` は Python 組み込みの `hash` より速く、プロセスをまたいで安定している（組み込み hash には randomization がある）。実務では SGLang の router は純粋なプレフィックスハッシュに加え、負荷分散の因子（cache ヒット vs. ノード負荷の折衷）を混ぜ、人気のプレフィックスが単一ノードを潰すのを避ける。

---

### R.2.8　意味キャッシュ（GPTCache-style）

**用途**：GPTCache 風にクエリを embed してベクトルデータベースに格納し、新しいクエリは ANN 検索を経て意味的に十分近ければ（cosine 類似度 > 閾値）キャッシュ済みの答えを直接返し、1回の LLM 推論を省く。

```python
import numpy as np
# 任意の embedding モデル + 任意のベクトルDB（ここでは Qdrant を例とするが Milvus でも可）
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class SemanticCache:
    def __init__(self, embed_fn, collection="llm_cache",
                 dim=768, threshold=0.98):
        self.embed_fn = embed_fn          # query -> np.ndarray[dim]
        self.threshold = threshold        # cosine の閾値、0.98 はやや保守的で誤ヒットを避ける
        self.collection = collection
        self.client = QdrantClient(":memory:")   # 本番環境では Qdrant/Milvus サービスに接続
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._next_id = 0

    def get(self, query: str):
        vec = self.embed_fn(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vec.tolist(),
            limit=1,
        )
        if hits and hits[0].score >= self.threshold:   # COSINE 距離はすでに類似度
            return hits[0].payload["answer"], hits[0].score
        return None, (hits[0].score if hits else 0.0)

    def put(self, query: str, answer: str) -> None:
        vec = self.embed_fn(query)
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._next_id,
                vector=vec.tolist(),
                payload={"query": query, "answer": answer},
            )],
        )
        self._next_id += 1

    def ask(self, query: str, llm_call) -> str:
        cached, score = self.get(query)
        if cached is not None:
            return cached                  # ヒット：LLM 推論を1回省く
        answer = llm_call(query)           # ミス：実際に LLM を叩く
        self.put(query, answer)
        return answer


if __name__ == "__main__":
    def fake_embed(q: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(q)) % (2**32))
        v = rng.standard_normal(768)
        return (v / np.linalg.norm(v)).astype(np.float32)

    cache = SemanticCache(fake_embed)
    print(cache.ask("What is the capital of France?",
                    lambda q: "Paris"))    # ミス -> LLM を呼ぶ
    # 意味的に近い言い回しは閾値設定の下でヒットしうる（ここではランダム embed で流れを示すのみ）
```

**講評**：意味キャッシュは前述の KV/プレフィックスキャッシュとは別の階層に属する——KV を再利用するのではなく「最終的な答え」を直接再利用し、ヒットすれば推論まるごと1回を省けるため、コスパが最も高いがリスクも最も高い。`threshold` が中核のつまみである：低すぎると（0.9 など）「フランスの首都」と「ドイツの首都」を同義と誤判定して誤った答えを返す。0.98 はやや保守的で、FAQ / 文書 QA のような確定性の高いシナリオに向くが、細部を正確に区別する必要のある対話には向かない。本番デプロイでは embedding モデルを業務コーパスと整合させ、ベクトルDB（Milvus / Qdrant）に TTL と容量に基づく淘汰を設定し、ヒット結果のサンプリング品質監視を行ってキャッシュ汚染（cache poisoning）を避けること。


---

## R.3　量子化・推論デプロイ・評価の中核実装

本節では「訓練後量子化」から「サービス投入」「自動評価」までの鍵となる実装をまとめる。すべてのコードはオープンソースのウェイトモデルとオープンライセンスのデータを前提とし、そのままエンジニアリングの参考にできる。各断片に1行の用途説明と簡潔な**講評**を付す。

---

### R.3.1　QLoRA ファインチューニング設定

> 用途：4-bit NF4 量子化でベースモデルをロードし、低ランクの LoRA adapter のみを訓練して、単一カードで大型モデルのファインチューニングを完了する。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit NF4 二重量子化：重みの VRAM 占有を顕著に下げ、計算時に bf16 に逆量子化する
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 勾配チェックポイント有効化 + k-bit 訓練の前処理（layernorm を fp32 に変換、入力勾配を有効化）
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,                       # alpha/r = 2、等価スケーリング係数が安定
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                       # attention と MLP の両投影群を同時にカバー
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()         # 訓練可能パラメータは通常 < 2%
```

**講評**：`target_modules` を q/k/v/o と gate/up/down の両方に掛けるのは、QLoRA 論文の「LoRA をすべての線形層に加える」の実務への落とし込みであり、q/v だけに掛けるより収束が安定する。`gradient_checkpointing` と NF4 は単一カードで 7B–13B を訓練する鍵である。注意点として、`prepare_model_for_kbit_training` の後に `get_peft_model` を呼ぶこと、また訓練時に `gradient_checkpointing_kwargs={"use_reentrant": False}` を設定して新版 PyTorch と互換にすること。

---

### R.3.2　GPTQ / AWQ 訓練後量子化

> 用途：少量のキャリブレーションコーパスで 4-bit の重み量子化を行い、そのまま推論できる量子化ウェイトを出力する。

```python
# --- GPTQ（gptqmodel、autogptq の後継メンテ版）---
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

calib = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib = [next(iter(calib))["text"] for _ in range(256)]   # 128~512 件のキャリブレーション標本で十分

quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True)
model = GPTQModel.load("meta-llama/Llama-3.1-8B", quant_cfg)
model.quantize(calib)
model.save("./Llama-3.1-8B-gptq-int4")
```

```python
# --- AWQ（autoawq）：activation-aware、チャネルごとに重要な重みを保護 ---
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
model.quantize(tok, quant_config=quant_config)        # 組み込みのデフォルトキャリブレーションセット、calib_data を渡すことも可
model.save_quantized("./Llama-3.1-8B-awq-int4")
tok.save_pretrained("./Llama-3.1-8B-awq-int4")
```

**講評**：GPTQ は層ごとの Hessian で再構成誤差を最小化し、`desc_act=True`（act-order）はたいてい perplexity をさらに下げられるが量子化はやや遅い。AWQ は重みを更新せずスケーリング保護のみを行い、量子化が速く指示モデルにやさしい。`group_size=128` は精度/サイズの一般的な折衷である。両者の出力とも vLLM で直接ロードできる（`--quantization gptq_marlin` / `awq_marlin`）。

---

### R.3.3　GGUF 変換と k-quant

> 用途：HF のウェイトを llama.cpp の GGUF 形式に変換し、さらに k-quant 圧縮を行って CPU/エッジ推論に供する。

```bash
# 1) HF -> GGUF（まず fp16/bf16 全精度に変換し、それから量子化）
python convert_hf_to_gguf.py ./Llama-3.1-8B \
    --outfile model-f16.gguf \
    --outtype f16

# 2) k-quant：Q4_K_M は品質/サイズの最もよく使うバランス点
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# 3)（任意）importance matrix で低ビットの品質を上げる
./llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ4_XS.gguf IQ4_XS
```

**講評**：`Q4_K_M` は多くの 7B–70B にとってサイズと精度のスイートスポットである。より小さくしたいなら `Q3_K_M`、より原型に近づけたいなら `Q5_K_M` または `Q6_K` を使う。4-bit 未満（`IQ4_XS`/`IQ3` など）は必ず imatrix と組み合わせること。さもないと劣化が顕著になる。変換の失敗はたいてい chat template か新アーキテクチャ未対応が原因で、llama.cpp を更新する必要がある。

---

### R.3.4　投機的デコーディング（Speculative Decoding）

> 用途：小さなドラフトモデルで一度に複数の token を提案し、大型モデルが並列で検証することで、出力分布を変えずに遅延を下げる。

```python
# vLLM：draft モデルによる投機的デコーディング（新版は speculative_config で渡す）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",  # 同じ tokenizer ファミリーの小型モデル
        "num_speculative_tokens": 5,
    },
    tensor_parallel_size=4,
)
out = llm.generate(["解釋投機解碼的原理。"], SamplingParams(temperature=0.0, max_tokens=256))
```

```bash
# llama.cpp：メインモデル + ドラフトモデル
./llama-server \
    -m  model-70B-Q4_K_M.gguf \
    -md model-1B-Q4_K_M.gguf \
    --draft-max 8 --draft-min 1 \
    -ngl 99 -c 8192
```

**講評**：ドラフトモデルはメインモデルと vocab を共有する必要がある（同じファミリーが最も安定）。高速化率は「受理率」次第で、低温・構造化・反復性の高い出力で効果が最大になる。高温の創造的生成では受理率が低く、割に合わないこともある。`num_speculative_tokens` が大きすぎるとかえって検証の演算力を浪費する。5–8 がよくある出発点である。

---

### R.3.5　vLLM サービス起動

> 用途：OpenAI 互換 API で高スループットの推論サービスを起動し、VRAM と長コンテキストを調整する。

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name llama3.1-8b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --max-num-seqs 256
# 以降は OpenAI SDK を http://localhost:8000/v1 に向ければよい
```

**講評**：`--gpu-memory-utilization` は KV cache の予約量を決める。0.85–0.92 が同一カード上の他の負荷次第の目安で、高すぎると OOM になりやすい。`--enable-chunked-prefill` は長い prompt と decode を交錯させ、TTFT のジッタを下げる。`--kv-cache-dtype fp8` は KV の VRAM を約半分節約し、極小の精度損失と引き換えにする、並行数やコンテキストを拡大するコスパの高い手段である。`--max-model-len` はウェイトが実際にサポートする長さを決して超えないこと。

---

### R.3.6　distilabel による合成蒸留データ

> 用途：「オープンウェイト」の教師モデルで指示への応答をバッチ生成し、SFT に使える蒸留データセットを作る。

```python
# 教師はオープンウェイトモデル（Llama-3.3-70B）で、ローカルの vLLM サービスから提供 —— コンプライアンスに適合、商用ライセンスの範囲内
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.models import OpenAILLM   # 自前 vLLM の OpenAI 互換エンドポイントを指す

teacher = OpenAILLM(
    model="llama3.3-70b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
)

with Pipeline(name="distill-sft") as pipeline:
    load = LoadDataFromHub(repo_id="HuggingFaceH4/instruction-dataset", split="test")
    gen = TextGeneration(llm=teacher, input_batch_size=16)   # instruction 列から generation を生成
    keep = KeepColumns(columns=["instruction", "generation"])
    load >> gen >> keep

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    distiset.push_to_hub("your-org/distilled-sft")
```

**講評**：教師は必ず**オープンウェイトか、ライセンスが蒸留を許可する**モデル（Llama / Qwen / Mistral のオープンソース版など）でなければならず、保護された商用 API をスクレイピングしたり迂回したりしてはならず、脱検閲に使ってもならない。`use_cache=True` は中断後の続行を可能にし演算力を節約する。出力はさらに品質/重複除去のフィルタ（distilabel の evol/quality スコアリングステップなど）を一段通してから SFT に入れることを推奨する。

---

### R.3.7　lm-eval-harness による評価と長コンテキスト

> 用途：業界標準の harness で下流ベンチマークを走らせ、量子化モデルの品質回帰を測る。

```bash
# 標準的な学科と数学推論のベンチマーク
lm_eval --model hf \
    --model_args pretrained=./Llama-3.1-8B-awq-int4,dtype=bfloat16 \
    --tasks mmlu,gsm8k,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./eval_results

# vLLM サービスを直接評価（スループットが高い）
lm_eval --model local-completions \
    --model_args base_url=http://localhost:8000/v1/completions,model=llama3.1-8b \
    --tasks gsm8k --batch_size 16
```

**長コンテキスト補足**：MMLU/GSM8K は長文の検索能力を反映できないため、別途 **RULER**（多様な合成タスク、4k–128k の複数長を指定可能）と **NIAH（Needle-in-a-Haystack）**「干し草の山から針を探す」テストを走らせ、コンテキスト増大に伴う再現率の減衰曲線を観察する必要がある。KV cache 量子化（R.3.5 の fp8 など）や RoPE 外挿設定は、この2項目で必ず回帰検証すること。

**講評**：量子化モデルを投入する前に、少なくとも GSM8K（推論）と MMLU（知識）で原モデルと比較すること。4-bit は通常 1–3 点下がる程度なら許容範囲である。大幅に劣化する場合はたいてい量子化のキャリブレーションセットか group_size が不適切である。`--num_fewshot` とバージョンを固定して初めて、スコアを回をまたいで比較できる。

---

### R.3.8　Ollama Modelfile によるパッケージ化

> 用途：GGUF のウェイトを推論パラメータと対話テンプレートごと、ワンクリックで配布できる Ollama モデルにパッケージ化する。

```text
FROM ./model-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER stop "<|eot_id|>"

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

SYSTEM """你是一個樂於助人、回答精確的中文助理。"""
```

```bash
ollama create my-llama3 -f Modelfile
ollama run my-llama3 "用一句話解釋投機解碼"
```

**講評**：`TEMPLATE` はそのベースの chat format（ここでは Llama-3 の header マーカー）と一致しなければならず、誤ったテンプレートはモデルに見当違いの答えをさせたり止まらなくさせたりする。`stop` と `num_ctx` は訓練/量子化時と一致させること。Modelfile は「ウェイト ＋ パラメータ ＋ テンプレート」を単一のバージョン管理可能な成果物にし、チームでの再現に大いに役立つ。

---

### R.3.9　二重モデルの動的ルーティング

> 用途：複雑度のヒューリスティックで、簡単なリクエストは小型モデルへ、難しいリクエストは大型モデルへ振り分け、コストと品質のバランスを取る。

```python
# 軽量ルーティング：長さ/キーワードで複雑度を見積もり、安価または強力なモデルへ分流（OpenAI 互換エンドポイントなら何でも可）
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re

app = FastAPI()
cheap = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # 小型モデル vLLM
big   = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")   # 大型モデル vLLM

HARD_HINTS = re.compile(r"(證明|推導|程式|debug|逐步|多步|規劃|數學|code|prove)", re.I)

class Query(BaseModel):
    prompt: str

def complexity_score(text: str) -> float:
    score = 0.0
    score += min(len(text) / 800, 1.0)          # 長さのシグナル
    score += 0.5 if HARD_HINTS.search(text) else 0.0  # タスク型キーワード
    score += 0.3 if text.count("\n") >= 4 else 0.0    # 複数段/構造化された入力
    return score

@app.post("/route")
def route(q: Query):
    hard = complexity_score(q.prompt) >= 0.8
    client, model = (big, "llama3.1-70b") if hard else (cheap, "llama3.1-8b")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": q.prompt}],
        temperature=0.3,
    )
    return {"model": model, "routed_as": "hard" if hard else "easy",
            "content": resp.choices[0].message.content}
```

**講評**：ヒューリスティックルーティングは追加遅延ゼロで解釈しやすく、投入初期の最良の出発点である。トラフィックが貯まれば「小型モデルで採点」や専用の router モデル（RouteLLM など）による学習式ルーティングに置き換えられる。必ず**フォールバック**を設定すること——大型モデルがタイムアウトや 5xx のときは小型モデルへ降格し、ルーティング判断を記録して誤判定率をオフラインで突き合わせられるようにする。LiteLLM を使えば、上記の2エンドポイントをその Router に掛け、組み込みの負荷分散と fallback の管理に委ねられる。


---
