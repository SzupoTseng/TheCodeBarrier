# 附錄R　核心實作源碼參考

> 本附錄把全書關鍵技術的**參考實作**集中起來——蒸餾損失、KV Cache/PagedAttention、量化與部署。程式碼以可讀、可改、貼近真實函式庫為目標（PyTorch / transformers / vLLM / llama.cpp / peft / distilabel），每段附**目標**與**點評**。所有教師模型一律假設為**你有權使用的對象**（開放權重或你自有模型）；本附錄**不含**任何規避防護、反指紋、去審查（abliteration）程式碼——那條紅線見附錄 E 與第 6 章。

> ⚠️ 這些是**參考骨架**，非生產級複製貼上即用：實際部署需自行補上錯誤處理、形狀檢查、分散式同步與版本相容。把它們當成「看懂原理後的起手式」。

---

## R.1　知識蒸餾核心實作

本節彙整知識蒸餾（Knowledge Distillation, KD）在大型語言模型訓練中的核心程式碼骨架，皆以 PyTorch 與 Hugging Face `transformers` 生態為基礎。所有教師模型均假設為你具有合法使用權的開源或自有權重模型（例如以自家大模型蒸餾出小模型）。程式碼以可讀、可直接移植為目標，實務上需依資料管線與裝置數量微調。

---

### R.1.1　Hinton 軟硬標籤 KD 損失

最經典的 KD 損失：以溫度 `T` 軟化教師與學生的機率分布，用 KL 散度逼近教師「暗知識」，再以 `α` 與真實標籤的交叉熵混合。注意軟標籤梯度因 softmax 對 `1/T` 的縮放會被壓低，需乘回 `T²` 以維持與硬標籤同量級。

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5,
            ignore_index=-100):
    """Hinton (2015) soft + hard label distillation.

    student_logits, teacher_logits: (B, V) or (B, L, V)
    labels: (B,) or (B, L) ground-truth token ids
    """
    # 軟標籤：教師不參與反傳，需 detach
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)  # 補回因 1/T 縮放被壓低的梯度量級

    # 硬標籤：標準交叉熵（自動忽略 padding）
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

**點評**：`reduction="batchmean"` 才是數學定義上正確的 KL 平均（`"mean"` 會多除以詞表維度而失真）；`T*T` 縮放與教師 `detach()` 是兩個最常被遺漏的細節。

---

### R.1.2　Forward 與 Reverse KL（mode-covering vs mode-seeking）

KD 的方向性決定學生如何「妥協」。Forward KL `KL(p_teacher ‖ p_student)` 是 mode-covering：學生被迫覆蓋教師所有支撐集，對小模型容易學出過度平滑、含幻覺的分布。Reverse KL `KL(p_student ‖ p_teacher)` 是 mode-seeking：學生聚焦教師高機率模式，產生更銳利、更穩定的輸出，這是 MiniLLM 與 GKD 在 LLM 蒸餾上偏好 reverse KL 的原因。

```python
import torch.nn.functional as F

def forward_kl(student_logits, teacher_logits, T=1.0):
    """KL(teacher || student) — mode-covering，學生覆蓋教師全分布。"""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def reverse_kl(student_logits, teacher_logits, T=1.0):
    """KL(student || teacher) — mode-seeking，學生聚焦教師主峰。"""
    # 注意：第一引數須為 log-prob，故對 teacher 取 log_softmax；
    # target 為 student 機率，但須允許其反傳，故不 detach student。
    p_student = F.softmax(student_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits.detach() / T, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean() * (T * T)
```

**點評**：生成式 LLM 蒸餾通常選 reverse KL（或 GKD 的 Jensen-Shannon 廣義插值），避免 forward KL 逼學生「什麼都學一點」而產生流暢但不可靠的長尾；分類或詞表受限任務則 forward KL 仍夠用。

---

### R.1.3　序列級 KD／on-policy 蒸餾 loop 骨架

序列級 KD（GKD/on-policy distillation）的關鍵：訓練資料由**學生自己採樣**，再用教師對學生產生的序列打分，從而消除 teacher-forcing 下的曝光偏差（exposure bias）。以下為單訓練步驟骨架。

```python
import torch

@torch.no_grad()
def sample_from_student(student, tokenizer, prompts, max_new_tokens=256):
    """On-policy：由學生自身生成序列（之後交給教師評分）。"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(student.device)
    gen = student.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=1.0, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].size(1)
    return gen, prompt_len  # gen: (B, prompt_len + new_len)

def on_policy_step(student, teacher, tokenizer, prompts, kl_fn, optimizer, T=1.0):
    # 1) 學生自採樣（on-policy data）
    seqs, prompt_len = sample_from_student(student, tokenizer, prompts)
    attn = (seqs != tokenizer.pad_token_id).long()

    # 2) 學生對自己序列前傳（需梯度）
    s_logits = student(seqs, attention_mask=attn).logits  # (B, L, V)

    # 3) 教師對同序列評分（凍結、no_grad）
    with torch.no_grad():
        t_logits = teacher(seqs, attention_mask=attn).logits

    # 4) 僅對生成段（response）計 loss，shift 對齊 next-token
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

**點評**：on-policy 採樣成本高，實務常以 `lambda`（GKD 的 `lmbda`）機率混合 on-policy 與固定資料；`trl.GKDTrainer` 已封裝此邏輯，自寫骨架主要用於理解與客製。

---

### R.1.4　CoT 加權損失（`<think>...</think>` 內 token 加權）

蒸餾推理鏈（chain-of-thought）時，思考段往往比最終答案更承載教師的推理能力。以下依序列中 `<think>` 與 `</think>` 標記建立逐 token 權重遮罩，對思考段給予較高權重。

```python
import torch

def build_cot_weight_mask(input_ids, tokenizer, think_weight=2.0, base_weight=1.0,
                          open_tag="<think>", close_tag="</think>"):
    """為每個 token 產生權重；<think>...</think> 區間內權重較高。"""
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
    """per_token_loss/(weight_mask/pad_mask): (B, L)；回傳純量。"""
    w = weight_mask * pad_mask  # padding 權重歸零
    return (per_token_loss * w).sum() / w.sum().clamp_min(1.0)
```

**點評**：`per_token_loss` 可由 `F.cross_entropy(..., reduction="none")` 或逐 token KL 得到；標籤切分若用特殊 token（建議將 `<think>`/`</think>` 加入 tokenizer special tokens），可避免被 BPE 拆成多個子詞而漏抓邊界。

---

### R.1.5　跨詞表對齊（cross-tokenizer logit projection）

當教師與學生使用不同 tokenizer 時，logit 無法逐位對齊。常見作法：取教師 top-k token，解碼成字串，用學生 tokenizer 重新編碼，再把教師機率質量重分配到學生詞表上對應的（首）token。以下為示意骨架，省略了完整的子詞合併策略。

```python
import torch

@torch.no_grad()
def project_teacher_to_student(teacher_logits, teacher_tok, student_tok,
                               student_vocab_size, top_k=64, T=1.0):
    """將教師單步分布投影到學生詞表（top-k 字串重編碼 + 質量重分配）。

    teacher_logits: (V_t,) 單一位置的教師 logits
    回傳: (V_s,) 學生詞表上的目標機率分布
    """
    probs = torch.softmax(teacher_logits / T, dim=-1)
    topk = torch.topk(probs, k=top_k)
    student_dist = torch.zeros(student_vocab_size, device=teacher_logits.device)

    for tok_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
        # 教師 token -> 字串 -> 學生 tokenizer 重新編碼
        piece = teacher_tok.decode([tok_id]).strip()
        if not piece:
            continue
        s_ids = student_tok.encode(piece, add_special_tokens=False)
        if not s_ids:
            continue
        # 質量歸給學生編碼的首 token（簡化策略；可改為沿子詞鏈分攤）
        student_dist[s_ids[0]] += p

    total = student_dist.sum()
    if total > 0:
        student_dist /= total  # 重新歸一化，丟棄無法對齊的尾部
    return student_dist
```

**點評**：此為示意而非生產級——嚴謹作法需處理多子詞 token 的機率沿鏈分攤、空白前綴（`▁`/`Ġ`）差異與歸一化偏差；ULD（Universal Logit Distillation）以最佳傳輸（optimal transport）對齊兩詞表，可作為更穩健的替代。

---

### R.1.6　完整訓練步驟（HF Trainer-style）

整合上述模組為自訂 `Trainer`：教師凍結並設 `eval()`，學生可搭配 PEFT LoRA 降低顯存；分散式以 `accelerate` 設定檔啟用 DeepSpeed ZeRO-3 或 FSDP，並開啟梯度檢查點。

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
        # 蒸餾損失（R.1.1）：next-token 對齊需 shift
        s_logits = s_out.logits[:, :-1, :].contiguous()
        t_logits = t_out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = kd_loss(s_logits, t_logits, shift_labels,
                       T=self.T, alpha=self.alpha)
        return (loss, s_out) if return_outputs else loss

# --- 組裝 ---
student = AutoModelForCausalLM.from_pretrained("your/small-base",
                                               torch_dtype=torch.bfloat16)
student.gradient_checkpointing_enable()         # 省顯存
student.config.use_cache = False                # 與梯度檢查點互斥
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
    deepspeed="ds_zero3.json",   # 或以 accelerate config 啟用 FSDP
    logging_steps=10, num_train_epochs=3,
)
trainer = DistillTrainer(teacher_model=teacher, T=2.0, alpha=0.5,
                         model=student, args=args,
                         train_dataset=train_ds, tokenizer=tokenizer)
trainer.train()
```

啟動時以 `accelerate launch --config_file fsdp.yaml train.py`，或在 `TrainingArguments` 直接掛 `deepspeed="ds_zero3.json"`（ZeRO-3 將參數、梯度、優化器狀態三者分片至各 GPU）。教師若過大，可單獨以另一進程／量化（如 8-bit）載入並僅做前向評分。

**點評**：教師務必 `eval()` + `requires_grad_(False)` 並包在 `no_grad` 內，否則顯存與算力浪費在不更新的梯度上；`gradient_checkpointing` 與 `use_cache=True` 互斥，需手動關閉 cache 才不會報警告或失效。


---

## R.2　KV Cache 與快取核心實作

本節彙整推論服務中與 KV Cache 相關的核心參考實作，涵蓋從顯存估算、PagedAttention 分頁管理、CUDA 尋址、RadixAttention 前綴共享，到 FlashDecoding、KV 量化與多層快取路由。所有程式碼皆以可讀性為先、語意正確為本，命名與第 10、11 章一致（vLLM / SGLang / FlashAttention / Triton）。

---

### R.2.1　KV Cache 顯存計算器

**用途**：估算一段序列的 KV Cache 佔用位元組，是容量規劃與 batch 上限推導的起點。

```python
def kv_cache_bytes(
    L: int,            # Transformer 層數 (num_layers)
    kv_heads: int,     # KV head 數 (GQA/MQA 後的 num_key_value_heads)
    head_dim: int,     # 每個 head 維度
    seq_len: int,      # 序列長度 (prompt + generated)
    batch: int = 1,    # 並發序列數
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp8/int8 = 1
) -> int:
    """KV Cache 顯存 = 2 (K 與 V 各一份) * dtype_bytes
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

**點評**：每 token 128KB 是 Llama-3-8B 的關鍵常數——`2 * 2 * 32 * 8 * 128 = 131072 = 128KB`。GQA 把 KV head 從 32 砍到 8，等於把 cache 縮小 4 倍；若用 MHA（kv_heads=32）則每 token 暴增到 512KB，batch=128×8K 直接吃滿 512GB。實務上 `2 * dtype_bytes * L * kv_heads * head_dim` 應視為「每 token 常數」先算出來，再乘 `seq_len * batch`，方便快速心算容量天花板。

---

### R.2.2　PagedAttention BlockManager / BlockTable

**用途**：以 vLLM 風格的固定大小區塊（block）管理 KV Cache 顯存，消除連續配置造成的內外部碎片。

```python
from collections import deque
from typing import Dict, List


class PhysicalTokenBlock:
    """一塊固定容納 block_size 個 token 的物理顯存區塊。"""
    def __init__(self, block_number: int, block_size: int):
        self.block_number = block_number   # 物理區塊索引
        self.block_size = block_size
        self.ref_count = 0                 # 被幾個序列共享 (copy-on-write 用)

    def __repr__(self):
        return f"PhysicalTokenBlock(#{self.block_number}, ref={self.ref_count})"


class BlockAllocator:
    """以 free-list 管理物理區塊的配置/釋放，O(1) 配置。"""
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
    """單一序列的 logical block -> physical block 映射。"""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.blocks: List[PhysicalTokenBlock] = []
        self.num_tokens = 0

    def append_token(self) -> None:
        block_size = self.allocator.block_size
        # 最後一塊已滿（或尚無區塊）才需要再配一塊新的物理區塊
        if self.num_tokens % block_size == 0:
            self.blocks.append(self.allocator.allocate())
        self.num_tokens += 1

    def physical_index(self, logical_token_idx: int) -> int:
        """把 logical token 位置翻譯成 (block_number, offset)。"""
        block_size = self.allocator.block_size
        block = self.blocks[logical_token_idx // block_size]
        offset = logical_token_idx % block_size
        return block.block_number, offset

    def fork(self) -> "BlockTable":
        """beam search / parallel sampling 的 copy-on-write 分叉：
           共享既有區塊，僅增加 ref_count。"""
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

**點評**：PagedAttention 的精髓是把 KV Cache 切成 OS 分頁式的固定區塊，`BlockTable` 即「頁表」，`BlockAllocator.free` 即「空閒頁鏈」。`ref_count` 讓 parallel sampling / beam search 共享同一份 prompt 的 KV（copy-on-write），只有在某分支寫入已共享的區塊時才真正複製。`num_tokens % block_size == 0` 是配置新區塊的唯一條件——這是分頁攤平記憶體碎片、把利用率推到 90%+ 的關鍵。當 `num_free == 0` 時必須丟出讓排程器走 preemption（swap-out 或 recompute），不可靜默失敗。

---

### R.2.3　paged_attention CUDA kernel 尋址核心

**用途**：示意 CUDA kernel 內部如何透過 block_table 把 logical token 翻譯成物理顯存指標，再讀取 K/V。

```cuda
// 對單一 query head，掃過某序列的所有 KV token。
// block_table 為該序列的「頁表」：logical block idx -> physical block number。
// kv_cache 佈局: [num_blocks, num_kv_heads, head_dim, block_size]
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    scalar_t*       __restrict__ out,          // [num_heads, head_dim]
    const scalar_t* __restrict__ q,            // [num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,      // 物理 KV 池
    const scalar_t* __restrict__ v_cache,
    const int*      __restrict__ block_table,  // [max_blocks_per_seq]
    const int       context_len,               // 此序列目前 token 數
    const int       num_kv_heads) {

    const int kv_head = blockIdx.y;            // 此 thread block 負責的 KV head
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int logical_block = 0; logical_block < num_blocks; ++logical_block) {
        // 1) 頁表查找：logical block -> 物理 block number
        const int physical_block = block_table[logical_block];

        // 2) 本塊內要處理的 token 數（最後一塊可能不滿）
        const int tokens_in_block =
            min(BLOCK_SIZE, context_len - logical_block * BLOCK_SIZE);

        for (int t = 0; t < tokens_in_block; ++t) {
            // 3) 計算物理位移：定位到 (physical_block, kv_head) 的 head_dim 連續區段
            const int64_t base =
                (((int64_t)physical_block * num_kv_heads + kv_head)
                 * HEAD_DIM) * BLOCK_SIZE;

            const scalar_t* k_ptr = k_cache + base + t;  // K[..., :, t]
            const scalar_t* v_ptr = v_cache + base + t;

            // 4) dot(q, k) -> score，累積到 online-softmax（此處略）
            //    實作中會配合 running max / running sum 做數值穩定的線上 softmax
            //    並把 score * v 累加進 out。
            (void)k_ptr; (void)v_ptr;
        }
    }
}
```

**點評**：相較傳統 attention 假設 KV 在顯存中「連續」，paged kernel 多了一層 `block_table[logical_block]` 的間接尋址——這正是 PagedAttention 能用非連續物理頁拼出邏輯連續序列的代價與精髓。佈局 `[num_blocks, num_kv_heads, head_dim, block_size]` 把同一 head 的連續 token 放在最內維，讓 warp 內 thread 能 coalesced 讀取。實際 vLLM kernel 在內層用 online-softmax（running max + running sum）避免存下完整 score 矩陣；本 sketch 聚焦在「頁表查找 → 物理位移」這條尋址主線。

---

### R.2.4　RadixAttention 前綴樹

**用途**：以 SGLang 風格的 radix tree 共享相同前綴（system prompt、few-shot、多輪歷史）的 KV Cache，達成跨請求自動快取重用。

```python
import time
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """邊上存一段 token 序列，節點掛該前綴對應的 KV cache 指標（區塊清單）。"""
    def __init__(self):
        self.children: Dict[int, "RadixNode"] = {}  # 以首 token 為 key
        self.key: List[int] = []                    # 此節點到父節點的 token 段
        self.kv_blocks: List[int] = []              # 對應 KV 物理區塊 id
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
        """回傳已被快取的最長前綴 KV 區塊，與停在的節點。"""
        node, matched, i = self.root, [], 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            m = self._match_len(child.key, tokens[i:])
            node_kv = child.kv_blocks[:m]
            matched.extend(node_kv)
            if m < len(child.key):     # 部分匹配，停在邊中間
                return matched, child
            node, i = child, i + m
            node.last_access = time.monotonic()
        return matched, node

    def insert(self, tokens: List[int], kv_blocks: List[int]) -> None:
        """插入一條完整序列；自動在分歧點分裂節點。"""
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
            if m < len(child.key):     # 邊上分歧 -> 分裂出中間節點
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
        """LRU 逐出：只逐出葉節點，保留 root，回收的 KV 區塊回傳給 allocator。"""
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

**點評**：RadixAttention 把「prefix caching」一般化成樹——任意兩個請求只要共享前綴（不限於 system prompt），就能在 `match_prefix` 命中既有 KV 區塊而免去重算。`_split` 是 radix tree 的靈魂：當新序列在某條邊中途分歧時，必須把邊裂成「共享段 + 兩個分支」。逐出策略刻意只動葉節點並保留 root，確保被多請求引用的熱前綴（通常靠近 root）不會被誤逐。SGLang 量測顯示，在多輪對話與 few-shot 場景下命中率可達 50%+，TTFT 顯著下降。

---

### R.2.5　FlashDecoding split-KV reduction

**用途**：示意 FlashDecoding 如何在 decode 階段把超長 KV 沿序列維切成多塊平行計算，再用 log-sum-exp 合併部分結果，提升長序列下的 GPU 佔用率。

```python
import numpy as np


def flash_decoding(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   num_splits: int = 4):
    """q: [d], K/V: [seq, d]。把 KV 沿 seq 切 num_splits 塊平行算，
       每塊各自做數值穩定 softmax，最後用 log-sum-exp 合併。"""
    seq, d = K.shape
    scale = 1.0 / np.sqrt(d)
    chunk = (seq + num_splits - 1) // num_splits

    partial_out = []   # 每塊的加權 V 累積（已除以塊內 sum）
    partial_lse = []   # 每塊的 log-sum-exp (含 running max 修正)

    # --- 各 split 獨立計算（GPU 上對應不同 thread block，可平行）---
    for s in range(num_splits):
        lo, hi = s * chunk, min((s + 1) * chunk, seq)
        if lo >= hi:
            continue
        scores = (K[lo:hi] @ q) * scale          # [chunk]
        m = scores.max()                          # 塊內 running max
        p = np.exp(scores - m)                    # 穩定化
        l = p.sum()                               # 塊內分母
        o = (p @ V[lo:hi]) / l                    # 塊內加權平均
        partial_out.append(o)
        partial_lse.append(m + np.log(l))         # log-sum-exp

    # --- reduction：跨 split 用 log-sum-exp 重新加權合併 ---
    lses = np.array(partial_lse)
    global_max = lses.max()
    weights = np.exp(lses - global_max)           # 各塊權重 (= 各塊真實分母比例)
    weights /= weights.sum()
    out = sum(w * o for w, o in zip(weights, partial_out))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, seq = 128, 8192
    q = rng.standard_normal(d)
    K = rng.standard_normal((seq, d))
    V = rng.standard_normal((seq, d))

    # 與單塊 reference softmax 對照，驗證數值一致
    ref_scores = (K @ q) / np.sqrt(d)
    ref = (np.exp(ref_scores - ref_scores.max()) @ V) / \
          np.exp(ref_scores - ref_scores.max()).sum()
    fd = flash_decoding(q, K, V, num_splits=8)
    print("max abs diff:", np.abs(ref - fd).max())   # ~1e-15
```

**點評**：decode 階段 batch 常常很小（甚至 =1），單一 query 對上很長的 KV，傳統 kernel 只能讓少數 SM 工作、GPU 嚴重閒置。FlashDecoding 沿 KV 的 seq 維再切一刀，讓多個 SM 平行掃不同段，最後用 log-sum-exp 把各段的部分 softmax 正確合併——關鍵在於每段都帶著自己的 `m`（running max）與 `lse`，合併時以 `exp(lse_i - global_max)` 重新加權，數學上與單塊 softmax 完全等價（上例誤差 ~1e-15）。長 context 推論的 decode 吞吐提升多來自此。

---

### R.2.6　KV Cache 量化（FP8 / INT8 per-token）

**用途**：在寫入 KV Cache 前對 K/V 做 per-token 量化（INT8 或 FP8），讀取時反量化，於精度損失極小下把 cache 顯存與頻寬砍半。

```python
import numpy as np


def quantize_kv_int8(x: np.ndarray):
    """per-token 對稱量化：x [num_tokens, head_dim] -> int8 + per-token scale。
       每個 token 用自己的 absmax 求 scale，避免離群 token 拖垮整體。"""
    absmax = np.abs(x).max(axis=-1, keepdims=True)     # [num_tokens, 1]
    scale = absmax / 127.0
    scale = np.maximum(scale, 1e-8)                    # 防除零
    q = np.round(x / scale).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_kv_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """讀取時還原回 fp16/fp32 供 attention 計算。"""
    return q.astype(np.float32) * scale


def quantize_kv_fp8(x: np.ndarray):
    """FP8 (E4M3) per-token：以 scale 把數值縮放進 E4M3 動態範圍(±448)再轉存。
       這裡用 numpy 模擬，實際由 Triton/cuda 寫入 torch.float8_e4m3fn。"""
    FP8_MAX = 448.0
    absmax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(absmax / FP8_MAX, 1e-8)
    scaled = np.clip(x / scale, -FP8_MAX, FP8_MAX)
    # 模擬 E4M3 的有限尾數：以 round-to-nearest 量化尾數位
    q = _round_to_e4m3(scaled)
    return q, scale.astype(np.float32)


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    exp = np.floor(np.log2(np.maximum(ax, 1e-12)))
    mant_step = np.power(2.0, exp - 3)        # E4M3：3 個尾數位
    return sign * np.round(ax / mant_step) * mant_step


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = rng.standard_normal((16, 128)).astype(np.float32)

    q, s = quantize_kv_int8(k)
    err = np.abs(k - dequantize_kv_int8(q, s)).mean()
    print("INT8 mean abs err:", err)          # ~1e-3, 顯存減半
```

**點評**：KV Cache 量化是「以少量精度換顯存與頻寬」的高性價比手段——INT8 直接砍半、FP8（E4M3）在 Hopper/Ada 上更有原生硬體支援。per-token 量化（每個 token 一個 scale）比 per-tensor 穩健得多，因為 attention 中常有少數離群 token 的數值特別大，共用一個 scale 會讓其他 token 量化解析度被犧牲。實務上 K 比 V 對量化更敏感（K 進 softmax 會被指數放大），因此常見「K 用 FP8、V 用 INT8」或對 K 保留較高精度的混合策略。

---

### R.2.7　Cache-aware routing（前綴雜湊一致性路由）

**用途**：在多節點推論叢集中，依請求前綴（system prompt + 歷史）算雜湊，用一致性雜湊選節點，把相同前綴的請求穩定路由到同一節點以最大化 RadixAttention / prefix cache 命中。

```python
import bisect
import mmh3   # MurmurHash3，快速且分佈均勻


class ConsistentHashRouter:
    """一致性雜湊環 + 虛擬節點，避免節點增減造成大規模 cache 失效。"""
    def __init__(self, nodes, vnodes: int = 160):
        self.ring = {}           # hash -> node
        self.sorted_keys = []
        self.vnodes = vnodes
        for n in nodes:
            self.add_node(n)

    def _hash(self, key: str) -> int:
        # 取 32-bit 無號雜湊作為環上座標
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
        """以前綴雜湊選節點：相同前綴永遠落在環上同一位置 -> 同節點。"""
        prefix_key = system_prompt + "\x00" + history
        h = self._hash(prefix_key)
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


if __name__ == "__main__":
    router = ConsistentHashRouter(["gpu-0", "gpu-1", "gpu-2", "gpu-3"])
    sys_p = "You are a helpful assistant."
    print(router.route(sys_p, "user: hi"))      # 同前綴穩定命中同節點
    print(router.route(sys_p, "user: hi"))       # 與上行同節點
```

**點評**：prefix cache 只有在「相同前綴打到同一節點」時才有意義，否則每台節點各算一份 KV、命中率歸零。一致性雜湊（含虛擬節點）的價值在於：節點擴縮時只有 `1/N` 的 key 需要重新映射，而非全量重洗——避免一台機器上下線就讓全叢集 cache 失效。`mmh3.hash` 比 Python 內建 `hash` 更快且跨進程穩定（內建 hash 有 randomization）。實務上 SGLang router 會在純前綴雜湊之外，再混入負載均衡因子（cache 命中 vs. 節點壓力的折衷），避免熱門前綴把單一節點打爆。

---

### R.2.8　語意快取（GPTCache-style）

**用途**：以 GPTCache 風格把查詢 embed 後存進向量資料庫，新查詢經 ANN 檢索若語意夠近（cosine 相似度 > 門檻）即直接回傳快取答案，省去一次 LLM 推論。

```python
import numpy as np
# 任一 embedding 模型 + 任一向量庫（此處以 Qdrant 為例，亦可 Milvus）
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class SemanticCache:
    def __init__(self, embed_fn, collection="llm_cache",
                 dim=768, threshold=0.98):
        self.embed_fn = embed_fn          # query -> np.ndarray[dim]
        self.threshold = threshold        # cosine 門檻，0.98 偏保守避免誤命中
        self.collection = collection
        self.client = QdrantClient(":memory:")   # 正式環境連 Qdrant/Milvus 服務
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
        if hits and hits[0].score >= self.threshold:   # COSINE 距離已是相似度
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
            return cached                  # 命中：免一次 LLM 推論
        answer = llm_call(query)           # 未命中：真正打 LLM
        self.put(query, answer)
        return answer


if __name__ == "__main__":
    def fake_embed(q: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(q)) % (2**32))
        v = rng.standard_normal(768)
        return (v / np.linalg.norm(v)).astype(np.float32)

    cache = SemanticCache(fake_embed)
    print(cache.ask("What is the capital of France?",
                    lambda q: "Paris"))    # 未命中 -> 呼叫 LLM
    # 語意相近的問法可在門檻設定下命中（此處用隨機 embed 僅示意流程）
```

**點評**：語意快取與前述 KV/前綴快取屬不同層級——它不重用 KV，而是直接重用「最終答案」，命中即省下整次推論，性價比最高但風險也最高。`threshold` 是核心旋鈕：太低（如 0.9）會把「法國首都」與「德國首都」誤判為同義而回錯答案；0.98 偏保守，適合 FAQ / 文件問答這類確定性高的場景，而不適合需要精確區分細節的對話。正式部署時 embedding 模型要與業務語料對齊，向量庫（Milvus / Qdrant）需設定 TTL 與容量淘汰，並對命中結果做抽樣品質監控，避免快取毒化（cache poisoning）。


---

## R.3　量化、推理部署與評測核心實作

本節彙整從「訓練後量化」到「上線服務」與「自動評測」的關鍵實作。所有程式碼以開源權重模型與開放授權資料為前提，可直接作為工程參考。各片段附一行用途說明與簡短**點評**。

---

### R.3.1　QLoRA 微調配置

> 用途：以 4-bit NF4 量化載入基座模型，僅訓練低秩 LoRA adapter，於單卡完成大模型微調。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit NF4 雙量化：顯著降低權重顯存佔用，計算時反量化為 bf16
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

# 啟用梯度檢查點 + 為 k-bit 訓練做前處理（轉 layernorm 為 fp32、開啟輸入梯度）
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,                       # alpha/r = 2，等效縮放係數穩定
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                       # 同時覆蓋 attention 與 MLP 兩組投影
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()         # 可訓練參數通常 < 2%
```

**點評**：`target_modules` 同時掛到 q/k/v/o 與 gate/up/down 是 QLoRA 論文「把 LoRA 加到所有線性層」的實務落地，比只掛 q/v 收斂更穩。`gradient_checkpointing` 與 NF4 是單卡訓 7B–13B 的關鍵；注意要在 `prepare_model_for_kbit_training` 後再 `get_peft_model`，且訓練時設 `gradient_checkpointing_kwargs={"use_reentrant": False}` 以相容新版 PyTorch。

---

### R.3.2　GPTQ / AWQ 訓練後量化

> 用途：以少量校準語料做 4-bit 權重量化，輸出可直接推理的量化權重。

```python
# --- GPTQ（gptqmodel，autogptq 的後繼維護版）---
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

calib = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib = [next(iter(calib))["text"] for _ in range(256)]   # 128~512 條校準樣本即可

quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True)
model = GPTQModel.load("meta-llama/Llama-3.1-8B", quant_cfg)
model.quantize(calib)
model.save("./Llama-3.1-8B-gptq-int4")
```

```python
# --- AWQ（autoawq）：activation-aware，逐通道保護顯著權重 ---
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
model.quantize(tok, quant_config=quant_config)        # 內建預設校準集，亦可傳入 calib_data
model.save_quantized("./Llama-3.1-8B-awq-int4")
tok.save_pretrained("./Llama-3.1-8B-awq-int4")
```

**點評**：GPTQ 以逐層 Hessian 最小化重建誤差，`desc_act=True`（act-order）多半能再壓低困惑度但量化稍慢。AWQ 不更新權重、只做縮放保護，量化快且對指令模型友善。`group_size=128` 是精度/體積的常見折衷；兩者輸出皆可被 vLLM 直接載入（`--quantization gptq_marlin` / `awq_marlin`）。

---

### R.3.3　GGUF 轉換與 k-quant

> 用途：將 HF 權重轉為 llama.cpp 的 GGUF 格式，再做 k-quant 壓縮供 CPU/邊緣推理。

```bash
# 1) HF -> GGUF（先轉 fp16/bf16 全精度，再量化）
python convert_hf_to_gguf.py ./Llama-3.1-8B \
    --outfile model-f16.gguf \
    --outtype f16

# 2) k-quant：Q4_K_M 為品質/體積最常用平衡點
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# 3)（可選）用 importance matrix 提升低位元品質
./llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ4_XS.gguf IQ4_XS
```

**點評**：`Q4_K_M` 對多數 7B–70B 是體積與精度的甜蜜點；要更小可用 `Q3_K_M`，要更接近原始用 `Q5_K_M` 或 `Q6_K`。低於 4-bit（如 `IQ4_XS`/`IQ3`）務必搭配 imatrix，否則退化明顯。轉檔失敗多半是 chat template 或新架構未支援，需更新 llama.cpp。

---

### R.3.4　投機解碼（Speculative Decoding）

> 用途：用小草稿模型一次提出多個 token，由大模型平行驗證，降低延遲而不改變輸出分布。

```python
# vLLM：draft 模型投機解碼（新版以 speculative_config 傳入）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",  # 同 tokenizer 家族的小模型
        "num_speculative_tokens": 5,
    },
    tensor_parallel_size=4,
)
out = llm.generate(["解釋投機解碼的原理。"], SamplingParams(temperature=0.0, max_tokens=256))
```

```bash
# llama.cpp：主模型 + 草稿模型
./llama-server \
    -m  model-70B-Q4_K_M.gguf \
    -md model-1B-Q4_K_M.gguf \
    --draft-max 8 --draft-min 1 \
    -ngl 99 -c 8192
```

**點評**：草稿模型需與主模型共用 vocab（同家族最穩）。加速比取決於「接受率」，在低溫、結構化或重複性高的輸出上收益最大；高溫創意生成接受率低、可能不划算。`num_speculative_tokens` 過大反而浪費驗證算力，5–8 為常見起點。

---

### R.3.5　vLLM 服務啟動

> 用途：以 OpenAI 相容 API 啟動高吞吐推理服務，並調校顯存與長上下文。

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name llama3.1-8b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --max-num-seqs 256
# 之後即可用 OpenAI SDK 指向 http://localhost:8000/v1
```

**點評**：`--gpu-memory-utilization` 決定 KV cache 預留，0.85–0.92 視同卡其他負載而定，過高易 OOM。`--enable-chunked-prefill` 讓長 prompt 與 decode 交錯，降低 TTFT 抖動。`--kv-cache-dtype fp8` 約省一半 KV 顯存、換取極小精度損失，是擴大並發或上下文的高 CP 值手段；`--max-model-len` 切勿超過權重實際支援長度。

---

### R.3.6　distilabel 合成蒸餾資料

> 用途：用「開放權重」教師模型批次生成指令回應，產出可用於 SFT 的蒸餾資料集。

```python
# 教師為開放權重模型（Llama-3.3-70B），以本地 vLLM 服務提供 —— 合規、可商用授權範圍內
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.models import OpenAILLM   # 指向自建 vLLM 的 OpenAI 相容端點

teacher = OpenAILLM(
    model="llama3.3-70b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
)

with Pipeline(name="distill-sft") as pipeline:
    load = LoadDataFromHub(repo_id="HuggingFaceH4/instruction-dataset", split="test")
    gen = TextGeneration(llm=teacher, input_batch_size=16)   # 以 instruction 欄生成 generation
    keep = KeepColumns(columns=["instruction", "generation"])
    load >> gen >> keep

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    distiset.push_to_hub("your-org/distilled-sft")
```

**點評**：教師必須是**開放權重或授權允許蒸餾**的模型（如 Llama / Qwen / Mistral 開源版），切勿爬取或繞過受保護的商用 API、亦不得用於去審查。`use_cache=True` 可在中斷後續跑、省算力；產出建議再過一層品質/去重過濾（如 distilabel 的 evol/quality 評分步驟）再進 SFT。

---

### R.3.7　lm-eval-harness 評測與長上下文

> 用途：以業界標準 harness 跑下游基準，量化模型品質回歸。

```bash
# 標準學科與數學推理基準
lm_eval --model hf \
    --model_args pretrained=./Llama-3.1-8B-awq-int4,dtype=bfloat16 \
    --tasks mmlu,gsm8k,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./eval_results

# 直接評測 vLLM 服務（吞吐更高）
lm_eval --model local-completions \
    --model_args base_url=http://localhost:8000/v1/completions,model=llama3.1-8b \
    --tasks gsm8k --batch_size 16
```

**長上下文補充**：MMLU/GSM8K 無法反映長文檢索能力，需另跑 **RULER**（多種合成任務、可指定 4k–128k 多個長度）與 **NIAH（Needle-in-a-Haystack）**「大海撈針」測試，觀察隨上下文增長的召回衰減曲線；KV cache 量化（如 R.3.5 的 fp8）或 RoPE 外推設定務必在這兩項上回歸驗證。

**點評**：量化模型上線前，至少在 GSM8K（推理）與 MMLU（知識）對比原模型，4-bit 通常掉 1–3 分屬可接受；若大幅退化多半是量化校準集或 group_size 不當。固定 `--num_fewshot` 與版本，分數才可跨次比較。

---

### R.3.8　Ollama Modelfile 封裝

> 用途：把 GGUF 權重連同推理參數與對話模板封裝成可一鍵分發的 Ollama 模型。

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

**點評**：`TEMPLATE` 必須吻合該基座的 chat format（此處為 Llama-3 的 header 標記），錯誤模板會讓模型答非所問或停不下來。`stop` 與 `num_ctx` 要與訓練/量化時一致；Modelfile 讓「權重 + 參數 + 模板」成為單一可版本化產物，極利於團隊複現。

---

### R.3.9　雙模型動態路由

> 用途：以複雜度啟發式把簡單請求導向小模型、困難請求導向大模型，平衡成本與品質。

```python
# 輕量路由：依長度/關鍵詞估複雜度，分流到便宜或強力模型（OpenAI 相容端點皆可）
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re

app = FastAPI()
cheap = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # 小模型 vLLM
big   = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")   # 大模型 vLLM

HARD_HINTS = re.compile(r"(證明|推導|程式|debug|逐步|多步|規劃|數學|code|prove)", re.I)

class Query(BaseModel):
    prompt: str

def complexity_score(text: str) -> float:
    score = 0.0
    score += min(len(text) / 800, 1.0)          # 長度訊號
    score += 0.5 if HARD_HINTS.search(text) else 0.0  # 任務型關鍵詞
    score += 0.3 if text.count("\n") >= 4 else 0.0    # 多段/結構化輸入
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

**點評**：啟發式路由零額外延遲、易解釋，是上線初期最佳起點；待累積流量後，可換成「以小模型打分」或專用 router 模型（如 RouteLLM）做學習式路由。務必設定**回退**——大模型超時或 5xx 時降級到小模型，並記錄路由決策以便離線比對誤判率。若用 LiteLLM，可把上述兩端點掛進其 Router 並交由內建負載均衡與 fallback 管理。


---

