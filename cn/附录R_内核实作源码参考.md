# 附录R　内核实作源码参考

> 本附录把全书关键技术的**参考实作**集中起来——蒸馏损失、KV Cache/PagedAttention、量化与部署。代码以可读、可改、贴近真实函数库为目标（PyTorch / transformers / vLLM / llama.cpp / peft / distilabel），每段附**目标**与**点评**。所有教师模型一律假设为**你有权使用的对象**（开放权重或你自有模型）；本附录**不含**任何规避防护、反指纹、去审查（abliteration）代码——那条红线见附录 E 与第 6 章。

> ⚠️ 这些是**参考骨架**，非生产级拷贝粘贴即用：实际部署需自行补上错误处理、形状检查、分布式同步与版本兼容。把它们当成「看懂原理后的起手式」。

---

## R.1　知识蒸馏内核实作

本节汇整知识蒸馏（Knowledge Distillation, KD）在大型语言模型训练中的内核代码骨架，皆以 PyTorch 与 Hugging Face `transformers` 生态为基础。所有教师模型均假设为你具有合法使用权的开源或自有权重模型（例如以自家大模型蒸馏出小模型）。代码以可读、可直接移植为目标，实务上需依数据管线与设备数量微调。

---

### R.1.1　Hinton 软硬标签 KD 损失

最经典的 KD 损失：以温度 `T` 软化教师与学生的几率分布，用 KL 散度逼近教师「暗知识」，再以 `α` 与真实标签的交叉熵混合。注意软标签梯度因 softmax 对 `1/T` 的缩放会被压低，需乘回 `T²` 以维持与硬标签同量级。

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5,
            ignore_index=-100):
    """Hinton (2015) soft + hard label distillation.

    student_logits, teacher_logits: (B, V) or (B, L, V)
    labels: (B,) or (B, L) ground-truth token ids
    """
    # 软标签：教师不参与反传，需 detach
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)  # 补回因 1/T 缩放被压低的梯度量级

    # 硬标签：标准交叉熵（自动忽略 padding）
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

**点评**：`reduction="batchmean"` 才是数学定义上正确的 KL 平均（`"mean"` 会多除以词表维度而失真）；`T*T` 缩放与教师 `detach()` 是两个最常被遗漏的细节。

---

### R.1.2　Forward 与 Reverse KL（mode-covering vs mode-seeking）

KD 的方向性决定学生如何「妥协」。Forward KL `KL(p_teacher ‖ p_student)` 是 mode-covering：学生被迫覆盖教师所有支撑集，对小模型容易学出过度平滑、含幻觉的分布。Reverse KL `KL(p_student ‖ p_teacher)` 是 mode-seeking：学生聚焦教师高几率模式，产生更锐利、更稳定的输出，这是 MiniLLM 与 GKD 在 LLM 蒸馏上偏好 reverse KL 的原因。

```python
import torch.nn.functional as F

def forward_kl(student_logits, teacher_logits, T=1.0):
    """KL(teacher || student) — mode-covering，学生覆盖教师全分布。"""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def reverse_kl(student_logits, teacher_logits, T=1.0):
    """KL(student || teacher) — mode-seeking，学生聚焦教师主峰。"""
    # 注意：第一参数须为 log-prob，故对 teacher 取 log_softmax；
    # target 为 student 几率，但须允许其反传，故不 detach student。
    p_student = F.softmax(student_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits.detach() / T, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean() * (T * T)
```

**点评**：生成式 LLM 蒸馏通常选 reverse KL（或 GKD 的 Jensen-Shannon 广义插值），避免 forward KL 逼学生「什么都学一点」而产生流畅但不可靠的长尾；分类或词表受限任务则 forward KL 仍够用。

---

### R.1.3　串行级 KD／on-policy 蒸馏 loop 骨架

串行级 KD（GKD/on-policy distillation）的关键：训练数据由**学生自己采样**，再用教师对学生产生的串行打分，从而消除 teacher-forcing 下的曝光偏差（exposure bias）。以下为单训练步骤骨架。

```python
import torch

@torch.no_grad()
def sample_from_student(student, tokenizer, prompts, max_new_tokens=256):
    """On-policy：由学生自身生成串行（之后交给教师评分）。"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(student.device)
    gen = student.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=1.0, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].size(1)
    return gen, prompt_len  # gen: (B, prompt_len + new_len)

def on_policy_step(student, teacher, tokenizer, prompts, kl_fn, optimizer, T=1.0):
    # 1) 学生自采样（on-policy data）
    seqs, prompt_len = sample_from_student(student, tokenizer, prompts)
    attn = (seqs != tokenizer.pad_token_id).long()

    # 2) 学生对自己串行前传（需梯度）
    s_logits = student(seqs, attention_mask=attn).logits  # (B, L, V)

    # 3) 教师对同串行评分（冻结、no_grad）
    with torch.no_grad():
        t_logits = teacher(seqs, attention_mask=attn).logits

    # 4) 仅对生成段（response）计 loss，shift 对齐 next-token
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

**点评**：on-policy 采样成本高，实务常以 `lambda`（GKD 的 `lmbda`）几率混合 on-policy 与固定数据；`trl.GKDTrainer` 已封装此逻辑，自写骨架主要用于理解与客制。

---

### R.1.4　CoT 加权损失（`<think>...</think>` 内 token 加权）

蒸馏推理链（chain-of-thought）时，思考段往往比最终答案更承载教师的推理能力。以下依串行中 `<think>` 与 `</think>` 标记创建逐 token 权重遮罩，对思考段给予较高权重。

```python
import torch

def build_cot_weight_mask(input_ids, tokenizer, think_weight=2.0, base_weight=1.0,
                          open_tag="<think>", close_tag="</think>"):
    """为每个 token 产生权重；<think>...</think> 区间内权重较高。"""
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
    """per_token_loss/(weight_mask/pad_mask): (B, L)；回传纯量。"""
    w = weight_mask * pad_mask  # padding 权重归零
    return (per_token_loss * w).sum() / w.sum().clamp_min(1.0)
```

**点评**：`per_token_loss` 可由 `F.cross_entropy(..., reduction="none")` 或逐 token KL 得到；标签切分若用特殊 token（建议将 `<think>`/`</think>` 加入 tokenizer special tokens），可避免被 BPE 拆成多个子词而漏抓边界。

---

### R.1.5　跨词表对齐（cross-tokenizer logit projection）

当教师与学生使用不同 tokenizer 时，logit 无法逐位对齐。常见作法：取教师 top-k token，解码成字符串，用学生 tokenizer 重新编码，再把教师几率质量重分配到学生词表上对应的（首）token。以下为示意骨架，省略了完整的子词合并策略。

```python
import torch

@torch.no_grad()
def project_teacher_to_student(teacher_logits, teacher_tok, student_tok,
                               student_vocab_size, top_k=64, T=1.0):
    """将教师单步分布投影到学生词表（top-k 字符串重编码 + 质量重分配）。

    teacher_logits: (V_t,) 单一位置的教师 logits
    回传: (V_s,) 学生词表上的目标几率分布
    """
    probs = torch.softmax(teacher_logits / T, dim=-1)
    topk = torch.topk(probs, k=top_k)
    student_dist = torch.zeros(student_vocab_size, device=teacher_logits.device)

    for tok_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
        # 教师 token -> 字符串 -> 学生 tokenizer 重新编码
        piece = teacher_tok.decode([tok_id]).strip()
        if not piece:
            continue
        s_ids = student_tok.encode(piece, add_special_tokens=False)
        if not s_ids:
            continue
        # 质量归给学生编码的首 token（简化策略；可改为沿子词链分摊）
        student_dist[s_ids[0]] += p

    total = student_dist.sum()
    if total > 0:
        student_dist /= total  # 重新归一化，丢弃无法对齐的尾部
    return student_dist
```

**点评**：此为示意而非生产级——严谨作法需处理多子词 token 的几率沿链分摊、空白前缀（`▁`/`Ġ`）差异与归一化偏差；ULD（Universal Logit Distillation）以最佳传输（optimal transport）对齐两词表，可作为更稳健的替代。

---

### R.1.6　完整训练步骤（HF Trainer-style）

集成上述模块为自订 `Trainer`：教师冻结并设 `eval()`，学生可搭配 PEFT LoRA 降低显存；分布式以 `accelerate` 设置档激活 DeepSpeed ZeRO-3 或 FSDP，并打开梯度检查点。

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
        # 蒸馏损失（R.1.1）：next-token 对齐需 shift
        s_logits = s_out.logits[:, :-1, :].contiguous()
        t_logits = t_out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = kd_loss(s_logits, t_logits, shift_labels,
                       T=self.T, alpha=self.alpha)
        return (loss, s_out) if return_outputs else loss

# --- 组装 ---
student = AutoModelForCausalLM.from_pretrained("your/small-base",
                                               torch_dtype=torch.bfloat16)
student.gradient_checkpointing_enable()         # 省显存
student.config.use_cache = False                # 与梯度检查点互斥
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
    deepspeed="ds_zero3.json",   # 或以 accelerate config 激活 FSDP
    logging_steps=10, num_train_epochs=3,
)
trainer = DistillTrainer(teacher_model=teacher, T=2.0, alpha=0.5,
                         model=student, args=args,
                         train_dataset=train_ds, tokenizer=tokenizer)
trainer.train()
```

启动时以 `accelerate launch --config_file fsdp.yaml train.py`，或在 `TrainingArguments` 直接挂 `deepspeed="ds_zero3.json"`（ZeRO-3 将参数、梯度、优化器状态三者分片至各 GPU）。教师若过大，可单独以另一进程／量化（如 8-bit）加载并仅做前向评分。

**点评**：教师务必 `eval()` + `requires_grad_(False)` 并包在 `no_grad` 内，否则显存与算力浪费在不更新的梯度上；`gradient_checkpointing` 与 `use_cache=True` 互斥，需手动关闭 cache 才不会报警告或失效。


---

## R.2　KV Cache 与缓存内核实作

本节汇整推论服务中与 KV Cache 相关的内核参考实作，涵盖从显存估算、PagedAttention 分页管理、CUDA 寻址、RadixAttention 前缀共享，到 FlashDecoding、KV 量化与多层缓存路由。所有代码皆以可读性为先、语意正确为本，命名与第 10、11 章一致（vLLM / SGLang / FlashAttention / Triton）。

---

### R.2.1　KV Cache 显存计算器

**用途**：估算一段串行的 KV Cache 占用字节，是容量规划与 batch 上限推导的起点。

```python
def kv_cache_bytes(
    L: int,            # Transformer 层数 (num_layers)
    kv_heads: int,     # KV head 数 (GQA/MQA 后的 num_key_value_heads)
    head_dim: int,     # 每个 head 维度
    seq_len: int,      # 串行长度 (prompt + generated)
    batch: int = 1,    # 并发串行数
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp8/int8 = 1
) -> int:
    """KV Cache 显存 = 2 (K 与 V 各一份) * dtype_bytes
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

**点评**：每 token 128KB 是 Llama-3-8B 的关键常数——`2 * 2 * 32 * 8 * 128 = 131072 = 128KB`。GQA 把 KV head 从 32 砍到 8，等于把 cache 缩小 4 倍；若用 MHA（kv_heads=32）则每 token 暴增到 512KB，batch=128×8K 直接吃满 512GB。实务上 `2 * dtype_bytes * L * kv_heads * head_dim` 应视为「每 token 常数」先算出来，再乘 `seq_len * batch`，方便快速心算容量天花板。

---

### R.2.2　PagedAttention BlockManager / BlockTable

**用途**：以 vLLM 风格的固定大小区块（block）管理 KV Cache 显存，消除连续配置造成的内外部碎片。

```python
from collections import deque
from typing import Dict, List


class PhysicalTokenBlock:
    """一块固定容纳 block_size 个 token 的物理显存区块。"""
    def __init__(self, block_number: int, block_size: int):
        self.block_number = block_number   # 物理区块索引
        self.block_size = block_size
        self.ref_count = 0                 # 被几个串行共享 (copy-on-write 用)

    def __repr__(self):
        return f"PhysicalTokenBlock(#{self.block_number}, ref={self.ref_count})"


class BlockAllocator:
    """以 free-list 管理物理区块的配置/释放，O(1) 配置。"""
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
    """单一串行的 logical block -> physical block 映射。"""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.blocks: List[PhysicalTokenBlock] = []
        self.num_tokens = 0

    def append_token(self) -> None:
        block_size = self.allocator.block_size
        # 最后一块已满（或尚无区块）才需要再配一块新的物理区块
        if self.num_tokens % block_size == 0:
            self.blocks.append(self.allocator.allocate())
        self.num_tokens += 1

    def physical_index(self, logical_token_idx: int) -> int:
        """把 logical token 位置翻译成 (block_number, offset)。"""
        block_size = self.allocator.block_size
        block = self.blocks[logical_token_idx // block_size]
        offset = logical_token_idx % block_size
        return block.block_number, offset

    def fork(self) -> "BlockTable":
        """beam search / parallel sampling 的 copy-on-write 分叉：
           共享既有区块，仅增加 ref_count。"""
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

**点评**：PagedAttention 的精髓是把 KV Cache 切成 OS 分页式的固定区块，`BlockTable` 即「页表」，`BlockAllocator.free` 即「空闲页链」。`ref_count` 让 parallel sampling / beam search 共享同一份 prompt 的 KV（copy-on-write），只有在某分支写入已共享的区块时才真正拷贝。`num_tokens % block_size == 0` 是配置新区块的唯一条件——这是分页摊平内存碎片、把利用率推到 90%+ 的关键。当 `num_free == 0` 时必须丢出让调度器走 preemption（swap-out 或 recompute），不可静默失败。

---

### R.2.3　paged_attention CUDA kernel 寻址内核

**用途**：示意 CUDA kernel 内部如何通过 block_table 把 logical token 翻译成物理显存指针，再读取 K/V。

```cuda
// 对单一 query head，扫过某串行的所有 KV token。
// block_table 为该串行的「页表」：logical block idx -> physical block number。
// kv_cache 布局: [num_blocks, num_kv_heads, head_dim, block_size]
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    scalar_t*       __restrict__ out,          // [num_heads, head_dim]
    const scalar_t* __restrict__ q,            // [num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,      // 物理 KV 池
    const scalar_t* __restrict__ v_cache,
    const int*      __restrict__ block_table,  // [max_blocks_per_seq]
    const int       context_len,               // 此串行目前 token 数
    const int       num_kv_heads) {

    const int kv_head = blockIdx.y;            // 此 thread block 负责的 KV head
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int logical_block = 0; logical_block < num_blocks; ++logical_block) {
        // 1) 页表查找：logical block -> 物理 block number
        const int physical_block = block_table[logical_block];

        // 2) 本块内要处理的 token 数（最后一块可能不满）
        const int tokens_in_block =
            min(BLOCK_SIZE, context_len - logical_block * BLOCK_SIZE);

        for (int t = 0; t < tokens_in_block; ++t) {
            // 3) 计算物理位移：定位到 (physical_block, kv_head) 的 head_dim 连续区段
            const int64_t base =
                (((int64_t)physical_block * num_kv_heads + kv_head)
                 * HEAD_DIM) * BLOCK_SIZE;

            const scalar_t* k_ptr = k_cache + base + t;  // K[..., :, t]
            const scalar_t* v_ptr = v_cache + base + t;

            // 4) dot(q, k) -> score，累积到 online-softmax（此处略）
            //    实作中会配合 running max / running sum 做数值稳定的在线 softmax
            //    并把 score * v 累加进 out。
            (void)k_ptr; (void)v_ptr;
        }
    }
}
```

**点评**：相较传统 attention 假设 KV 在显存中「连续」，paged kernel 多了一层 `block_table[logical_block]` 的间接寻址——这正是 PagedAttention 能用非连续物理页拼出逻辑连续串行的代价与精髓。布局 `[num_blocks, num_kv_heads, head_dim, block_size]` 把同一 head 的连续 token 放在最内维，让 warp 内 thread 能 coalesced 读取。实际 vLLM kernel 在内层用 online-softmax（running max + running sum）避免存下完整 score 矩阵；本 sketch 聚焦在「页表查找 → 物理位移」这条寻址主线。

---

### R.2.4　RadixAttention 前缀树

**用途**：以 SGLang 风格的 radix tree 共享相同前缀（system prompt、few-shot、多轮历史）的 KV Cache，达成跨请求自动缓存重用。

```python
import time
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """边上存一段 token 串行，节点挂该前缀对应的 KV cache 指针（区块清单）。"""
    def __init__(self):
        self.children: Dict[int, "RadixNode"] = {}  # 以首 token 为 key
        self.key: List[int] = []                    # 此节点到父节点的 token 段
        self.kv_blocks: List[int] = []              # 对应 KV 物理区块 id
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
        """回传已被缓存的最长前缀 KV 区块，与停在的节点。"""
        node, matched, i = self.root, [], 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            m = self._match_len(child.key, tokens[i:])
            node_kv = child.kv_blocks[:m]
            matched.extend(node_kv)
            if m < len(child.key):     # 部分匹配，停在边中间
                return matched, child
            node, i = child, i + m
            node.last_access = time.monotonic()
        return matched, node

    def insert(self, tokens: List[int], kv_blocks: List[int]) -> None:
        """插入一条完整串行；自动在分歧点分裂节点。"""
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
            if m < len(child.key):     # 边上分歧 -> 分裂出中间节点
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
        """LRU 逐出：只逐出叶节点，保留 root，回收的 KV 区块回传给 allocator。"""
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

**点评**：RadixAttention 把「prefix caching」一般化成树——任意两个请求只要共享前缀（不限于 system prompt），就能在 `match_prefix` 命中既有 KV 区块而免去重算。`_split` 是 radix tree 的灵魂：当新串行在某条边中途分歧时，必须把边裂成「共享段 + 两个分支」。逐出策略刻意只动叶节点并保留 root，确保被多请求引用的热前缀（通常靠近 root）不会被误逐。SGLang 量测显示，在多轮对话与 few-shot 场景下命中率可达 50%+，TTFT 显著下降。

---

### R.2.5　FlashDecoding split-KV reduction

**用途**：示意 FlashDecoding 如何在 decode 阶段把超长 KV 沿串行维切成多块并行计算，再用 log-sum-exp 合并部分结果，提升长串行下的 GPU 占用率。

```python
import numpy as np


def flash_decoding(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   num_splits: int = 4):
    """q: [d], K/V: [seq, d]。把 KV 沿 seq 切 num_splits 块平行算，
       每块各自做数值稳定 softmax，最后用 log-sum-exp 合并。"""
    seq, d = K.shape
    scale = 1.0 / np.sqrt(d)
    chunk = (seq + num_splits - 1) // num_splits

    partial_out = []   # 每块的加权 V 累积（已除以块内 sum）
    partial_lse = []   # 每块的 log-sum-exp (含 running max 修正)

    # --- 各 split 独立计算（GPU 上对应不同 thread block，可平行）---
    for s in range(num_splits):
        lo, hi = s * chunk, min((s + 1) * chunk, seq)
        if lo >= hi:
            continue
        scores = (K[lo:hi] @ q) * scale          # [chunk]
        m = scores.max()                          # 块内 running max
        p = np.exp(scores - m)                    # 稳定化
        l = p.sum()                               # 块内分母
        o = (p @ V[lo:hi]) / l                    # 块内加权平均
        partial_out.append(o)
        partial_lse.append(m + np.log(l))         # log-sum-exp

    # --- reduction：跨 split 用 log-sum-exp 重新加权合并 ---
    lses = np.array(partial_lse)
    global_max = lses.max()
    weights = np.exp(lses - global_max)           # 各块权重 (= 各块真实分母比例)
    weights /= weights.sum()
    out = sum(w * o for w, o in zip(weights, partial_out))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, seq = 128, 8192
    q = rng.standard_normal(d)
    K = rng.standard_normal((seq, d))
    V = rng.standard_normal((seq, d))

    # 与单块 reference softmax 对照，验证数值一致
    ref_scores = (K @ q) / np.sqrt(d)
    ref = (np.exp(ref_scores - ref_scores.max()) @ V) / \
          np.exp(ref_scores - ref_scores.max()).sum()
    fd = flash_decoding(q, K, V, num_splits=8)
    print("max abs diff:", np.abs(ref - fd).max())   # ~1e-15
```

**点评**：decode 阶段 batch 常常很小（甚至 =1），单一 query 对上很长的 KV，传统 kernel 只能让少数 SM 工作、GPU 严重闲置。FlashDecoding 沿 KV 的 seq 维再切一刀，让多个 SM 平行扫不同段，最后用 log-sum-exp 把各段的部分 softmax 正确合并——关键在于每段都带着自己的 `m`（running max）与 `lse`，合并时以 `exp(lse_i - global_max)` 重新加权，数学上与单块 softmax 完全等价（上例误差 ~1e-15）。长 context 推论的 decode 吞吐提升多来自此。

---

### R.2.6　KV Cache 量化（FP8 / INT8 per-token）

**用途**：在写入 KV Cache 前对 K/V 做 per-token 量化（INT8 或 FP8），读取时反量化，于精度损失极小下把 cache 显存与带宽砍半。

```python
import numpy as np


def quantize_kv_int8(x: np.ndarray):
    """per-token 对称量化：x [num_tokens, head_dim] -> int8 + per-token scale。
       每个 token 用自己的 absmax 求 scale，避免离群 token 拖垮整体。"""
    absmax = np.abs(x).max(axis=-1, keepdims=True)     # [num_tokens, 1]
    scale = absmax / 127.0
    scale = np.maximum(scale, 1e-8)                    # 防除零
    q = np.round(x / scale).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_kv_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """读取时还原回 fp16/fp32 供 attention 计算。"""
    return q.astype(np.float32) * scale


def quantize_kv_fp8(x: np.ndarray):
    """FP8 (E4M3) per-token：以 scale 把数值缩放进 E4M3 动态范围(±448)再转存。
       这里用 numpy 仿真，实际由 Triton/cuda 写入 torch.float8_e4m3fn。"""
    FP8_MAX = 448.0
    absmax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(absmax / FP8_MAX, 1e-8)
    scaled = np.clip(x / scale, -FP8_MAX, FP8_MAX)
    # 仿真 E4M3 的有限尾数：以 round-to-nearest 量化尾数字
    q = _round_to_e4m3(scaled)
    return q, scale.astype(np.float32)


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    exp = np.floor(np.log2(np.maximum(ax, 1e-12)))
    mant_step = np.power(2.0, exp - 3)        # E4M3：3 个尾数字
    return sign * np.round(ax / mant_step) * mant_step


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = rng.standard_normal((16, 128)).astype(np.float32)

    q, s = quantize_kv_int8(k)
    err = np.abs(k - dequantize_kv_int8(q, s)).mean()
    print("INT8 mean abs err:", err)          # ~1e-3, 显存减半
```

**点评**：KV Cache 量化是「以少量精度换显存与带宽」的高性价比手段——INT8 直接砍半、FP8（E4M3）在 Hopper/Ada 上更有原生硬件支持。per-token 量化（每个 token 一个 scale）比 per-tensor 稳健得多，因为 attention 中常有少数离群 token 的数值特别大，共用一个 scale 会让其他 token 量化分辨率被牺牲。实务上 K 比 V 对量化更敏感（K 进 softmax 会被指数放大），因此常见「K 用 FP8、V 用 INT8」或对 K 保留较高精度的混合策略。

---

### R.2.7　Cache-aware routing（前缀哈希一致性路由）

**用途**：在多节点推论集群中，依请求前缀（system prompt + 历史）算哈希，用一致性哈希选节点，把相同前缀的请求稳定路由到同一节点以最大化 RadixAttention / prefix cache 命中。

```python
import bisect
import mmh3   # MurmurHash3，快速且分布均匀


class ConsistentHashRouter:
    """一致性哈希环 + 虚拟节点，避免节点增减造成大规模 cache 失效。"""
    def __init__(self, nodes, vnodes: int = 160):
        self.ring = {}           # hash -> node
        self.sorted_keys = []
        self.vnodes = vnodes
        for n in nodes:
            self.add_node(n)

    def _hash(self, key: str) -> int:
        # 取 32-bit 无号哈希作为环上座标
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
        """以前缀哈希选节点：相同前缀永远落在环上同一位置 -> 同节点。"""
        prefix_key = system_prompt + "\x00" + history
        h = self._hash(prefix_key)
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


if __name__ == "__main__":
    router = ConsistentHashRouter(["gpu-0", "gpu-1", "gpu-2", "gpu-3"])
    sys_p = "You are a helpful assistant."
    print(router.route(sys_p, "user: hi"))      # 同前缀稳定命中同节点
    print(router.route(sys_p, "user: hi"))       # 与上行同节点
```

**点评**：prefix cache 只有在「相同前缀打到同一节点」时才有意义，否则每台节点各算一份 KV、命中率归零。一致性哈希（含虚拟节点）的价值在于：节点扩缩时只有 `1/N` 的 key 需要重新映射，而非全量重洗——避免一台机器上下线就让全集群 cache 失效。`mmh3.hash` 比 Python 内置 `hash` 更快且跨进程稳定（内置 hash 有 randomization）。实务上 SGLang router 会在纯前缀哈希之外，再混入负载均衡因子（cache 命中 vs. 节点压力的折衷），避免热门前缀把单一节点打爆。

---

### R.2.8　语意缓存（GPTCache-style）

**用途**：以 GPTCache 风格把查找 embed 后存进矢量数据库，新查找经 ANN 检索若语意够近（cosine 相似度 > 门槛）即直接回传缓存答案，省去一次 LLM 推论。

```python
import numpy as np
# 任一 embedding 模型 + 任一矢量库（此处以 Qdrant 为例，亦可 Milvus）
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class SemanticCache:
    def __init__(self, embed_fn, collection="llm_cache",
                 dim=768, threshold=0.98):
        self.embed_fn = embed_fn          # query -> np.ndarray[dim]
        self.threshold = threshold        # cosine 门槛，0.98 偏保守避免误命中
        self.collection = collection
        self.client = QdrantClient(":memory:")   # 正式环境连 Qdrant/Milvus 服务
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
        if hits and hits[0].score >= self.threshold:   # COSINE 距离已是相似度
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
            return cached                  # 命中：免一次 LLM 推论
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
                    lambda q: "Paris"))    # 未命中 -> 调用 LLM
    # 语意相近的问法可在门槛设置下命中（此处用随机 embed 仅示意流程）
```

**点评**：语意缓存与前述 KV/前缀缓存属不同层级——它不重用 KV，而是直接重用「最终答案」，命中即省下整次推论，性价比最高但风险也最高。`threshold` 是内核旋钮：太低（如 0.9）会把「法国首都」与「德国首都」误判为同义而回错答案；0.98 偏保守，适合 FAQ / 文档问答这类确定性高的场景，而不适合需要精确区分细节的对话。正式部署时 embedding 模型要与业务语料对齐，矢量库（Milvus / Qdrant）需设置 TTL 与容量淘汰，并对命中结果做抽样品质监控，避免缓存毒化（cache poisoning）。


---

## R.3　量化、推理部署与评测内核实作

本节汇整从「训练后量化」到「上线服务」与「自动评测」的关键实作。所有代码以开源权重模型与开放授权数据为前提，可直接作为工程参考。各片段附一行用途说明与简短**点评**。

---

### R.3.1　QLoRA 微调配置

> 用途：以 4-bit NF4 量化加载基座模型，仅训练低秩 LoRA adapter，於单卡完成大模型微调。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit NF4 双量化：显著降低权重显存占用，计算时反量化为 bf16
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

# 激活梯度检查点 + 为 k-bit 训练做前处理（转 layernorm 为 fp32、打开输入梯度）
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,                       # alpha/r = 2，等效缩放系数稳定
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                       # 同时覆盖 attention 与 MLP 两组投影
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()         # 可训练参数通常 < 2%
```

**点评**：`target_modules` 同时挂到 q/k/v/o 与 gate/up/down 是 QLoRA 论文「把 LoRA 加到所有线性层」的实务落地，比只挂 q/v 收敛更稳。`gradient_checkpointing` 与 NF4 是单卡训 7B–13B 的关键；注意要在 `prepare_model_for_kbit_training` 后再 `get_peft_model`，且训练时设 `gradient_checkpointing_kwargs={"use_reentrant": False}` 以兼容新版 PyTorch。

---

### R.3.2　GPTQ / AWQ 训练后量化

> 用途：以少量校准语料做 4-bit 权重量化，输出可直接推理的量化权重。

```python
# --- GPTQ（gptqmodel，autogptq 的后继维护版）---
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

calib = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib = [next(iter(calib))["text"] for _ in range(256)]   # 128~512 条校准样本即可

quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True)
model = GPTQModel.load("meta-llama/Llama-3.1-8B", quant_cfg)
model.quantize(calib)
model.save("./Llama-3.1-8B-gptq-int4")
```

```python
# --- AWQ（autoawq）：activation-aware，逐信道保护显著权重 ---
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
model.quantize(tok, quant_config=quant_config)        # 内置缺省校准集，亦可传入 calib_data
model.save_quantized("./Llama-3.1-8B-awq-int4")
tok.save_pretrained("./Llama-3.1-8B-awq-int4")
```

**点评**：GPTQ 以逐层 Hessian 最小化重建误差，`desc_act=True`（act-order）多半能再压低困惑度但量化稍慢。AWQ 不更新权重、只做缩放保护，量化快且对指令模型友善。`group_size=128` 是精度/体积的常见折衷；两者输出皆可被 vLLM 直接加载（`--quantization gptq_marlin` / `awq_marlin`）。

---

### R.3.3　GGUF 转换与 k-quant

> 用途：将 HF 权重转为 llama.cpp 的 GGUF 格式，再做 k-quant 压缩供 CPU/边缘推理。

```bash
# 1) HF -> GGUF（先转 fp16/bf16 全精度，再量化）
python convert_hf_to_gguf.py ./Llama-3.1-8B \
    --outfile model-f16.gguf \
    --outtype f16

# 2) k-quant：Q4_K_M 为品质/体积最常用平衡点
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# 3)（可选）用 importance matrix 提升低比特品质
./llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ4_XS.gguf IQ4_XS
```

**点评**：`Q4_K_M` 对多数 7B–70B 是体积与精度的甜蜜点；要更小可用 `Q3_K_M`，要更接近原始用 `Q5_K_M` 或 `Q6_K`。低于 4-bit（如 `IQ4_XS`/`IQ3`）务必搭配 imatrix，否则退化明显。转档失败多半是 chat template 或新架构未支持，需更新 llama.cpp。

---

### R.3.4　投机解码（Speculative Decoding）

> 用途：用小草稿模型一次提出多个 token，由大模型平行验证，降低延迟而不改变输出分布。

```python
# vLLM：draft 模型投机解码（新版以 speculative_config 传入）
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",  # 同 tokenizer 家族的小模型
        "num_speculative_tokens": 5,
    },
    tensor_parallel_size=4,
)
out = llm.generate(["解释投机解码的原理。"], SamplingParams(temperature=0.0, max_tokens=256))
```

```bash
# llama.cpp：主模型 + 草稿模型
./llama-server \
    -m  model-70B-Q4_K_M.gguf \
    -md model-1B-Q4_K_M.gguf \
    --draft-max 8 --draft-min 1 \
    -ngl 99 -c 8192
```

**点评**：草稿模型需与主模型共用 vocab（同家族最稳）。加速比取决于「接受率」，在低温、结构化或重复性高的输出上收益最大；高温创意生成接受率低、可能不划算。`num_speculative_tokens` 过大反而浪费验证算力，5–8 为常见起点。

---

### R.3.5　vLLM 服务启动

> 用途：以 OpenAI 兼容 API 启动高吞吐推理服务，并调校显存与长上下文。

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name llama3.1-8b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --max-num-seqs 256
# 之后即可用 OpenAI SDK 指向 http://localhost:8000/v1
```

**点评**：`--gpu-memory-utilization` 决定 KV cache 预留，0.85–0.92 视同卡其他负载而定，过高易 OOM。`--enable-chunked-prefill` 让长 prompt 与 decode 交错，降低 TTFT 抖动。`--kv-cache-dtype fp8` 约省一半 KV 显存、换取极小精度损失，是扩大并发或上下文的高 CP 值手段；`--max-model-len` 切勿超过权重实际支持长度。

---

### R.3.6　distilabel 合成蒸馏数据

> 用途：用「开放权重」教师模型批量生成指令回应，产出可用于 SFT 的蒸馏数据集。

```python
# 教师为开放权重模型（Llama-3.3-70B），以本地 vLLM 服务提供 —— 合规、可商用授权范围内
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.models import OpenAILLM   # 指向自建 vLLM 的 OpenAI 兼容端点

teacher = OpenAILLM(
    model="llama3.3-70b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
)

with Pipeline(name="distill-sft") as pipeline:
    load = LoadDataFromHub(repo_id="HuggingFaceH4/instruction-dataset", split="test")
    gen = TextGeneration(llm=teacher, input_batch_size=16)   # 以 instruction 栏生成 generation
    keep = KeepColumns(columns=["instruction", "generation"])
    load >> gen >> keep

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    distiset.push_to_hub("your-org/distilled-sft")
```

**点评**：教师必须是**开放权重或授权允许蒸馏**的模型（如 Llama / Qwen / Mistral 开源版），切勿爬取或绕过受保护的商用 API、亦不得用于去审查。`use_cache=True` 可在中断后续跑、省算力；产出建议再过一层品质/去重过滤（如 distilabel 的 evol/quality 评分步骤）再进 SFT。

---

### R.3.7　lm-eval-harness 评测与长上下文

> 用途：以业界标准 harness 跑下游基准，量化模型品质回归。

```bash
# 标准学科与数学推理基准
lm_eval --model hf \
    --model_args pretrained=./Llama-3.1-8B-awq-int4,dtype=bfloat16 \
    --tasks mmlu,gsm8k,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./eval_results

# 直接评测 vLLM 服务（吞吐更高）
lm_eval --model local-completions \
    --model_args base_url=http://localhost:8000/v1/completions,model=llama3.1-8b \
    --tasks gsm8k --batch_size 16
```

**长上下文补充**：MMLU/GSM8K 无法反映长文检索能力，需另跑 **RULER**（多种合成任务、可指定 4k–128k 多个长度）与 **NIAH（Needle-in-a-Haystack）**「大海捞针」测试，观察随上下文增长的召回衰减曲线；KV cache 量化（如 R.3.5 的 fp8）或 RoPE 外推设置务必在这两项上回归验证。

**点评**：量化模型上线前，至少在 GSM8K（推理）与 MMLU（知识）对比原模型，4-bit 通常掉 1–3 分属可接受；若大幅退化多半是量化校准集或 group_size 不当。固定 `--num_fewshot` 与版本，分数才可跨次比较。

---

### R.3.8　Ollama Modelfile 封装

> 用途：把 GGUF 权重连同推理参数与对话模板封装成可一键分发的 Ollama 模型。

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

SYSTEM """你是一个乐于助人、回答精确的中文助理。"""
```

```bash
ollama create my-llama3 -f Modelfile
ollama run my-llama3 "用一句话解释投机解码"
```

**点评**：`TEMPLATE` 必须吻合该基座的 chat format（此处为 Llama-3 的 header 标记），错误模板会让模型答非所问或停不下来。`stop` 与 `num_ctx` 要与训练/量化时一致；Modelfile 让「权重 + 参数 + 模板」成为单一可版本化产物，极利于团队复现。

---

### R.3.9　双模型动态路由

> 用途：以复杂度启发式把简单请求导向小模型、困难请求导向大模型，平衡成本与品质。

```python
# 轻量路由：依长度/关键词估复杂度，分流到便宜或强力模型（OpenAI 兼容端点皆可）
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re

app = FastAPI()
cheap = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # 小模型 vLLM
big   = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")   # 大模型 vLLM

HARD_HINTS = re.compile(r"(证明|推导|程序|debug|逐步|多步|规划|数学|code|prove)", re.I)

class Query(BaseModel):
    prompt: str

def complexity_score(text: str) -> float:
    score = 0.0
    score += min(len(text) / 800, 1.0)          # 长度信号
    score += 0.5 if HARD_HINTS.search(text) else 0.0  # 任务型关键词
    score += 0.3 if text.count("\n") >= 4 else 0.0    # 多段/结构化输入
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

**点评**：启发式路由零额外延迟、易解释，是上线初期最佳起点；待累积流量后，可换成「以小模型打分」或专用 router 模型（如 RouteLLM）做学习式路由。务必设置**回退**——大模型超时或 5xx 时降级到小模型，并记录路由决策以便脱机比对误判率。若用 LiteLLM，可把上述两端点挂进其 Router 并交由内置负载均衡与 fallback 管理。


---

