# Appendix R — Core Reference Implementations / Source Code

> This appendix collects the **reference implementations** of the book's key techniques in one place—distillation loss, KV Cache/PagedAttention, quantization, and deployment. The code aims to be readable, modifiable, and close to real-world libraries (PyTorch / transformers / vLLM / llama.cpp / peft / distilabel), and each snippet comes with a **Purpose** and **Commentary**. Every teacher model is assumed to be **something you are entitled to use** (open weights or your own model); this appendix contains **no** code for bypassing safeguards, anti-fingerprinting, or de-censorship (abliteration)—that red line is covered in Appendix E and Chapter 6.

> ⚠️ These are **reference skeletons**, not production-grade copy-paste-and-run: actual deployment requires you to add error handling, shape checks, distributed synchronization, and version compatibility yourself. Treat them as "an opening move once you understand the principles."

---

## R.1　Core Implementations of Knowledge Distillation

This section gathers the core code skeletons for knowledge distillation (KD) in large-language-model training, all based on PyTorch and the Hugging Face `transformers` ecosystem. Every teacher model is assumed to be an open-source or self-owned weight model that you are legally entitled to use (for example, distilling a small model from your own large model). The code aims to be readable and directly portable; in practice you will need to tune it to your data pipeline and device count.

---

### R.1.1　Hinton Soft + Hard Label KD Loss

The most classic KD loss: soften the teacher's and student's probability distributions with a temperature `T`, use KL divergence to approximate the teacher's "dark knowledge," then mix in cross-entropy against the ground-truth labels via `α`. Note that the soft-label gradient is suppressed by softmax's `1/T` scaling, so it must be multiplied back by `T²` to keep the same magnitude as the hard label.

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5,
            ignore_index=-100):
    """Hinton (2015) soft + hard label distillation.

    student_logits, teacher_logits: (B, V) or (B, L, V)
    labels: (B,) or (B, L) ground-truth token ids
    """
    # Soft label: the teacher does not participate in backprop, so detach it
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)  # restore the gradient magnitude suppressed by 1/T scaling

    # Hard label: standard cross-entropy (automatically ignores padding)
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

**Commentary**: `reduction="batchmean"` is the mathematically correct KL average (`"mean"` divides by the vocabulary dimension as well and distorts the value); the `T*T` scaling and the teacher's `detach()` are the two most commonly forgotten details.

---

### R.1.2　Forward and Reverse KL (mode-covering vs mode-seeking)

The directionality of KD determines how the student "compromises." Forward KL `KL(p_teacher ‖ p_student)` is mode-covering: the student is forced to cover the teacher's entire support, and for small models this easily produces an over-smoothed, hallucination-prone distribution. Reverse KL `KL(p_student ‖ p_teacher)` is mode-seeking: the student focuses on the teacher's high-probability modes, producing sharper, more stable outputs—this is why MiniLLM and GKD prefer reverse KL for LLM distillation.

```python
import torch.nn.functional as F

def forward_kl(student_logits, teacher_logits, T=1.0):
    """KL(teacher || student) — mode-covering; the student covers the teacher's full distribution."""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def reverse_kl(student_logits, teacher_logits, T=1.0):
    """KL(student || teacher) — mode-seeking; the student focuses on the teacher's main peak."""
    # Note: the first argument must be a log-prob, so take log_softmax on the teacher;
    # the target is the student probability, but it must allow backprop, so do NOT detach the student.
    p_student = F.softmax(student_logits / T, dim=-1)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits.detach() / T, dim=-1)
    return (p_student * (log_p_student - log_p_teacher)).sum(-1).mean() * (T * T)
```

**Commentary**: Generative LLM distillation usually picks reverse KL (or GKD's generalized Jensen-Shannon interpolation) to avoid forward KL forcing the student to "learn a little of everything," which yields a fluent but unreliable long tail; for classification or restricted-vocabulary tasks, forward KL is still sufficient.

---

### R.1.3　Sequence-Level KD / On-Policy Distillation Loop Skeleton

The key to sequence-level KD (GKD/on-policy distillation): the training data is **sampled by the student itself**, then scored by the teacher on the student-generated sequences, thereby eliminating the exposure bias of teacher forcing. The following is a single-training-step skeleton.

```python
import torch

@torch.no_grad()
def sample_from_student(student, tokenizer, prompts, max_new_tokens=256):
    """On-policy: generate sequences from the student itself (then handed to the teacher for scoring)."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(student.device)
    gen = student.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=1.0, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].size(1)
    return gen, prompt_len  # gen: (B, prompt_len + new_len)

def on_policy_step(student, teacher, tokenizer, prompts, kl_fn, optimizer, T=1.0):
    # 1) Student self-sampling (on-policy data)
    seqs, prompt_len = sample_from_student(student, tokenizer, prompts)
    attn = (seqs != tokenizer.pad_token_id).long()

    # 2) Student forward pass over its own sequence (gradients needed)
    s_logits = student(seqs, attention_mask=attn).logits  # (B, L, V)

    # 3) Teacher scores the same sequence (frozen, no_grad)
    with torch.no_grad():
        t_logits = teacher(seqs, attention_mask=attn).logits

    # 4) Compute loss only over the generated segment (response), shifted to align next-token
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

**Commentary**: On-policy sampling is costly, so in practice one often mixes on-policy and fixed data with probability `lambda` (GKD's `lmbda`); `trl.GKDTrainer` already encapsulates this logic, and hand-written skeletons are mainly for understanding and customization.

---

### R.1.4　CoT-Weighted Loss (weighting tokens inside `<think>...</think>`)

When distilling a chain-of-thought, the thinking segment often carries more of the teacher's reasoning ability than the final answer. The following builds a per-token weight mask based on the `<think>` and `</think>` markers in the sequence, giving the thinking segment higher weight.

```python
import torch

def build_cot_weight_mask(input_ids, tokenizer, think_weight=2.0, base_weight=1.0,
                          open_tag="<think>", close_tag="</think>"):
    """Produce a weight for each token; the weight is higher within the <think>...</think> span."""
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
    """per_token_loss / weight_mask / pad_mask: (B, L); returns a scalar."""
    w = weight_mask * pad_mask  # zero out padding weights
    return (per_token_loss * w).sum() / w.sum().clamp_min(1.0)
```

**Commentary**: `per_token_loss` can be obtained from `F.cross_entropy(..., reduction="none")` or per-token KL; if you split labels with special tokens (it is recommended to add `<think>`/`</think>` to the tokenizer's special tokens), you avoid having them split into multiple subwords by BPE and missing the boundaries.

---

### R.1.5　Cross-Tokenizer Alignment (cross-tokenizer logit projection)

When teacher and student use different tokenizers, logits cannot be aligned position-by-position. A common approach: take the teacher's top-k tokens, decode them to strings, re-encode them with the student tokenizer, and redistribute the teacher's probability mass onto the corresponding (first) token in the student vocabulary. The following is an illustrative skeleton that omits a full subword-merge strategy.

```python
import torch

@torch.no_grad()
def project_teacher_to_student(teacher_logits, teacher_tok, student_tok,
                               student_vocab_size, top_k=64, T=1.0):
    """Project the teacher's single-step distribution onto the student vocabulary
       (top-k string re-encoding + mass redistribution).

    teacher_logits: (V_t,) teacher logits at a single position
    returns: (V_s,) target probability distribution over the student vocabulary
    """
    probs = torch.softmax(teacher_logits / T, dim=-1)
    topk = torch.topk(probs, k=top_k)
    student_dist = torch.zeros(student_vocab_size, device=teacher_logits.device)

    for tok_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
        # teacher token -> string -> re-encode with the student tokenizer
        piece = teacher_tok.decode([tok_id]).strip()
        if not piece:
            continue
        s_ids = student_tok.encode(piece, add_special_tokens=False)
        if not s_ids:
            continue
        # Assign mass to the first token of the student encoding
        # (simplified strategy; could instead spread it along the subword chain)
        student_dist[s_ids[0]] += p

    total = student_dist.sum()
    if total > 0:
        student_dist /= total  # renormalize, discarding the tail that cannot be aligned
    return student_dist
```

**Commentary**: This is illustrative rather than production-grade—a rigorous approach must handle spreading probability along the chain for multi-subword tokens, leading-whitespace differences (`▁`/`Ġ`), and normalization bias; ULD (Universal Logit Distillation) aligns the two vocabularies with optimal transport and can serve as a more robust alternative.

---

### R.1.6　Complete Training Step (HF Trainer-style)

Integrate the modules above into a custom `Trainer`: the teacher is frozen and set to `eval()`, and the student can be paired with PEFT LoRA to reduce VRAM; distribution is enabled with an `accelerate` config file running DeepSpeed ZeRO-3 or FSDP, with gradient checkpointing turned on.

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
        # Distillation loss (R.1.1): shift needed for next-token alignment
        s_logits = s_out.logits[:, :-1, :].contiguous()
        t_logits = t_out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = kd_loss(s_logits, t_logits, shift_labels,
                       T=self.T, alpha=self.alpha)
        return (loss, s_out) if return_outputs else loss

# --- Assembly ---
student = AutoModelForCausalLM.from_pretrained("your/small-base",
                                               torch_dtype=torch.bfloat16)
student.gradient_checkpointing_enable()         # save VRAM
student.config.use_cache = False                # mutually exclusive with gradient checkpointing
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
    deepspeed="ds_zero3.json",   # or enable FSDP via accelerate config
    logging_steps=10, num_train_epochs=3,
)
trainer = DistillTrainer(teacher_model=teacher, T=2.0, alpha=0.5,
                         model=student, args=args,
                         train_dataset=train_ds, tokenizer=tokenizer)
trainer.train()
```

Launch with `accelerate launch --config_file fsdp.yaml train.py`, or attach `deepspeed="ds_zero3.json"` directly in `TrainingArguments` (ZeRO-3 shards parameters, gradients, and optimizer states across the GPUs). If the teacher is too large, you can load it in a separate process / quantized (e.g. 8-bit) and only run forward scoring.

**Commentary**: The teacher must be set to `eval()` + `requires_grad_(False)` and wrapped in `no_grad`, otherwise VRAM and compute are wasted on gradients that are never updated; `gradient_checkpointing` and `use_cache=True` are mutually exclusive, so you must manually disable the cache to avoid warnings or it becoming ineffective.


---

## R.2　Core Implementations of KV Cache and Caching

This section gathers the core reference implementations related to the KV Cache in inference serving, covering everything from VRAM estimation, PagedAttention paging, CUDA addressing, RadixAttention prefix sharing, to FlashDecoding, KV quantization, and multi-layer cache routing. All code prioritizes readability and is grounded in semantic correctness, with naming consistent with Chapters 10 and 11 (vLLM / SGLang / FlashAttention / Triton).

---

### R.2.1　KV Cache VRAM Calculator

**Purpose**: Estimate the bytes a sequence's KV Cache occupies—the starting point for capacity planning and deriving batch ceilings.

```python
def kv_cache_bytes(
    L: int,            # number of Transformer layers (num_layers)
    kv_heads: int,     # number of KV heads (post-GQA/MQA num_key_value_heads)
    head_dim: int,     # dimension per head
    seq_len: int,      # sequence length (prompt + generated)
    batch: int = 1,    # number of concurrent sequences
    dtype_bytes: int = 2,  # fp16/bf16 = 2, fp8/int8 = 1
) -> int:
    """KV Cache VRAM = 2 (one each for K and V) * dtype_bytes
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
    print("per token:", human(per_token))                # 128.00KB

    total = kv_cache_bytes(L=32, kv_heads=8, head_dim=128,
                           seq_len=8192, batch=128, dtype_bytes=2)
    print("batch=128 x 8K:", human(total))               # ~128.00GB
```

**Commentary**: 128KB per token is a key constant for Llama-3-8B—`2 * 2 * 32 * 8 * 128 = 131072 = 128KB`. GQA cuts KV heads from 32 down to 8, which is the same as shrinking the cache by 4×; if you use MHA (kv_heads=32), per-token jumps to 512KB and batch=128×8K immediately maxes out 512GB. In practice, treat `2 * dtype_bytes * L * kv_heads * head_dim` as a "per-token constant" to compute first, then multiply by `seq_len * batch`, for quick mental estimation of the capacity ceiling.

---

### R.2.2　PagedAttention BlockManager / BlockTable

**Purpose**: Manage KV Cache VRAM with vLLM-style fixed-size blocks, eliminating the internal and external fragmentation caused by contiguous allocation.

```python
from collections import deque
from typing import Dict, List


class PhysicalTokenBlock:
    """A physical VRAM block that fixedly holds block_size tokens."""
    def __init__(self, block_number: int, block_size: int):
        self.block_number = block_number   # physical block index
        self.block_size = block_size
        self.ref_count = 0                 # how many sequences share it (for copy-on-write)

    def __repr__(self):
        return f"PhysicalTokenBlock(#{self.block_number}, ref={self.ref_count})"


class BlockAllocator:
    """Manage allocation/release of physical blocks with a free-list, O(1) allocation."""
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
    """Mapping of logical block -> physical block for a single sequence."""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.blocks: List[PhysicalTokenBlock] = []
        self.num_tokens = 0

    def append_token(self) -> None:
        block_size = self.allocator.block_size
        # Only allocate a new physical block when the last block is full (or there are no blocks yet)
        if self.num_tokens % block_size == 0:
            self.blocks.append(self.allocator.allocate())
        self.num_tokens += 1

    def physical_index(self, logical_token_idx: int) -> int:
        """Translate a logical token position into (block_number, offset)."""
        block_size = self.allocator.block_size
        block = self.blocks[logical_token_idx // block_size]
        offset = logical_token_idx % block_size
        return block.block_number, offset

    def fork(self) -> "BlockTable":
        """Copy-on-write fork for beam search / parallel sampling:
           share existing blocks, only increment ref_count."""
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

**Commentary**: The essence of PagedAttention is to cut the KV Cache into OS-page-style fixed blocks; the `BlockTable` is the "page table" and `BlockAllocator.free` is the "free-page chain." `ref_count` lets parallel sampling / beam search share the same prompt's KV (copy-on-write), and a true copy is only made when some branch writes to an already-shared block. `num_tokens % block_size == 0` is the sole condition for allocating a new block—this is the key to paging away memory fragmentation and pushing utilization to 90%+. When `num_free == 0` you must raise and let the scheduler go through preemption (swap-out or recompute); never fail silently.

---

### R.2.3　paged_attention CUDA Kernel Addressing Core

**Purpose**: Illustrate how a CUDA kernel internally translates logical tokens into physical VRAM pointers via the block_table, then reads K/V.

```cuda
// For a single query head, sweep over all KV tokens of a sequence.
// block_table is this sequence's "page table": logical block idx -> physical block number.
// kv_cache layout: [num_blocks, num_kv_heads, head_dim, block_size]
template <typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    scalar_t*       __restrict__ out,          // [num_heads, head_dim]
    const scalar_t* __restrict__ q,            // [num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,      // physical KV pool
    const scalar_t* __restrict__ v_cache,
    const int*      __restrict__ block_table,  // [max_blocks_per_seq]
    const int       context_len,               // current number of tokens in this sequence
    const int       num_kv_heads) {

    const int kv_head = blockIdx.y;            // the KV head this thread block handles
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int logical_block = 0; logical_block < num_blocks; ++logical_block) {
        // 1) Page-table lookup: logical block -> physical block number
        const int physical_block = block_table[logical_block];

        // 2) Number of tokens to process in this block (the last block may be partial)
        const int tokens_in_block =
            min(BLOCK_SIZE, context_len - logical_block * BLOCK_SIZE);

        for (int t = 0; t < tokens_in_block; ++t) {
            // 3) Compute the physical offset: locate the contiguous head_dim segment of (physical_block, kv_head)
            const int64_t base =
                (((int64_t)physical_block * num_kv_heads + kv_head)
                 * HEAD_DIM) * BLOCK_SIZE;

            const scalar_t* k_ptr = k_cache + base + t;  // K[..., :, t]
            const scalar_t* v_ptr = v_cache + base + t;

            // 4) dot(q, k) -> score, accumulate into online-softmax (omitted here)
            //    The real implementation pairs this with a running max / running sum for a
            //    numerically stable online softmax, and accumulates score * v into out.
            (void)k_ptr; (void)v_ptr;
        }
    }
}
```

**Commentary**: Compared with traditional attention which assumes the KV is "contiguous" in VRAM, the paged kernel adds a layer of indirect addressing via `block_table[logical_block]`—this is exactly the cost and the essence of how PagedAttention assembles a logically contiguous sequence out of non-contiguous physical pages. The layout `[num_blocks, num_kv_heads, head_dim, block_size]` puts the contiguous tokens of the same head on the innermost dimension, letting threads within a warp read in a coalesced manner. The real vLLM kernel uses online-softmax (running max + running sum) in the inner loop to avoid storing the full score matrix; this sketch focuses on the "page-table lookup → physical offset" addressing main line.

---

### R.2.4　RadixAttention Prefix Tree

**Purpose**: Share the KV Cache of identical prefixes (system prompt, few-shot, multi-turn history) with an SGLang-style radix tree, achieving automatic cross-request cache reuse.

```python
import time
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """An edge stores a token sequence; a node holds the KV cache pointers (block list) for that prefix."""
    def __init__(self):
        self.children: Dict[int, "RadixNode"] = {}  # keyed by the first token
        self.key: List[int] = []                    # token segment from this node to its parent
        self.kv_blocks: List[int] = []              # corresponding KV physical block ids
        self.parent: Optional["RadixNode"] = None
        self.last_access = time.monotonic()         # for LRU


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
        """Return the longest already-cached prefix's KV blocks, and the node it stops at."""
        node, matched, i = self.root, [], 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            m = self._match_len(child.key, tokens[i:])
            node_kv = child.kv_blocks[:m]
            matched.extend(node_kv)
            if m < len(child.key):     # partial match, stops in the middle of the edge
                return matched, child
            node, i = child, i + m
            node.last_access = time.monotonic()
        return matched, node

    def insert(self, tokens: List[int], kv_blocks: List[int]) -> None:
        """Insert a complete sequence; automatically split nodes at the divergence point."""
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
            if m < len(child.key):     # divergence on the edge -> split out an intermediate node
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
        """LRU eviction: only evict leaf nodes, keep the root; return reclaimed KV blocks to the allocator."""
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

**Commentary**: RadixAttention generalizes "prefix caching" into a tree—any two requests, as long as they share a prefix (not limited to the system prompt), can hit existing KV blocks in `match_prefix` and skip recomputation. `_split` is the soul of the radix tree: when a new sequence diverges midway on an edge, that edge must be split into a "shared segment + two branches." The eviction policy deliberately touches only leaf nodes and preserves the root, ensuring that hot prefixes referenced by many requests (usually near the root) are not mistakenly evicted. SGLang measurements show that in multi-turn dialogue and few-shot scenarios the hit rate can reach 50%+, with a significant drop in TTFT.

---

### R.2.5　FlashDecoding split-KV reduction

**Purpose**: Illustrate how FlashDecoding, during the decode phase, splits an ultra-long KV along the sequence dimension into multiple chunks computed in parallel, then merges the partial results with log-sum-exp, improving GPU occupancy for long sequences.

```python
import numpy as np


def flash_decoding(q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   num_splits: int = 4):
    """q: [d], K/V: [seq, d]. Split the KV along seq into num_splits chunks computed in parallel,
       each doing a numerically stable softmax, finally merged with log-sum-exp."""
    seq, d = K.shape
    scale = 1.0 / np.sqrt(d)
    chunk = (seq + num_splits - 1) // num_splits

    partial_out = []   # weighted-V accumulation per chunk (already divided by the chunk's sum)
    partial_lse = []   # log-sum-exp per chunk (with running-max correction)

    # --- Each split computed independently (corresponds to a different thread block on the GPU, parallelizable) ---
    for s in range(num_splits):
        lo, hi = s * chunk, min((s + 1) * chunk, seq)
        if lo >= hi:
            continue
        scores = (K[lo:hi] @ q) * scale          # [chunk]
        m = scores.max()                          # running max within the chunk
        p = np.exp(scores - m)                    # stabilization
        l = p.sum()                               # denominator within the chunk
        o = (p @ V[lo:hi]) / l                    # weighted average within the chunk
        partial_out.append(o)
        partial_lse.append(m + np.log(l))         # log-sum-exp

    # --- reduction: re-weight and merge across splits with log-sum-exp ---
    lses = np.array(partial_lse)
    global_max = lses.max()
    weights = np.exp(lses - global_max)           # weight of each chunk (= each chunk's true denominator proportion)
    weights /= weights.sum()
    out = sum(w * o for w, o in zip(weights, partial_out))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, seq = 128, 8192
    q = rng.standard_normal(d)
    K = rng.standard_normal((seq, d))
    V = rng.standard_normal((seq, d))

    # Compare against single-chunk reference softmax to verify numerical consistency
    ref_scores = (K @ q) / np.sqrt(d)
    ref = (np.exp(ref_scores - ref_scores.max()) @ V) / \
          np.exp(ref_scores - ref_scores.max()).sum()
    fd = flash_decoding(q, K, V, num_splits=8)
    print("max abs diff:", np.abs(ref - fd).max())   # ~1e-15
```

**Commentary**: In the decode phase the batch is often very small (even =1), with a single query against a very long KV; a traditional kernel can only keep a few SMs working while the GPU sits severely idle. FlashDecoding makes one more cut along the KV's seq dimension, letting multiple SMs sweep different segments in parallel, then correctly merges each segment's partial softmax with log-sum-exp—the key is that each segment carries its own `m` (running max) and `lse`, and at merge time is re-weighted by `exp(lse_i - global_max)`, mathematically exactly equivalent to a single-chunk softmax (error ~1e-15 in the example above). Most of the decode-throughput gain in long-context inference comes from this.

---

### R.2.6　KV Cache Quantization (FP8 / INT8 per-token)

**Purpose**: Quantize K/V per-token (INT8 or FP8) before writing to the KV Cache and dequantize on read, halving cache VRAM and bandwidth with minimal precision loss.

```python
import numpy as np


def quantize_kv_int8(x: np.ndarray):
    """Per-token symmetric quantization: x [num_tokens, head_dim] -> int8 + per-token scale.
       Each token uses its own absmax to derive the scale, avoiding outlier tokens dragging down the whole."""
    absmax = np.abs(x).max(axis=-1, keepdims=True)     # [num_tokens, 1]
    scale = absmax / 127.0
    scale = np.maximum(scale, 1e-8)                    # prevent division by zero
    q = np.round(x / scale).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_kv_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Restore back to fp16/fp32 on read for the attention computation."""
    return q.astype(np.float32) * scale


def quantize_kv_fp8(x: np.ndarray):
    """FP8 (E4M3) per-token: use scale to bring values into the E4M3 dynamic range (±448) before storing.
       Simulated here with numpy; in practice Triton/cuda writes torch.float8_e4m3fn."""
    FP8_MAX = 448.0
    absmax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.maximum(absmax / FP8_MAX, 1e-8)
    scaled = np.clip(x / scale, -FP8_MAX, FP8_MAX)
    # Simulate E4M3's limited mantissa: quantize the mantissa bits with round-to-nearest
    q = _round_to_e4m3(scaled)
    return q, scale.astype(np.float32)


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    exp = np.floor(np.log2(np.maximum(ax, 1e-12)))
    mant_step = np.power(2.0, exp - 3)        # E4M3: 3 mantissa bits
    return sign * np.round(ax / mant_step) * mant_step


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = rng.standard_normal((16, 128)).astype(np.float32)

    q, s = quantize_kv_int8(k)
    err = np.abs(k - dequantize_kv_int8(q, s)).mean()
    print("INT8 mean abs err:", err)          # ~1e-3, VRAM halved
```

**Commentary**: KV Cache quantization is a high-value-for-money means of "trading a little precision for VRAM and bandwidth"—INT8 halves it directly, and FP8 (E4M3) has native hardware support on Hopper/Ada. Per-token quantization (one scale per token) is far more robust than per-tensor, because attention often has a few outlier tokens with especially large values, and sharing one scale sacrifices the quantization resolution of the other tokens. In practice K is more quantization-sensitive than V (K is exponentially amplified going into softmax), so a common mixed strategy is "FP8 for K, INT8 for V" or retaining higher precision for K.

---

### R.2.7　Cache-aware routing (consistent prefix-hash routing)

**Purpose**: In a multi-node inference cluster, hash the request prefix (system prompt + history), use consistent hashing to pick a node, and stably route requests with the same prefix to the same node to maximize RadixAttention / prefix-cache hits.

```python
import bisect
import mmh3   # MurmurHash3, fast and evenly distributed


class ConsistentHashRouter:
    """Consistent-hash ring + virtual nodes, to avoid large-scale cache invalidation when nodes are added/removed."""
    def __init__(self, nodes, vnodes: int = 160):
        self.ring = {}           # hash -> node
        self.sorted_keys = []
        self.vnodes = vnodes
        for n in nodes:
            self.add_node(n)

    def _hash(self, key: str) -> int:
        # Take a 32-bit unsigned hash as the coordinate on the ring
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
        """Pick a node by prefix hash: the same prefix always lands at the same spot on the ring -> same node."""
        prefix_key = system_prompt + "\x00" + history
        h = self._hash(prefix_key)
        idx = bisect.bisect_right(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


if __name__ == "__main__":
    router = ConsistentHashRouter(["gpu-0", "gpu-1", "gpu-2", "gpu-3"])
    sys_p = "You are a helpful assistant."
    print(router.route(sys_p, "user: hi"))      # same prefix stably hits the same node
    print(router.route(sys_p, "user: hi"))       # same node as the line above
```

**Commentary**: A prefix cache is only meaningful when "the same prefix hits the same node," otherwise each node computes its own copy of the KV and the hit rate drops to zero. The value of consistent hashing (with virtual nodes) is: when nodes scale up or down, only `1/N` of the keys need remapping rather than a full reshuffle—avoiding one machine going online/offline invalidating the whole cluster's cache. `mmh3.hash` is faster than Python's built-in `hash` and stable across processes (the built-in hash has randomization). In practice the SGLang router mixes in a load-balancing factor on top of the pure prefix hash (a trade-off between cache hits and node pressure), to avoid a popular prefix overwhelming a single node.

---

### R.2.8　Semantic Cache (GPTCache-style)

**Purpose**: In GPTCache style, embed queries and store them in a vector database; if a new query, via ANN retrieval, is semantically close enough (cosine similarity > threshold), return the cached answer directly, saving an LLM inference.

```python
import numpy as np
# Any embedding model + any vector store (Qdrant shown here; Milvus also works)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class SemanticCache:
    def __init__(self, embed_fn, collection="llm_cache",
                 dim=768, threshold=0.98):
        self.embed_fn = embed_fn          # query -> np.ndarray[dim]
        self.threshold = threshold        # cosine threshold; 0.98 is conservative to avoid false hits
        self.collection = collection
        self.client = QdrantClient(":memory:")   # in production, connect to a Qdrant/Milvus service
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
        if hits and hits[0].score >= self.threshold:   # COSINE distance is already a similarity
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
            return cached                  # hit: save one LLM inference
        answer = llm_call(query)           # miss: actually call the LLM
        self.put(query, answer)
        return answer


if __name__ == "__main__":
    def fake_embed(q: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(q)) % (2**32))
        v = rng.standard_normal(768)
        return (v / np.linalg.norm(v)).astype(np.float32)

    cache = SemanticCache(fake_embed)
    print(cache.ask("What is the capital of France?",
                    lambda q: "Paris"))    # miss -> call the LLM
    # Semantically similar phrasings can hit under the threshold setting
    # (random embeddings used here only to illustrate the flow)
```

**Commentary**: The semantic cache is at a different level from the KV/prefix caches above—it does not reuse the KV, but directly reuses the "final answer," saving an entire inference on a hit; it is the most cost-effective but also the highest-risk. `threshold` is the core knob: too low (e.g. 0.9) will misjudge "capital of France" and "capital of Germany" as synonyms and return a wrong answer; 0.98 is conservative, suited to high-certainty scenarios like FAQ / document Q&A and not to dialogue requiring precise distinction of details. In production deployment, the embedding model must be aligned with the business corpus, the vector store (Milvus / Qdrant) needs TTL and capacity eviction configured, and hit results need sampled quality monitoring to avoid cache poisoning.


---

## R.3　Core Implementations of Quantization, Inference Deployment, and Evaluation

This section gathers the key implementations from "post-training quantization" to "going live in production" to "automated evaluation." All code presumes open-weight models and openly licensed data, and can serve directly as an engineering reference. Each snippet comes with a one-line purpose note and a short **Commentary**.

---

### R.3.1　QLoRA Fine-Tuning Configuration

> Purpose: Load the base model with 4-bit NF4 quantization, train only a low-rank LoRA adapter, and complete large-model fine-tuning on a single GPU.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit NF4 double quantization: significantly reduces weight VRAM, dequantized to bf16 for compute
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

# Enable gradient checkpointing + preprocess for k-bit training (cast layernorm to fp32, enable input grads)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,                       # alpha/r = 2, a stable equivalent scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                       # cover both attention and MLP projection groups
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()         # trainable params are usually < 2%
```

**Commentary**: Attaching `target_modules` to both q/k/v/o and gate/up/down is the practical realization of the QLoRA paper's "add LoRA to all linear layers," and converges more stably than attaching only to q/v. `gradient_checkpointing` and NF4 are key to training 7B–13B on a single GPU; note that you must call `get_peft_model` after `prepare_model_for_kbit_training`, and set `gradient_checkpointing_kwargs={"use_reentrant": False}` during training for compatibility with newer PyTorch.

---

### R.3.2　GPTQ / AWQ Post-Training Quantization

> Purpose: Do 4-bit weight quantization with a small calibration corpus, outputting quantized weights ready for inference.

```python
# --- GPTQ (gptqmodel, the successor maintained version of autogptq) ---
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

calib = load_dataset("allenai/c4", "en", split="train", streaming=True)
calib = [next(iter(calib))["text"] for _ in range(256)]   # 128~512 calibration samples suffice

quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True)
model = GPTQModel.load("meta-llama/Llama-3.1-8B", quant_cfg)
model.quantize(calib)
model.save("./Llama-3.1-8B-gptq-int4")
```

```python
# --- AWQ (autoawq): activation-aware, per-channel protection of salient weights ---
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path)

quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
model.quantize(tok, quant_config=quant_config)        # built-in default calibration set; you can also pass calib_data
model.save_quantized("./Llama-3.1-8B-awq-int4")
tok.save_pretrained("./Llama-3.1-8B-awq-int4")
```

**Commentary**: GPTQ minimizes reconstruction error layer-by-layer via the Hessian; `desc_act=True` (act-order) usually lowers perplexity further but quantizes slightly slower. AWQ does not update weights, only applies scaling protection—it quantizes fast and is friendly to instruction models. `group_size=128` is the common precision/size trade-off; both outputs can be loaded directly by vLLM (`--quantization gptq_marlin` / `awq_marlin`).

---

### R.3.3　GGUF Conversion and k-quant

> Purpose: Convert HF weights to llama.cpp's GGUF format, then apply k-quant compression for CPU/edge inference.

```bash
# 1) HF -> GGUF (first convert to full-precision fp16/bf16, then quantize)
python convert_hf_to_gguf.py ./Llama-3.1-8B \
    --outfile model-f16.gguf \
    --outtype f16

# 2) k-quant: Q4_K_M is the most commonly used quality/size balance point
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# 3) (optional) use an importance matrix to improve low-bit quality
./llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ4_XS.gguf IQ4_XS
```

**Commentary**: `Q4_K_M` is the sweet spot of size and precision for most 7B–70B; for smaller, use `Q3_K_M`, for closer to the original use `Q5_K_M` or `Q6_K`. Below 4-bit (e.g. `IQ4_XS`/`IQ3`) you must pair with an imatrix, otherwise degradation is obvious. Conversion failures are usually due to the chat template or a new architecture not being supported, requiring a llama.cpp update.

---

### R.3.4　Speculative Decoding

> Purpose: Use a small draft model to propose multiple tokens at once, verified in parallel by the large model, reducing latency without changing the output distribution.

```python
# vLLM: draft-model speculative decoding (newer versions pass it via speculative_config)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",  # a small model from the same tokenizer family
        "num_speculative_tokens": 5,
    },
    tensor_parallel_size=4,
)
out = llm.generate(["解釋投機解碼的原理。"], SamplingParams(temperature=0.0, max_tokens=256))
```

```bash
# llama.cpp: main model + draft model
./llama-server \
    -m  model-70B-Q4_K_M.gguf \
    -md model-1B-Q4_K_M.gguf \
    --draft-max 8 --draft-min 1 \
    -ngl 99 -c 8192
```

**Commentary**: The draft model must share a vocab with the main model (same family is most stable). The speedup depends on the "acceptance rate," with the greatest gains on low-temperature, structured, or highly repetitive outputs; high-temperature creative generation has a low acceptance rate and may not be worthwhile. Too large a `num_speculative_tokens` wastes verification compute instead; 5–8 is a common starting point.

---

### R.3.5　vLLM Service Launch

> Purpose: Launch a high-throughput inference service with an OpenAI-compatible API, and tune VRAM and long context.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name llama3.1-8b \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --max-num-seqs 256
# You can then point the OpenAI SDK at http://localhost:8000/v1
```

**Commentary**: `--gpu-memory-utilization` determines the KV cache reservation; 0.85–0.92 depends on other loads on the same card, and too high easily causes OOM. `--enable-chunked-prefill` interleaves long prompts with decode, reducing TTFT jitter. `--kv-cache-dtype fp8` saves roughly half the KV VRAM in exchange for minimal precision loss—a high-value-for-money means of expanding concurrency or context; `--max-model-len` must never exceed the length the weights actually support.

---

### R.3.6　distilabel Synthetic Distillation Data

> Purpose: Use an "open-weight" teacher model to batch-generate instruction responses, producing a distillation dataset usable for SFT.

```python
# Teacher is an open-weight model (Llama-3.3-70B), served by a local vLLM —— compliant, within commercially licensed scope
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.models import OpenAILLM   # points to a self-hosted vLLM OpenAI-compatible endpoint

teacher = OpenAILLM(
    model="llama3.3-70b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
)

with Pipeline(name="distill-sft") as pipeline:
    load = LoadDataFromHub(repo_id="HuggingFaceH4/instruction-dataset", split="test")
    gen = TextGeneration(llm=teacher, input_batch_size=16)   # generate the generation column from the instruction column
    keep = KeepColumns(columns=["instruction", "generation"])
    load >> gen >> keep

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=True)
    distiset.push_to_hub("your-org/distilled-sft")
```

**Commentary**: The teacher must be a model with **open weights or a license that permits distillation** (such as the open-source versions of Llama / Qwen / Mistral); never scrape or bypass protected commercial APIs, nor use it for de-censorship. `use_cache=True` lets you resume after an interruption and saves compute; it is recommended to pass the output through another layer of quality/dedup filtering (such as distilabel's evol/quality scoring steps) before SFT.

---

### R.3.7　lm-eval-harness Evaluation and Long Context

> Purpose: Run downstream benchmarks with the industry-standard harness to quantify model-quality regression.

```bash
# Standard subject-knowledge and math-reasoning benchmarks
lm_eval --model hf \
    --model_args pretrained=./Llama-3.1-8B-awq-int4,dtype=bfloat16 \
    --tasks mmlu,gsm8k,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path ./eval_results

# Evaluate the vLLM service directly (higher throughput)
lm_eval --model local-completions \
    --model_args base_url=http://localhost:8000/v1/completions,model=llama3.1-8b \
    --tasks gsm8k --batch_size 16
```

**Long-context supplement**: MMLU/GSM8K cannot reflect long-document retrieval ability; you need to separately run **RULER** (a variety of synthetic tasks, with multiple lengths from 4k–128k specifiable) and the **NIAH (Needle-in-a-Haystack)** test, observing the recall-decay curve as context grows; KV cache quantization (such as the fp8 in R.3.5) or RoPE extrapolation settings must be regression-verified on these two.

**Commentary**: Before a quantized model goes live, at minimum compare it against the original model on GSM8K (reasoning) and MMLU (knowledge); a 1–3 point drop for 4-bit is generally acceptable; a large degradation is usually due to an inappropriate quantization calibration set or group_size. Fix `--num_fewshot` and the version so scores are comparable across runs.

---

### R.3.8　Ollama Modelfile Packaging

> Purpose: Package GGUF weights together with inference parameters and the chat template into a one-click distributable Ollama model.

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

**Commentary**: `TEMPLATE` must match that base's chat format (here, Llama-3's header markers); a wrong template makes the model answer off-topic or fail to stop. `stop` and `num_ctx` must be consistent with training/quantization; the Modelfile makes "weights + parameters + template" a single versionable artifact, greatly aiding team reproducibility.

---

### R.3.9　Dual-Model Dynamic Routing

> Purpose: Use a complexity heuristic to route simple requests to the small model and hard requests to the large model, balancing cost and quality.

```python
# Lightweight routing: estimate complexity by length/keywords, route to a cheap or powerful model
# (any OpenAI-compatible endpoint works)
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re

app = FastAPI()
cheap = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # small-model vLLM
big   = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")   # large-model vLLM

HARD_HINTS = re.compile(r"(證明|推導|程式|debug|逐步|多步|規劃|數學|code|prove)", re.I)

class Query(BaseModel):
    prompt: str

def complexity_score(text: str) -> float:
    score = 0.0
    score += min(len(text) / 800, 1.0)          # length signal
    score += 0.5 if HARD_HINTS.search(text) else 0.0  # task-type keywords
    score += 0.3 if text.count("\n") >= 4 else 0.0    # multi-paragraph/structured input
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

**Commentary**: Heuristic routing has zero extra latency and is easy to explain—it is the best starting point in the early days of going live; once you have accumulated traffic, you can switch to "scoring with a small model" or a dedicated router model (such as RouteLLM) for learned routing. Be sure to set up a **fallback**—downgrade to the small model when the large model times out or returns 5xx, and log the routing decisions for offline comparison of the misroute rate. If you use LiteLLM, you can plug the two endpoints above into its Router and hand off to the built-in load balancing and fallback management.


---
