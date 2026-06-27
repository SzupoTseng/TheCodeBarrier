# 附录B　名词与工具速查

> 把全书出现的技术名词，按主题收拢成一张「解码器」。这些**全是真实、公开的技术**，与附录 A 那些「待核实的模型名」不同——掌握它们，你就能听懂任何蒸馏／本地部署的宣传到底在说什么。

## B.1　蒸馏内核概念

| 名词 | 一句话解释 | 出现章节 |
|---|---|---|
| **知识蒸馏（Knowledge Distillation, KD）** | 让大教师模型把能力传给小学生模型 | 全书 |
| **教师 / 学生（Teacher / Student）** | 被学的大模型 / 来学的小模型 | 第 4、5 章 |
| **硬标签（Hard Target）** | 只学最终答案 | 第 4 章 |
| **软标签 / Logits（Soft Target）** | 学教师的完整几率分布（「暗知识」）| 第 4 章 |
| **KL 散度（KL-Divergence）** | 衡量两个几率分布差多远的数学工具 | 第 4、5 章 |
| **思维链（Chain-of-Thought, CoT）** | 模型一步步推理的过程 | 第 2、4、5 章 |
| **CoT 对齐损失** | 给思考过程加权，逼学生「想清楚再答」 | 第 4、5 章 |
| **轨迹反转（Trace Inversion）** | 打乱/倒序推理步骤，防学生死背 | 第 2、4 章 |
| **语义改写（Synthetic Paraphrasing）** | 高温采样改写数据，扩泛化、防过拟合 | 第 4、5 章 |
| **过度拟合（Overfitting）** | 死背训练数据、换场景就崩 | 第 4 章 |
| **abliteration** | 消除权重中「拒绝」方向，做去审查 | 第 2 章 |

## B.2　训练与微调

| 名词 | 解释 |
|---|---|
| **SFT（指令微调）** | 用「指令→回应」对监督微调 |
| **QLoRA** | 量化 + 低秩适配的高效微调法（省显存）|
| **全参数微调（Full FT）** | 更新模型所有权重（最彻底、最耗资源）|
| **DeepSpeed ZeRO-3** | 把权重/梯度/优化器状态分片，省内存跑大模型 |
| **混合精度（BF16/FP16）** | 用较低精度加速训练、兼顾稳定 |
| **梯度裁剪 / 余弦退火 / Early Stopping** | 三种防训练崩坏、稳定收敛的护栏 |
| **YaRN + RoPE Base 扩展** | 把模型上下文窗口从 8K 渐进拉到 1M 的技术 |
| **FlashAttention-3** | 高效注意力计算，省内存、加速长串行 |

## B.3　量化与部署

| 名词 | 解释 |
|---|---|
| **量化（Quantization）** | 把权重从高精度压到低精度，换取体积/速度 |
| **GGUF** | 本地部署主流模型格式（llama.cpp/Ollama/LM Studio）|
| **Q4_K_M / Q5_K_M / Q8_0** | 常见量化等级；数字大=精确/大，K_M=混合精度 |
| **混合量化** | 注意力层保高精度、MLP 层压低精度 |
| **KV Cache** | 上下文内存；长文本的真正瓶颈 |
| **KV Cache 量化（FP8/INT8）** | 压缩上下文内存，释放长文本潜能 |
| **MoE（混合专家）** | 多专家、每次只激活少数；带宽受限机器的救星 |
| **激活参数 vs 总参数** | 前者决定速度，后者决定智商上限 |

## B.4　硬件（NVIDIA DGX Spark）

| 名词 | 解释 |
|---|---|
| **DGX Spark** | NVIDIA 的桌面型「个人 AI 超级电脑」|
| **GB10（Grace Blackwell）** | DGX Spark 搭载的超级芯片 |
| **UMA（统一内存架构）** | CPU 与 iGPU 共享同一池内存 |
| **128GB LPDDR5X（~600 GB/s）** | 容量巨大、但带宽不及独显 HBM/GDDR |
| **容量 vs 带宽铁律** | 容量决定「能不能跑」，带宽决定「跑多快」|
| **numactl --interleave=all / mlock** | 锁内存、交错信道，榨干 UMA 带宽 |

## B.5　训练、微调与蒸馏工具

| 工具 | 角色 | 章节 |
|---|---|---|
| **Hugging Face `transformers` / `datasets`** | 模型与数据的基础库 | 第 2、4、5 章 |
| **`peft`** | LoRA / QLoRA 等参数高效微调 | 第 2、5 章 |
| **`trl`（SFTTrainer / GKDTrainer / DPO）** | 监督微调、在线蒸馏、偏好对齐 | 第 2、4 章 |
| **Axolotl / LLaMA-Factory / Unsloth** | 开箱即用的微调框架（Unsloth 省显存）| 第 1、2 章 |
| **DeepSpeed（ZeRO-3）/ FSDP / `accelerate`** | 分布式训练、权重分片 | 第 4、5 章 |
| **`bitsandbytes`（NF4）** | QLoRA 量化基座、paged optimizer | 第 2、4 章 |
| **FlashAttention-3** | 高效注意力，长串行省内存 | 第 3、5 章 |
| **distilabel（Argilla）** | 合成数据 / 语义改写 / CoT 生成 | 第 4、5 章 |
| **mergekit** | LoRA/权重融合与模型合并 | 第 5 章 |
| **MiniLLM / GKD（实作）** | on-policy / reverse-KL 蒸馏 | 第 4 章 |
| **TransformerLens / heretic / refusal_direction** | abliteration（去审查）激活分析 | 第 2 章 |
| **Microsoft Presidio / spaCy NER** | 数据去隐私（PII 去识别）| 第 5 章 |
| **Redis 集群 + Apache Arrow** | 高吞吐零拷贝数据缓存层 | 第 5 章 |
| **Prometheus + Grafana** | 硬件/训练遥测监控 | 第 5 章 |

## B.6　量化工具

| 工具 | 角色 |
|---|---|
| **llama.cpp（GGUF / K-quants / `quantize`）** | 主流本地量化与推理 |
| **imatrix（importance matrix）** | 量化前校准，同体积更高品质 |
| **AutoGPTQ / AutoAWQ** | GPTQ / AWQ 训练后量化 |
| **ExLlamaV2** | 高速量化推理引擎 |
| **FP8 / INT8 KV-Cache 量化** | 压缩上下文内存（vLLM、llama.cpp）|

## B.7　本地推理与部署

| 工具 | 角色 |
|---|---|
| **Ollama** | 本地模型管理/推理后端，支持多模型动态加载 |
| **LM Studio** | 图形化本地推理前端 |
| **llama.cpp / vLLM / SGLang** | 推理引擎（vLLM/SGLang 高吞吐、支持投机解码）|
| **NVIDIA NIM / DGX OS stack** | DGX Spark 内置推理堆栈 |
| **Open WebUI / Cursor / Continue** | 前端接口，支持模型切换与「双模对决 Arena」|
| **OpenHands（OpenDevin）/ Aider** | 开源 Coding Agent，可对接本地模型 |

## B.8　评估基准

| 基准 | 测什么 |
|---|---|
| **MMLU-Pro** | 多领域知识与推理 |
| **HumanEval** | 代码生成能力 |
| **GSM8K** | 数学应用题推理 |
| **lm-evaluation-harness（EleutherAI）** | 统一跑各基准的工具 |
| **RULER / Needle in a Haystack** | 长上下文检索能力（验证「读得懂多长」）|
