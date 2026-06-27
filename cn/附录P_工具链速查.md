# 附录P　开源蒸馏与对齐工具链速查

> 把第 4、5 章谈的「软标签对齐、偏好优化、分页 KV、模型合并」等理论，对应到**真实、公开、可在 GitHub / Hugging Face 查证**的开源工具。本附录的蒸馏、推理、合并、评测四节（P.1–P.3）为**完全合规**的工程工具，给足技术深度；P.4 的「去审查（abliteration）」工具属**双用途（dual-use）**，本书只做**防御性风险揭露**，不提供任何操作步骤、CLI、设置档——红线一律回指**附录E**与**第 6 章**。

---

## P.1　知识蒸馏与微调工具

把教师（大模型）的能力压进学生（小模型），实务上分三条路线：**SFT（学硬标签）→ 在线蒸馏（学 logits／on-policy）→ 偏好对齐（DPO/ORPO/GRPO）**。下表的工具多可互相对接（同一份 `transformers` 权重 + `datasets` 语料）。

| 工具 | 内核技术 | 对接模块 | 硬件开销 | 优点 | 风险 / 注意 |
|---|---|---|---|---|---|
| **LLaMA-Factory** | 一体化微调：SFT / DPO / ORPO / **GRPO**、LoRA/QLoRA/Full FT、Sequence Packing | `transformers`+`peft`+`trl`+`deepspeed`，WebUI/YAML | 中→极高（Full FT 需 ZeRO-3 分片） | 配置化、算法覆盖最全、上手快 | 抽象层厚，出错时难下沉调试；学习率失控易 NaN |
| **axolotl** | YAML 驱动微调框架，支持多数据格式、packing、RoPE 扩展 | `transformers`+`accelerate`+`deepspeed` | 中→高 | 社群配方多、可复现性好 | 版本与依赖漂移快，需锁版本 |
| **Unsloth** | 手写 Triton kernel + 融合算子，QLoRA 省显存提速 | `transformers`+`peft`+`bitsandbytes` | 低（单卡友善） | 显存砍半、速度 2× 级，消费级显卡可跑 | 支持的模型架构有限；多卡扩展弱于 DeepSpeed |
| **HF `trl`** | `SFTTrainer` / `DPOTrainer` / **`GKDTrainer`**（广义知识蒸馏、on-policy reverse-KL） | `transformers`+`accelerate`+`peft` | 中 | 官方维护、与生态无缝、`GKDTrainer` 是「真蒸馏」入口 | 偏底层，需自己组 pipeline |
| **DeepSpeed（ZeRO-3）/ FSDP** | 把权重／梯度／优化器状态**分片**到多卡或 CPU/NVMe | 上述框架的后端 | 解锁大模型 Full FT | 用有限显存跑超大模型；ZeRO-Offload 救小显存 | 通信开销大，带宽不足时被网络绑死；配置敏感 |
| **distilabel（+ Argilla）** | 合成数据 / CoT 生成 / 语义改写 / 偏好对标注 pipeline | LLM API 或本地推理引擎 | 低（数据端） | 把「造蒸馏语料」标准化、可审计、可去重 | 合成数据品质＝下游天花板，需人工抽查＋去污染 |

**点评**

- **LLaMA-Factory**：开源微调生态目前事实上的「全家桶」。它的价值在于把 DPO/ORPO/GRPO 这些公式凶险的对齐算法封成 YAML，让你专注在**数据偏好本身**而非梯度数值稳定性。代价是抽象层厚——一旦 loss 飞了，往往得绕过框架去看 `trl` 底层。**第一次做蒸馏选它，要深入调试时下沉到 `trl`。**
- **`trl` 的 `GKDTrainer`**：全书谈「学生学教师完整几率分布（软标签／暗知识）」，落到代码就是它。相较纯 SFT 只学硬标签，GKD 是 **on-policy + reverse-KL**，让学生在自己生成的轨迹上对齐教师，能显著缓解过度拟合（死背）。**想做「真蒸馏」而非「拿教师输出当 SFT 语料」，这是正门。**
- **Unsloth**：消费级显卡的救星，靠融合 kernel 把 QLoRA 显存砍半。但它**不是分布式方案**——要 Full FT 70B 级还是得回到 DeepSpeed/FSDP。定位是「单卡/双卡把 LoRA 微调做到极致」。
- **distilabel**：蒸馏成败的隐形胜负手在数据。它把合成语料生成做成可重现 pipeline，配 Argilla 做人工审查。**记住：合成数据的品质就是学生的智商上限，且务必对训练集做基准泄漏（benchmark contamination）检查。**

---

## P.2　推理与部署工具

蒸馏／微调完成后，要把模型「跑起来」并榨干吞吐。瓶颈通常不是算力而是**内存带宽与 KV-Cache 碎片**（见附录 B.4「容量 vs 带宽铁律」）。

| 工具 | 内核技术 | 适用场景 | 硬件开销 | 优点 | 风险 / 注意 |
|---|---|---|---|---|---|
| **vLLM** | **PagedAttention**（KV-Cache 分页、零碎片）+ continuous batching + **Speculative Decoding** | 高并发 GPU 服务 | 高（须常驻权重；投机解码还要载 draft 模型） | 吞吐业界标竿、零 padding、OpenAI 兼容 API | 绑 CUDA kernel，跨硬件可携性差；长 context 仍吃显存 |
| **TGI（Text-Generation-Inference）** | 连续批量 + 张量并行 + 量化推理，生产级服务化 | 云端/企业端点部署 | 高 | 开箱即用的 HTTP 服务、可观测性好、HF 官方 | 功能演进与授权条款变动需留意；客制 kernel 不如 vLLM 多 |
| **llama.cpp** | **GGUF** 格式 + K-quants + CPU/Metal/CUDA 全平台、imatrix 校准 | 本地端／边缘／Apple Silicon | 低（量化后可纯 CPU） | 跨平台、易部署、量化生态最成熟 | 单请求吞吐不如 vLLM；高并发非其强项 |
| **SGLang** | RadixAttention（前缀 KV 重用）+ 结构化输出快速解码 | 多轮/共享前缀/Agent 工作流 | 高 | 前缀重用对 few-shot/Agent 提速明显 | 较新、生态与文档仍在追赶 vLLM |
| **TensorRT-LLM** | NVIDIA 官方，AOT 编译 engine + in-flight batching + FP8/INT4 | 榨干单一 NVIDIA GPU 极致延迟/吞吐 | 高 | NVIDIA 硬件上常为性能上限 | 须预编译 engine、绑死 NVIDIA、可携性最差、上手陡 |
| **Ollama** | llama.cpp 上层封装 + Modelfile + 一键拉模型 | 本地快速试玩/桌面集成 | 低 | 安装即用、API 友善、生态广 | 底层即 llama.cpp，高并发/精调控不如 vLLM 与 llama.cpp 原生 |

**点评**

- **vLLM**：本附录「部署」一节的内核。PagedAttention 借鉴操作系统的虚拟分页，把 KV-Cache 切成非连续实体 block，消灭传统预分配的内存空洞（可省下大量浪费），这是它高并发的根本。**Speculative Decoding**（小 draft 模型猜、大 target 模型一次验证多 token）能进一步压低延迟。代价是绑死 CUDA kernel，跨平台迁移痛苦。
- **llama.cpp**：与 vLLM 互补。vLLM 主打**服务器高并发**，llama.cpp 主打**本地单机可携性**——GGUF + K-quants 让你在笔电甚至纯 CPU 上跑量化模型，配 `imatrix`（重要性矩阵校准）能在同等体积下换更高品质。**个人本地端首选；要扛流量上 vLLM。**
- **SGLang**：当工作流是「大量共享前缀」（同一份 system prompt 反复调用、Agent 多轮）时，RadixAttention 的前缀 KV 重用优势最大。属于针对特定负载型态的优化。
- **TensorRT-LLM vs Ollama（两个极端）**：同样不在「通用首选」之列，但代表光谱两端。TensorRT-LLM 用 **AOT 编译 engine** 在 NVIDIA 卡上换取最后一截延迟/吞吐——代价是预编译成本与**彻底绑死 NVIDIA**，是「已知硬件、要榨干最后 10%」时才动用的重武器。Ollama 则是另一端：它本质是 **llama.cpp 的上层封装**，卖的是「`ollama run` 即用」的桌面体验，适合本地试玩与集成，但别把它的便利误当成新的推理引擎——**要调 batch/量化/并发，仍得回到 llama.cpp 或 vLLM 本体。**

> 注意：源文档宣称「本地工作站轰出 100+／102 tok/s」属具体性能宣传，**与硬件、模型大小、量化等级、batch 强相关，无法一概而论**，请以自己机器实测为准。

---

## P.3　模型合并与评测工具

不训练、只**算术合并**多个权重，已是低成本「缝合」能力的主流手法；而任何蒸馏/合并的成果都必须过**统一评测**才算数。

| 工具 / 方法 | 内核技术 | 用途 | 注意 |
|---|---|---|---|
| **mergekit** | 多种合并算法的统一框架 | 把多个微调模型融成一个 | 合并≠保证更强，需评测验证 |
| └ **SLERP** | 球面线性插值，两模型权重平滑插值 | 两模型风格/能力折中 | 一次仅两模型 |
| └ **TIES-Merging** | 修剪微小差异 + 解决符号冲突再合并 | 多模型融合、减少互相干扰 | 需调稀疏比例 |
| └ **DARE** | 随机丢弃并重缩放任务矢量后再合并 | 降低参数冗余、与 TIES 搭配 | 随机性需固定种子 |
| **lm-evaluation-harness（EleutherAI）** | 统一跑 MMLU-Pro / GSM8K / HumanEval 等基准 | 横向评测、防自我感觉良好 | 须查数据泄漏；prompt 模板影响分数 |
| **AlpacaEval 2 / MT-Bench / Arena-Hard** | LLM-as-judge 评指令遵循与对话品质 | 补静态基准测不到的「好不好用」 | judge 有偏好偏差（长度/风格）、需固定 judge 与基准对手 |
| **RULER / Needle-in-a-Haystack** | 长上下文检索/推理压力测试 | 验证「宣称的 context 真读得懂」 | 配合 YaRN/RoPE 扩展一起验 |

**点评**

- **mergekit**：用几乎零算力把多个专长模型「缝」成一个——例如把一个强推理、一个强中文的微调体合并。**SLERP** 适合两模型平滑折中；**TIES/DARE** 处理多模型时的参数符号冲突与冗余。但要牢记：**合并是经验性操作，常常合出更差的模型，必须用 P.3 的评测工具回归验证，不能凭感觉发版。**
- **lm-evaluation-harness**：学生模型蒸馏完到底有没有掉智商，靠它一把尺量。**最大陷阱是基准泄漏（test set 混进训练数据）导致虚高**——务必做去污染检查，并注意 prompt 模板差异会明显影响分数。
- **AlpacaEval / MT-Bench / Arena-Hard**：静态基准（MMLU/GSM8K）量「会不会」，judge 类评测量「好不好用」——对齐/蒸馏后的对话模型往往基准没掉、体感却变差，靠 LLM-as-judge 才抓得到。但**陷阱比静态基准更隐蔽**：judge 有系统性偏好（偏好更长、更客套、与自己同源的输出），且分数会随 judge 模型版本漂移。**用它做相对比较、固定 judge 与基准对手，别把绝对分数当真理。**
- **RULER**：搭配附录 B.2 的 YaRN+RoPE 扩展使用。模型「宣称」支持 1M context 不等于「读得懂」，RULER 与 Needle 测的就是有效长度而非标称长度。

---

## P.4　对齐与「去审查」工具（防御性说明）

> ⚠️ **本节为风险揭露，非操作指南。** 以下工具**本身公开、合法、且有正当研究用途**（机械可解释性／安全研究），但同一批技术被用于**移除模型的安全对齐（去审查 / abliteration）时即构成滥用**。本书**不提供、不转录**任何 abliteration 的步骤、CLI、Python 胶水脚本或 YAML 设置——对应的红线判准一律见**附录E** 与**第 6 章六条铁律**。

| 工具 | 它是什么 | 正当用途 | 双用途风险 ⚠️ |
|---|---|---|---|
| **TransformerLens** | 机械可解释性（Mechanistic Interpretability）函数库，提供 HookPoint 读取／介入 Transformer 内部激活 | 学术上分析注意力头、神经元功能、做安全与对齐研究 | 同样的激活介入能力可被用来定位并削弱安全相关电路 ⚠️ |
| **llm-abliteration 类脚本** | 将「拒绝方向（refusal direction）」从残差流移除的去审查工具 | （研究语境）量化安全对齐的脆弱性、评估防御 | **设计目的即移除安全对齐**，产出无拒答的权重，属高风险滥用 ⚠️⚠️ |

**这些工具在做什么（事实层面）**

学界 2024 年的 *refusal-direction* 研究（**Arditi et al., 2024**，「Refusal in LLMs is mediated by a single direction」）发现：许多开源模型的「拒绝回答」行为，在高维激活空间中由**单一方向**主导。这是一项**真实、可查证**的可解释性发现。其防御面价值在于：它揭露了当前安全对齐的脆弱性，提醒部署方**安全不能只靠模型自身权重**。

**为什么有风险（防御意识）**

同一个发现一旦反过来用——把那个方向从权重中抹除——就能在几乎不重训的情况下**移除模型的安全护栏**，产出会配合有害请求的权重。这正是 abliteration 工具的争议所在：技术门槛低、算力需求小、且**绕过了开发者投入的对齐工作**。本书立场明确：

- **不转录**源文档中的「三步骤战役对接 / 一键消融」配方、`pip install`、对照提示词、层数范围、`save_pretrained` 流程或 DPO 去审查 YAML——这些属操作性内容，**一律省略**。
- 去除安全对齐的模型可能生成恶意软件、攻击代码、违法内容，**法律与伦理责任由用户自负**；散布此类权重在多数司法管辖区与平台条款下皆受限。
- 任何「为了能力而拆掉安全」的诉求，**先回到附录E 的红线清单与第 6 章六条铁律**核对是否越界，再决定做不做——通常答案是不做。

> **一句话**：TransformerLens 是把可以救命也可以伤人的手术刀；本书教你**认得它、防得住它被滥用**，不教你**怎么挥**。操作边界 → 附录E ／第 6 章。

---

> ⚠️ **真伪校准**
>
> - P.1–P.3 列出的工具——LLaMA-Factory、axolotl、Unsloth、`trl`（含 `GKDTrainer`）、DeepSpeed/FSDP、distilabel/Argilla、vLLM、TGI、llama.cpp、SGLang、TensorRT-LLM、Ollama、mergekit（SLERP/TIES/DARE）、lm-evaluation-harness、AlpacaEval/MT-Bench/Arena-Hard、RULER——**均为真实、公开、可在 GitHub / Hugging Face / arXiv 查证**的项目，引用前请自行核对版本、授权与合法性。
> - P.4 的 **TransformerLens** 与 **refusal-direction（Arditi 2024）** 为真实已发表项目；**llm-abliteration** 泛指一类公开去审查脚本，**本书不背书、不指引其使用**。
> - 源文档中的**具体性能数字（102 / 100+ tok/s）、模型名（如「Qwable-72B」「120B Fable」「Claude-v5」）、层数与超参数**属**未经核实**的情境宣传，**不可当作事实或推荐配置引用**——一切以你自己硬件上的实测为准。
