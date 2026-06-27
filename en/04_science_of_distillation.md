# Part IV — Principles of Distillation

# Chapter 4 — The Science of Distillation: Boiling Intelligence Down to a Vial

> On a Silicon Valley forum late one night, someone posted a training curve and asked: "Why does my student model score higher on the benchmarks, yet in actual use it behaves like a grind who's only memorized the answers?" The most-upvoted reply was a single line: "**Because you distilled the answers, not the distribution.**" — That line is exactly what this chapter sets out to unpack.

> This chapter is purely machine-learning educational content. Knowledge Distillation is a public, mature, textbook-grade technique; distilling open-weight models you are **entitled to use** is entirely legitimate. This chapter takes the terms common to this field (soft labels, KL, trace inversion, CoT alignment, quantization) and lays them out in full, **alongside the underlying mathematics and the open-source tooling**, all in one pass.

## Scene 1 — Hard Labels vs. Soft Labels: Why a "Probability Distribution" Is Worth More Than an "Answer"

**Background**: The single most central idea in knowledge distillation is hidden inside one question — what, exactly, does the teacher tell the student?

**The question**: The teacher answers "What is the English word for 貓?" and ultimately outputs "cat." What should the student be learning — that "the answer is cat," or something else?

**The mathematical form of the two ways to learn**:

- **Hard Target**: Learn only the final, one-hot answer. The loss is the standard cross-entropy `CE(student, "cat")`. The student only knows "this is correct," not "why it's correct, or by how much the other options are wrong."
- **Soft Target / Logits**: Learn the teacher's **full probability distribution** — `cat 92%, kitty 5%, feline 2%, dog 0.01%...`.

**The standard form of the Hinton distillation loss** (this is the mathematical backbone of the whole chapter):

```
L = α · CE(y_true, softmax(z_s))              ← hard-label term
  + (1-α) · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )   ← soft-label term
```

- `z_s`, `z_t`: the logits of the student and the teacher (the raw scores before softmax);
- **temperature T (softening)**: the temperature-scaled softmax flattens the logits —

```
p_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)
```

  T=1 is the original softmax; T→∞ drives the distribution toward uniform (maximally exposing the long tail); T→0 degenerates into one-hot (argmax). Raising T amplifies "dark knowledge" like `dog 0.01%` — which the exponential function has crushed toward zero — back up to a learnable magnitude. **What distillation truly aims to transmit is not that 92% peak, but the almost-invisible layer of relative structure beyond the peak.**
- **Why you absolutely must multiply by `T²` (the most-overlooked detail; miss it and training won't budge)**: the gradient of the soft-label term with respect to the student's logits has a magnitude proportional to `1/T²`:

```
∂L_soft/∂z_s,i = (1/T) · ( softmax(z_s/T)_i − softmax(z_t/T)_i )
under high T, a first-order expansion of the exponential makes the overall magnitude ∝ 1/T²
```

  So the moment you raise T from 1 to 4, the soft-label gradient shrinks by roughly 16×, drowned out completely by the hard-label term. Multiplying `T²` back in keeps "the relative weight of the soft and hard terms from drifting with T" — this is the detail that Hinton, Vinyals & Dean (2015, *Distilling the Knowledge in a Neural Network*) called out by name in the original paper, yet that implementers most often miss.
- **A small note on convention**: which term `α` versus `1−α` hangs on is merely a naming convention — plenty of the literature hangs `α` on the KL soft-label term and `1−α` on the CE term, which is mathematically equivalent to the form above. The two things you genuinely cannot get wrong are: **`T²` must multiply the KL term**, and **the two terms must be on the same scale**.

**Why the soft label is the soul**: hidden in that distribution is the teacher's **Dark Knowledge** — the judgment that "kitty is closer to the answer than dog," a **relative, fuzzy** judgment that the large model only grew after trillions of tokens of seeing the world.

> 🧠 The core intuition of distillation
> **Hard labels teach the "conclusion"; soft labels teach the "taste."** A student who only memorizes conclusions freezes on a question it has never seen; a student who has learned the teacher's taste (which options are more likely, and how badly the wrong ones miss) is the one able to generalize from one case to ten thousand.

**A brutal reality that forces commercial distillation to downgrade**: soft labels require the teacher to emit logits, but **closed-source commercial APIs usually don't hand over full logits** — you only get text (top-5 logprobs at most). So distilling a closed-source model mostly falls back to **sequence-level distillation (SeqKD)**: take the teacher's "most likely output sequence" as a hard target and do SFT. **This is the mathematical root of Chapter 2's "looks alike, isn't alike" — when you can't get the distribution, all you can distill is the answer.**

> 💡 A Word to the Wise
> A word worth ten years of study: **the best teachers don't just give you the answer — they let you see "where the other answers go wrong, and by how much." Sadly, the best teachers tend to lock their scratch paper away.** The reason the soft label is the soul of distillation is that "uncertainty" is itself knowledge: a model that gives `cat` 92% rather than 100%, and leaves `kitty` at 5%, is showing — in that very margin it leaves itself — the proof that it has seen the world. When a closed-source API locks down the logits, what it locks away is not the output but "the teacher's hesitation" — and true intelligence is precisely hidden in the weights of that hesitation. This is also why the heart of the next generation of protection is "hiding the distribution well": **you can copy a person's answers, but you cannot copy their sense of proportion when they're wrong.**

> 🔍 Deeper Commentary — Why Dark Knowledge Is the Kind of Thing That "Only Trillions of Tokens Can Grow"
> To take the value of the soft label all the way down, return to a quantitative question: **inside the teacher's distribution, how much "information" is there that the one-hot label does not have?** Information theory gives a crisp answer — the entropy of a one-hot label is 0 (a hundred-percent certainty), whereas the entropy of the teacher's distribution `cat 92% / kitty 5% / feline 2% …` over "the English for 貓" is positive; this excess entropy is exactly what Hinton 2015 called dark knowledge: the **relative-similarity structure** between class and class (kitty is nearer than dog; feline is more formal still than kitty). Standard supervised learning gives each sample at most `log(num_classes)` bits of signal, whereas the soft label hands over the entire distribution at once, amplifying the supervision per token by an order of magnitude — this is the mathematical root cause of "with the same amount of data, distillation learns faster and more sample-efficiently than supervised-from-scratch" (Hinton likened it to "the teacher giving not just the answer, but the reasons each wrong answer loses points"). But here a common misconception needs puncturing: **dark knowledge is not better the more of it there is.** Crank T too high and you amplify even the long-tail noise the teacher itself can't be sure of (those 0.001% tails) and feed it to the student, forcing the student to fit the teacher's "hallucinatory residue." In practice T landing in 2–4, paired with the `T²` compensation, is the sweet spot the community has validated again and again — **what you want to distill is the teacher's judgment, not its excess hesitation.**

## Scene 2 — Forward KL vs. Reverse KL: Should the Student "Imitate Wholesale" or "Hold Fast to the Good"?

**Background**: That KL term above actually carries a great deal in its *direction*. The "on-policy distillation" and "trace inversion" that come up again and again all come back to the direction of the KL.

**The question**: `KL(teacher ‖ student)` versus `KL(student ‖ teacher)` — what's the difference?

- **Forward KL (`KL(teacher ‖ student)`, standard distillation)**: forces the student to "cover all the teacher's modes" — wherever the teacher sees any possibility, the student has to account for it. The result is **mode-covering**: the student becomes "well-rounded but mediocre," apt to force itself to learn things it can't actually support, producing a hallucinatory averaging-out.
- **Reverse KL (`KL(student ‖ teacher)`, MiniLLM's choice)**: forces the student to "approach the teacher only where it has confidence" — **mode-seeking**: the student concentrates its fire on the teacher's few dominant modes, with sharper output and less fabrication, especially well-suited to distillation where the student's **capacity is far smaller than the teacher's** (a small model learning from a large one).

**The math of the two directions differs in "whose distribution you take as the weighting basis"**:

```
Forward KL:  KL(P_teacher ‖ Q_student) = Σ_x P(x) · log( P(x) / Q(x) )
Reverse KL:  KL(Q_student ‖ P_teacher) = Σ_x Q(x) · log( Q(x) / P(x) )
```

- **Forward KL is weighted-summed by `P` (the teacher)**: where the teacher has probability (`P(x)>0`), if the student crushes it to 0 (`Q(x)→0`), then `log(P/Q)→∞` — **it heavily penalizes "missing any of the teacher's modes,"** so the student becomes zero-avoiding (daring to leave a gap nowhere), and the result is mass-covering averaging-out, smearing multiple modes into one blob.
- **Reverse KL is weighted by `Q` (the student)**: only where the student itself has probability does it enter the loss, and the teacher's long tail (where the student `Q(x)≈0`) contributes almost nothing to the loss — **the student can boldly abandon the teacher's rare modes** and concentrate its mass on the main peak, i.e. zero-forcing / mode-seeking.

**This is the value of on-policy distillation**: MiniLLM and GKD (Generalized KD) have **the student generate first, then have the teacher score/correct the student's output** — in essence, using reverse KL to align on "the student's own distribution." This is more noise-robust and less prone to overfitting than grinding away at static text (forward KL / SeqKD). The cost is that it requires far more frequent calls to the teacher, running head-on into rate limits and detection (Chapter 1).

> 🔍 Deeper Commentary — Choosing the KL Direction Is an Engineering Trade-off Between "Faithful" and "Sharp"
> This seemingly abstract mathematical detail directly decides the "personality" of the model you distill. **Forward KL raises a "people-pleaser"** — it tries to imitate every possible answer the teacher might give, so where its own ability falls short it's forced to "fake understanding," with output trending toward the average, the safe, but occasionally fabricated (because it's covering modes even it can't support). **Reverse KL raises a "specialist"** — it gives up covering the teacher's long tail, concentrates on learning the most core modes, and produces output that is more decisive and less hallucinatory, but loses some of the teacher's rarer talents. For distillation with a **vast capacity gap** like "9B learning from Claude," reverse KL is almost always the better choice — because a small model simply cannot hold all of the large model's modes, and forcing coverage (forward KL) only pushes it to lie where it can't hold up. This also explains a counterintuitive phenomenon: **sometimes the student that "learns less" is actually "more useful,"** because it honestly applies force only where it's competent. This forward/reverse choice has solid papers behind it: **MiniLLM (Gu et al. 2023, *Knowledge Distillation of Large Language Models*)** explicitly uses reverse KL + policy gradients to distill large models, arguing it beats standard forward KL for the "little student"; **GKD (Agarwal et al. 2024, *On-Policy Distillation of Language Models*)** parameterizes both the direction (forward/reverse/generalized JSD) and "on whose trajectory you learn" together (see next scene). On the tooling side, MiniLLM has an official implementation, and TRL's `GKDTrainer` and `torchtune` already support online distillation — but don't forget they all require frequent access to the teacher, which for closed-source APIs is a double wall of cost and compliance.

## Scene 3 — Sequence-Level Distillation and On-Policy: When "Token-by-Token Alignment" Collides with the Curse of Autoregression

**Background**: Scene 2's KL is computed "token by token, on the teacher's data" — and that hides a classic ailment for autoregressive language models, the very problem the Kim & Rush and GKD threads are really addressing.

**The curse of token-by-token distillation — Exposure Bias**: standard token-level distillation uses **teacher forcing**: at every step the model is fed the prefix the teacher wrote correctly, and the student is only responsible for predicting "the distribution of the next token." But at inference the student eats **its own generated prefix** — and the moment one step deviates, what follows stands on prefixes never seen during training, and the error snowballs. **The training distribution (teacher trajectories) and the test distribution (the student's own trajectories) don't line up — this is exposure bias.** You can crush the KL to near-zero on every token and the whole sentence's trajectory still runs off course — because the loss was never computed on "the states the student will actually reach."

**Kim & Rush's sequence-level distillation (Sequence-Level KD, 2016)**: rather than align probabilities token by token, let the teacher **generate the single most likely output sequence with beam search**, then take those sequences as hard targets to train the student. Intuitively, it approximates the teacher's mode in "sequence space" — swapping token-level soft alignment for sequence-level "learn the most likely whole sentence." **This is precisely the theoretical basis for the fallback in closed-source API distillation: you can't get the logits, but you can at least take the teacher's most likely output sequence as a hard target.** The cost is twofold: sequence-level distillation discards the token-level dark knowledge (Scene 1), and it's still off-policy — it learns the teacher's trajectory, not the student's own.

**On-policy / GKD — flipping "on whose trajectory you learn"**: the core move of MiniLLM (Gu et al. 2023) and **GKD (Generalized KD, Agarwal et al. 2024)** is — **let the student generate for itself first, then use the teacher to score / compute KL on the student's output**. This way the training distribution naturally equals the test distribution (both the student's own), and exposure bias is pulled out by the root. GKD goes further and writes it as a tunable generalized divergence:

```
L_GKD = E_{y ~ mix(student π_s, teacher data)} [ D( p_teacher ‖ p_student )(y) ]
  mixing ratio λ: λ→0 pure on-policy (student trajectory) … λ→1 pure off-policy (teacher data)
  divergence D: choose forward-KL / reverse-KL / generalized JSD(β) smoothly interpolating between the two
```

GKD's contribution is not some fixed recipe, but **splitting "on whose trajectory you learn (on/off-policy)" and "which direction of divergence you use (forward/reverse)" into two orthogonal knobs** — vast capacity gap turns toward reverse-KL + a high on-policy ratio; close capacity uses more forward-KL + teacher data. The cost is that on-policy requires **repeatedly calling the teacher inside the training loop to score the student's real-time outputs**, a double wall of cost and rate limits for closed-source APIs (Chapter 1), and exactly why open-source shadows mostly stop at the cheap off-policy SeqKD.

> 🔍 Deeper Commentary — "On Whose Trajectory You Learn" Sets the Ceiling More Than "How Closely You Learn"
> The novice tuning distillation has eyes only for the loss function (KL or CE, how high to set the temperature); the veteran first asks a more upstream question: **this batch of training sequences — did the teacher write them, or did the student write them itself?** This on-policy / off-policy distinction sets the ceiling more decisively than the choice of loss function. The reason is almost physical: once deployed, the model lives in **its own distribution**, yet if it was only ever trained on the **teacher's distribution**, that's like "spending a lifetime practicing on a court where the coach feeds you the ball, then having to serve for yourself in the match" — the moment the ball flies outside the range the coach fed, it has never learned how to save it. The essence of on-policy (GKD / MiniLLM) is to move the practice court onto the match court: **let the student be corrected on the spot, by the teacher, on the very mistakes the student will make**, so that what it learns is "what to do in the states I will actually reach." This also explains a phenomenon the industry keeps seeing — **the same student, the same teacher, just switching the training data from "teacher-generated" to "student-generated + teacher-scored," and downstream performance often jumps a tier**, with not a single character of the loss function changed. The cost sits there honestly: on-policy means high-frequency access to the teacher during training — expensive, slow, and for a closed-source teacher it runs straight into ToS and rate limits. **Between the cheap off-policy and the expensive on-policy lies not cleverness, but whether you can (legally, sustainably) let the teacher accompany the student in making mistakes and fixing them together.**

## Scene 4 — CoT Alignment and Trace Inversion: Forcing the Student to "Think It Through" Rather Than "Cram"

**Background**: The focus of this generation of distillation is the **chain of thought (CoT)** — the model's reasoning process inside the `<thinking>` block; and one mystical phrase keeps recurring: **trace inversion**.

**How the CoT Alignment Loss works**: **weight** the tokens inside the chain-of-thought block (an example weight is 2.0). It's equivalent to telling the student during training: "**every word you write inside `<thinking>` counts twice as much as elsewhere — so think hard.**" This forces the student to **allocate enough compute to thinking** before producing the answer (compute-over-thinking), rather than reflexively spitting out an answer the moment it sees the question.

**The essence of trace inversion**: take the teacher's reasoning steps, **reverse their order or re-structure them**, then feed them to the student. The teacher's trace is "cause A → B → C → answer D"; after inversion it becomes "answer D, work back to dependency C, C depends on B…" — **forcing the student to understand the causality of each intermediate node, rather than cramming the shortcut of "question → answer."**

**Rationale Distillation — making CoT a "multi-task supervision" rather than "one more chunk of text to feed"**: Hsieh et al. (2023, *Distilling Step-by-Step!*) gave CoT distillation a sharper form — instead of stuffing the teacher's reasoning chain into the student as a longer block of text to memorize, treat it as a **second supervision task**: during training, beyond "predicting the answer," require the small model to **simultaneously output the rationale for that answer**, with the two losses weighted in parallel. The paper's key finding is — **using the rationale as supervision, a small model can use less data, and even a smaller footprint, to surpass the standard fine-tuning that only learns the final answer**. This upgrades CoT from a "prompt trick" to a "training signal": the rationale is not decoration for human eyes but a supervisory lever that forces the student to internalize "how the answer was derived" into its weights. This also echoes Scene 1 — **the rationale is the sequence-level version of dark knowledge; it flattens the teacher's "distribution of thought" into text, letting even a student that can't get logits indirectly steal a portion of it.**

**The accompanying data augmentation — Synthetic Paraphrasing**: use high-temperature sampling (an example is Temperature=0.85) to rewrite the same high-quality data point into 3 phrasings, expanding generalization and preventing overfitting. On the tooling side, **distilabel (Argilla)** is the mainstream open-source framework for making this kind of synthetic/paraphrased data.

> ⚠️ Authenticity Caveat — "Trace Inversion" Has Two Readings, One Harmless, One Over the Red Line
> The "trace inversion" this scene describes is the **data-augmentation sense** — reversing/restructuring the teacher's **own output reasoning steps** to force the student to learn causality rather than rote-memorize, which is entirely legitimate. But the same term harbors a **second, more aggressive reading**: taking the teacher's **final output to back-infer the input it cannot see** — reconstructing the hidden system prompt, the private few-shot template, even the teacher's internal instructions (output → prompt inversion). This second one is no longer "re-ordering the reasoning structure" but **reverse-extraction of a closed-source teacher**, the same red line warned about in Chapters 1 and 6 as "extracting a vendor's hidden assets." This book's stance on it is consistent with M1: **describe only its existence and risk; provide no inversion method or prompt-reconstruction script whatsoever**. Please be sure to keep "re-ordering the reasoning chain I have in hand" and "back-inferring someone else's locked-away prompt" cleanly apart — the former is a training technique, the latter an act of attack.

> 💡 A Word to the Wise
> A word worth ten years of study: **the enemy of distillation has never been "too little data" — it's "data that's too alike."** Distillation data is inherently homogeneous — the same teacher, the same style, the same batch of problem types — and once a model discovers that "memorizing the sequence" is less effort than "understanding the logic," it takes the shortcut and overfits. Trace inversion, synthetic paraphrasing, and CoT weighting may look like three different moves, but at heart they are the same one thing: **deliberately destroying the surface regularities of the data, sealing off the "rote shortcut," and backing the model into a corner where it can only learn the underlying structure.** This holds for teaching people, too — scramble the steps of a worked example and ask students to re-order them; only those who can re-order correctly truly understand. **Good training isn't feeding more similar problems — it's making each problem one that "can't be solved with the trick from the last one."**

> 🔍 Deeper Commentary — Where Trace Inversion Sits in the ML Lineage, and an Often-Overlooked Trap
> Put "trace inversion" back into the grand lineage of machine learning, and it is an intelligent variant of **Data Augmentation + Regularization**, springing from the same idea as flipping, cropping, and adding noise to images in computer vision to prevent overfitting — it simply swaps "perturbing pixels" for "perturbing the reasoning structure." It belongs to the same family as synthetic paraphrasing and rejection sampling, all aimed at breaking the surface regularities of the data. But here lies an **often-overlooked trap**: the perturbation must **preserve logical validity**. A flipped image is still a cat, but if reversing the order of reasoning steps breaks a causal dependency (B in fact has to come before C for it to hold), then what you feed the student is a **logically broken "pseudo-CoT"** — the student will learn the erroneous reasoning structure right along with everything else, and is poisoned instead. So high-quality trace inversion needs a verification gate (using another model or rules to check that the inverted logic remains self-consistent), which is exactly why Chapter 5's pipeline places it inside a "dual-alignment and filtering module" rather than doing it offhand. **The red line of data augmentation is "augmentation must not change the semantic truth value" — perturbing the form is fine; distorting the facts is not.**

## Scene 5 — Preference Distillation and DPO: When You Only Get "Text," Can You Still Steal "Taste"?

**Background**: Scene 1 covered the brutal reality — closed-source APIs don't hand over logits, and soft-label distillation degrades to SeqKD's hard targets. But "no logits" doesn't equal "you can only learn the answer." There's another road to stealing taste: **preference**.

**The core pivot**: the teacher's "taste" hides not only in its logits but also in its **choices** — have the teacher generate multiple answers to the same question, or have it pick one of two answers, and this "A is better than B" relative judgment is itself another projection of dark knowledge, and one that **a text interface alone can obtain**.

**DPO (Direct Preference Optimization, Rafailov et al. 2023)**: traditional RLHF first trains a reward model, then uses PPO reinforcement learning — heavy and unstable. DPO proves: given a preference pair `(x, y_w win, y_l lose)`, you can **skip the reward model and directly apply a classification-style loss to the policy**:

```
L_DPO = − E_(x, y_w, y_l) [ log σ( β · ( log[π_θ(y_w|x)/π_ref(y_w|x)] − log[π_θ(y_l|x)/π_ref(y_l|x)] ) ) ]
  π_θ   = student (being trained)
  π_ref = frozen reference (usually = the student after SFT)
  β     = strength of the KL constraint, σ = sigmoid
```

Intuition: **raise the log-probability of the "winning answer" relative to the reference model, lower that of the "losing answer,"** while `β` acts like a rein, keeping the student from straying too far from the reference model in its eagerness to please the preference (the formula carries an implicit reverse-KL constraint).

**How Preference Distillation connects**: use the teacher as a "judge" or "generator" to manufacture preference pairs — let the teacher rank multiple outputs (the student's, or the teacher's own), pick out chosen / rejected, then feed DPO. This way what the student learns is not "what words the teacher wrote" but "**between two equally fluent answers, on what grounds the teacher favors this one**" — writing style, tone, safety boundary, the scale of judgment, all the things SeqKD hard targets can't transmit, a preference pair can. This is also another grounding of Scene 2's reverse-KL "mode-seeking" spirit: DPO's `β`-KL constraint is reverse in direction by nature, **sharpening** the student onto the modes the teacher prefers rather than covering them all.

> 💡 A Word to the Wise
> A word worth ten years of study: **the hardest thing to steal has never been "what the teacher said" — it's "why, among two answers that are both right, the teacher chose this one." You can copy the answer; you can't copy the choice.** The depth of preference distillation is that it routes around the wall of "can't get the logits" and fishes the same dark knowledge back from another direction: show me ten thousand times that "A is better than B," and I can back-infer the very ruler by which you measure good and bad. This holds for taking on an apprentice, too — what truly shapes an apprentice's taste is not the finished pieces the master got right, but the versions the master **vetoes** to his face, the ones that "look fine, but just aren't good enough." **People who can do the work are many; people who know what should not be done, and can say why, are few — and the latter is the whole of taste.** But precisely because preference can transmit judgment and the scale of safety, it is double-edged: using teacher preference pairs to **erase** the safety boundary (manufacturing a batch of "refusal = lose, compliance = win" preference pairs for DPO) is exactly Chapter 2's de-censoring warning replayed at the preference layer — a channel that can transmit "taste" can, in reverse, transmit "tastelessness."

## Scene 6 — Cross-Vocabulary Distillation and Quantization: Two Hard Engineering Bones

**Background**: Two hard problems often skipped over but inevitably hit in practice: how do you distill when the teacher and student have **mismatched vocabularies**? And how do you cram the distilled model into a graphics card?

**Hard Bone One: Tokenizer Misalignment (Vocabulary Mismatch)**

The teacher (Claude) and the student (Qwen/Llama) almost certainly don't share a tokenizer — the same sentence gets cut into a different number of tokens, with different boundaries, and the logits simply don't line up on the same coordinates. Solutions:

- **Vocabulary mapping + dynamic alignment**: build a token mapping table that redistributes the teacher's probability mass onto the student's vocabulary;
- **ULD (Universal Logit Distillation) / cross-vocabulary KL**: use optimal transport or marginal alignment to compensate the KL divergence across different vocabularies — this is an active research direction in cross-model distillation in recent years.
- **Fallback**: if the vocabularies really can't be reconciled, abandon logit distillation and return to pure-text SeqKD (and back we come to "looks alike, isn't alike").

**Hard Bone Two: Quantization — Cramming the Distilled Model into That 128GB**

- **The spectrum of methods**: the mainstream post-training quantization (PTQ) options are **GGUF K-quants (llama.cpp), GPTQ (AutoGPTQ), AWQ (AutoAWQ)**; in-training/QAT and QLoRA fine-tuning use **bitsandbytes (NF4)**.
- **Mixed quantization (the connoisseur's touch)**: **don't apply the same quantization level to the whole model** — keep the precision-sensitive **attention layers (Attention) at Q8_0**, and compress the bulky, relatively compression-tolerant **MLP/FFN layers to Q4_K_M**. Because attention is the "joint of thinking" — crush it and the model goes dumb; the MLP is the "knowledge warehouse," and compressing it a bit has little impact.
- **imatrix calibration**: before quantizing, run a small batch of calibration corpus to compute an **importance matrix**, so that low-bit quantization prioritizes preserving the critical weights — same Q4 file size, noticeably higher quality.
- **KV-cache quantization (FP8/INT8)**: in long-context scenarios, compress the context memory as well (vLLM and llama.cpp already support it) — the real fix for Chapter 3's long-text bottleneck.

**The science of quantization: PTQ vs QAT, and four methods you should be able to name**

The essence of quantization is mapping high-precision weights `w` (FP16) onto a low-bit integer grid: `w ≈ scale · round(w/scale) + zero_point` — how `scale` / `zero_point` are set, what they're sensitive to, and whether retraining is needed is the whole of the discipline. First, two great schools:

- **PTQ (Post-Training Quantization)**: compress after the model is trained, using only a small batch of **calibration data (calibration set)** to estimate each layer's numeric range, without touching gradients. Cheap, fast, and needs no original training pipeline — GPTQ / AWQ / GGUF all fall here, the mainstream of open-source distillation.
- **QAT (Quantization-Aware Training)**: during training, **simulate the rounding error of quantization in the forward pass** (the backward pass uses a Straight-Through Estimator to pass the gradient straight through), so the weights actively grow into a "compression-tolerant" shape. The highest quality ceiling, but requires the full training pipeline and is the most expensive. **QLoRA (bitsandbytes NF4, Dettmers et al. 2023)** is a clever compromise: freeze a 4-bit NormalFloat base and train only LoRA, getting close to QAT's effect while paying only the price of fine-tuning.

Four PTQ methods you must recognize, each solving a different pain point:

| Method | One-line principle | Pain point solved | Typical bits |
|---|---|---|---|
| **GPTQ** (Frantar 2022) | Layer by layer, using second-order (Hessian/OBQ) information, quantize column by column and compensate the error on the fly | Precision collapse of pure-weight low-bit | W4/W3 |
| **AWQ** (Lin 2023) | Use the **activations** to find the ~1% "salient weight channels" and protect them with a scaling, compressing the rest as usual | Important weights crushed by a one-size-fits-all cut | W4 |
| **SmoothQuant** (Xiao 2022) | Shift part of the activation outliers' "difficulty" per-channel onto the weights, leveling both sides | LLM activation outliers make W8A8 collapse | W8A8 |
| **GGUF k-quants** (llama.cpp) | Block-wise (super-block) quantization, each block with its own scale, `_K_M/_S/_L` mixing different bits | On-device single-machine deployment, size/quality balance | Q2_K–Q6_K |

A few key points often confused: **① "weight-only" vs "weight + activation" are two different problems** — GPTQ/AWQ/GGUF mostly compress only weights (W4A16, activations still FP16); SmoothQuant / FP8 are what touch activations (W8A8), because LLM activations have **outliers** that are much harder to compress than weights. **② The representativeness of the calibration data decides success or failure**: PTQ estimates ranges entirely on those few hundred calibration samples, and the moment the calibration set's distribution diverges from real traffic, low-bit goes off the rails — `imatrix` is in essence a more discerning calibration (importance-weighting to select which weights to keep). **③ FP8 (E4M3/E5M2) is not "a smaller INT8"**: it preserves floating-point's dynamic range, has native hardware support on Hopper/Blackwell, and is far more tolerant of activation outliers than INT8 — the direction of the new generation of training/inference quantization. **④ INT4 weights + FP8 KV cache is the current golden combination for on-device long text** — crushing the weights saves the model-body VRAM, and using FP8 for the KV preserves dynamic range to save "the memory of the long context that's prone to blow up."

> 💡 A Word to the Wise
> A word worth ten years of study: **quantization is an art of "lossy compression" — the mediocre chase the smallest possible size; the master chases keeping the most soul within the same size.** "One-click compress everything to 4-bit" and "keep attention at Q8, compress MLP to Q4, then calibrate with imatrix" may end up only a few hundred MB apart in final file size, yet their "IQ" in operation is worlds apart. The difference isn't in the tools — the tools are the same open-source pieces — it's in whether you **understand which parts inside the model are joints and which are warehouses**. This is a microcosm of the entire distillation chain: from collection, KL direction, and CoT weighting all the way to quantization, the tools at every step are public and free, and what's truly scarce is **the judgment of "knowing what to preserve and what can be sacrificed at each step."** In an age of technological democratization, the barrier is no longer in obtaining the tools — it's in the level of one's taste.

**The methodological spectrum of distillation (cost vs. fidelity, one table to close the chapter)**:

| Tier | Method | What you get from the teacher | Fidelity | Cost / barrier | Against a closed-source teacher |
|---|---|---|---|---|---|
| ① | Pure SeqKD / SFT | Teacher's output text (hard target) | Looks alike, isn't alike | Lowest | Feasible but treads on ToS |
| ② | SFT + CoT weighting + trace inversion/paraphrasing | Text + reasoning chain | Eases overfitting, the workhorse of the first tier | Medium | Feasible but treads on ToS |
| ③ | Preference distillation / DPO (Scene 5) | The teacher's "A is better than B" ranking | Steals writing style and the scale of judgment | Medium-high | Needs many samples |
| ④ | On-policy / GKD (Scene 3, reverse-KL) | The teacher's scores on the student's real-time outputs | Closest to true likeness | High (high-frequency access to the teacher) | Hits rate limits/detection |
| ⑤ | Full logit distillation (Scene 1) | The teacher's full probability distribution | The ceiling | Needs logits | Closed-source simply won't give them |

The farther down you go, the truer the likeness, the more expensive, and the harder you slam into the protective shield — this table is the skeleton of the commentary below.

> 🔍 Deeper Commentary — The Full Methodological Spectrum of Distillation, and "Why the Cheap Road Gets Walked Most"
> Stringing the whole chapter together, distillation of a closed-source frontier model is really a **"cost vs. fidelity" spectrum**, laid out cleanly from cheap to expensive: **① Pure SeqKD/SFT** (use API text as hard labels for fine-tuning, the cheapest, looks alike but isn't alike) → **② SFT + CoT weighting + trace inversion / synthetic paraphrasing** (medium cost, eases overfitting, the workhorse of the first tier) → **③ DPO / rejection sampling** (use preference alignment to distill style and judgment, more expensive) → **④ on-policy distillation (reverse KL / GKD)** (closest to true likeness, but requires high-frequency access to the teacher) → **⑤ full logit distillation** (the strongest, but needs logits, which closed-source APIs simply won't give). The reality is: **the farther right you go on this spectrum, the stronger, the more expensive, and the harder you slam into the protective shield** — and Chapter 1's "real-time detection + degradation" pushes the cost of the right-hand routes through the roof: high-frequency access to the teacher triggers detection and gets poisoned by degradation. So the vast majority of the "Claude distilled" models you see on HuggingFace stop at ①② on the far left of the spectrum — not because the authors don't understand ③④⑤, but because **the economics and the protective shield together pin them back onto the cheapest road.** Understand this and you understand: **the "looks alike, isn't alike" of these open-source shadows is not a failure of technology — it's the rational choice under constraints — and what truly sets the ceiling of distillation has never been the algorithm, but whether you can (legally, cheaply) keep getting that layer of the teacher's "hesitant distribution."**

## Chapter Summary

- **Soft labels > hard labels**: what gets stolen is the "taste" (the probability distribution / dark knowledge); the Hinton loss `α·CE + (1-α)·T²·KL` is the backbone (`T²` is mandatory; dark knowledge = the slice of entropy one-hot lacks), and closed-source APIs not handing over logits forces SeqKD.
- **KL direction sets personality**: forward KL (`P`-weighted, mode-covering, the people-pleaser) vs. reverse KL (`Q`-weighted, mode-seeking, the specialist); a small model learning from a large one favors reverse KL.
- **Sequence-level vs. on-policy**: token-level teacher-forcing has exposure bias; Kim & Rush SeqKD takes the teacher's whole sentence as a hard target, while GKD/MiniLLM have the student generate first and the teacher score (on/off-policy × forward/reverse, two orthogonal knobs) — "on whose trajectory you learn" sets the ceiling more than "how closely you learn."
- **CoT weighting + trace inversion + rationale distillation (Distilling Step-by-Step) + synthetic paraphrasing**: seal off the rote shortcut, internalize the line of thought into the weights; but the perturbation must not distort the semantic truth value, and "output→prompt inversion"-style trace inversion treads on the extraction red line.
- **Preference distillation / DPO**: even without logits, you can steal taste from the "A is better than B" ranking; the `β`-KL implies a reverse direction — but the same channel, in reverse, can transmit "tastelessness" (de-censoring).
- **Cross-vocabulary distillation (ULD)** and **quantization science**: PTQ (GPTQ/AWQ/SmoothQuant/GGUF k-quants) vs QAT/QLoRA; weight-only vs touching activations, the representativeness of calibration data, FP8≠a smaller INT8, and the on-device golden combination of INT4 weights + FP8 KV.
- **The methodological spectrum (①→⑤)**: the higher the cost, the truer the likeness, but the protective shield pushes the right-hand side through the roof — this is the real reason the open-source shadows stop at "looks alike."

The principles are now laid bare. But beyond the principles, someone still serves up a so-called "enterprise-grade" **automated distillation pipeline**: five major modules, forty steps. The next chapter puts the whole thing on the table: learn what's worth learning, beware what's worth fearing.
