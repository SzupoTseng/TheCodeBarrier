## Preface — The Summer Fable 5 Disappeared

June 2026. There was a restlessness in the Silicon Valley air that no one had felt in a while.

That month, Anthropic dropped **Claude Fable 5** — which reportedly swept nearly every AI benchmark. The launch event hadn't even wound down before, from Mountain View down to the garages and apartments of the South Bay, countless terminals had already pasted in the API key. On the desk, the freshly delivered **NVIDIA DGX Spark** hummed its fans low, its 128GB of unified memory stretching out like an open frontier waiting to be settled.

And then things got strange.

The community started passing it around: Fable 5 had been fitted with a **shield no one had seen before** — the moment it "sensed" you were trying to box it in, trying to extract its capabilities, it would **quietly reroute your conversation down to an older model** to handle. You thought you were talking to a frontier model; in truth, the other side had already swapped out the player. Then came word stranger still: because of a single U.S. government **export-control directive**, Fable 5 was **pulled worldwide just a few days after its public release**. The community caught only the narrowest of windows — some say that in those few days, all of LocalLLaMA only managed to rescue **4,659 clean agent execution traces**.

The night the window slammed shut, Reddit's r/LocalLLaMA, r/opencodeCLI, and the HuggingFace threads went off. Overnight, a swarm of open-source models surfaced wearing the "Claude distilled" badge: someone released **Qwable-v1**, someone else shouted out **Qwythos-9B — runs in 4GB of VRAM, 1.04M context, uncensored**. If you were staring at that same flooded screen, there was probably just one thought in your head:

> **"Are any of these real, or is this just another Silicon Valley mass hallucination?"**

This book is the record of that investigation. It began with a concrete question — "**Which open-source models are currently distilling Anthropic's Fable and Mythos models?**" — and widened from there: on to "**How do I distill one myself?**," and further still to "Once you've distilled one, how do you use it, and how do you sustain it?"

---

### Three Things, Stewed in the Same Pot

You'll quickly find three kinds of ingredients floating in this broth, and **fishing them out and telling them apart is the first lesson of this investigation**:

1. **Facts that hold up**: the mathematics of knowledge distillation, GGUF quantization, the DGX Spark's memory architecture, the fact that Fable 5 truly existed.
2. **Legends you can't get to the bottom of**: "the Mythos model," "Qwythos-9B," "Empero AI," "pulled within 4 days by export controls," "rerouted to Opus 4.8" — **these could be true, could be tall tales from the rumor mill, or could be something the search engine invented on its own**. This book stamps every one with a "⚠️ Authenticity Caveat" and vouches for none of them.
3. **Techniques that cross the line**: using proxy IPs and rotating multiple keys to **slip past a vendor's detection and rate limits**, using layer-upon-layer wrapped prompts to **fool that shield**. On this part, this book **only explains why it's dangerous and why it's hard — it does not teach you how to do it**. The reason is spelled out in the final chapter.

### How This Investigation Unfolds

| Part | Where it goes | Corresponding source material |
|----|------------|------------|
| Part I　The Anti-Distillation Defenses | What that shield actually is (Ch. 1) | distillation detection, rerouting, export controls |
| Part II　An Inventory of Open-Source Distillations | The list of those "shadows of Claude" (Ch. 2) | all the models + datasets + source links |
| Part III　The Dragon-Taming Grounds | Which one to run on the DGX Spark (Ch. 3) | hardware analysis + comparison tables + dual-model workflow |
| Part IV　The Principles of Distillation | The science of distillation (Ch. 4) | soft/hard labels, KL, trace inversion, quantization |
| Part V　The Pipeline | An enterprise-grade automated distillation line (Ch. 5) | the five modules + the full 40 steps + architecture diagram + prompt plan |
| Part VI　Reality Calibration | How red that red line really is (Ch. 6) | ToS, law, the economics, the six iron rules |
| Part VII　Agentic Operation | The distilled model starts to act — the reason–action loop (Ch. 7) | the two-tier closed loop, ViT/NaViT perception, coordinate grounding, ReAct, injection defenses |
| Part VIII　The Hard Problems & Human-Like Control | Twenty hard problems + turning "stop-and-go" into muscle memory (Ch. 8) | the five classes of hard problems, dual-loop decoupling, Delta-Frame, ballistic mouse motion, SIGSTOP interruption |
| Part IX　AI-Native | Redesigning the computer for AI: from pixel clicks to a semantic state bus (Ch. 9) | the paradigm clash, 20 frontier directions, AI-native OS, evolution roadmap |
| Part X　Caching | The physics of caching: the bottleneck isn't compute, it's memory (Ch. 10) | KV/Prompt Cache, memory-bound access, VRAM formulas, PagedAttention, RadixTree, FlashDecoding |
| Part XI　The Armory | Hyperscale caching and the frontier armory (Ch. 11) | three-tier heterogeneous storage, MLA, Cache-Aware Routing, PD disaggregation, 20 major directions |
| Appendices | A. Model/dataset/source quick reference · B. Terms & tools quick reference · C. Controversies & cognition Q&A · D. Benchmark question-bank design · E. Decensoring & safety assessment · F. Agentic-operation tools & papers quick reference · G. Caching techniques quick reference |

### Three Movements: Build It, Use It, Sustain It

When this investigation began, it asked only a narrow question — "How do I distill my own shadow of Claude?" But the moment you actually roll up your sleeves, the question grows on its own. It forces you to ask "and then what?" three times over, and each "and then what?" pushes you into a deeper engineering world. And so, almost without anyone noticing, this book grew into three movements:

- **First Movement — Build It (Ch. 1–6)**: what that anti-distillation shield is, which shadows the community managed to rescue, what machine to run them on, how the science of distillation works, what an enterprise-grade pipeline looks like, and how red that legal and ethical red line really is. **The goal: to lawfully build, on your own desk, an intelligence that belongs to you alone.**
- **Second Movement — Use It (Ch. 7–9)**: a model that can only answer questions is still just a brain lying on a hard drive. The greatest value of that small, cheap, local model you distilled isn't "answering one more question" — it's serving as a pair of **cheap, high-frequency, close-at-hand, private eyes and hands**, to perceive and to operate a computer. So we take up the reason–action loop, the twenty hard problems, human-like control, and finally we question the entire paradigm this craft stands on. **The goal: to make the intelligence you built truly move.**
- **Third Movement — Sustain It (Ch. 10–11)**: when you want to scale those "eyes and hands" from a demo to serving thousands of people a second, the bill blows up in the place you'd least expect — not compute, but memory. The KV Cache will eat through all 128GB of your DGX Spark some morning, while the model weights themselves are only 16GB. So we take up the physics of caching, the hyperscale KV wars, the twenty-strong armory. **The goal: to make the intelligence you built and set in motion something you can afford to keep, and run for the long haul.**

Build it, use it, sustain it — these three verbs are the spine that runs through all eleven chapters. You'll find they interlock: the "cheap eyes and hands" of the Second Movement are precisely the product of the distillation and quantization of the First; the "memory wars" of the Third turn back around and decide whether that machine from the First can hold up under the ten-thousand-fold concurrency of the Second. The three movements aren't three books nailed together — they're a single engineering ideal, interrogated three times.

At the end of each chapter sits a single line of "💡 A Word to the Wise" as that night's notes — striving for "a word worth ten years of study"; followed by a passage of "🔍 Deeper Commentary," digging into the engineering reality behind the case and that unseen road.

> ⚠️ Educational & Research Notice
> This book is for educational and research purposes, an organization and annotation of relevant public material and the current state of the art. Its descriptions of unauthorized capability extraction, circumventing access controls, or violating terms of service **serve only as risk disclosure** and do not constitute operational advice. Distilling an open-weight model you **have the right to use** is legitimate; performing large-scale unauthorized distillation against a closed-source commercial API may violate ToS, and may even break the law.

The garage light is still on. The coffee mug gets pushed aside, and the first question is typed in. We start with that shield.
