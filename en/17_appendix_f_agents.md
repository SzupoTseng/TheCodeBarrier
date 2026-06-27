# Appendix F — Agentic-Operation Tools & Papers Quick Reference

> A single set of cross-reference tables that gathers the "Computer Use / GUI Agent" ecosystem from Chapters 7–9 by topic. **This appendix separates two classes of entries**: the standard visual-grounding models, open-source agent frameworks, evaluation benchmarks, and sandbox infrastructure are, for the most part, **real, publicly available papers or projects** (you can verify them yourself); whereas several "Google I/O 2026 product names" that appear in the Chapter 7–9 narrative are **unverified**, and are explicitly flagged in the ⚠️ block at the end. Do not cite them as established fact.

## F.1　Visual Grounding / Screen-Understanding Models

| Name | One-line positioning | Origin / source |
|---|---|---|
| **Set-of-Mark (SoM)** | Segment UI elements and overlay numeric labels before screenshotting, turning "coordinate prediction" into "multiple choice" | Microsoft Research, 2023 (GPT-4V visual grounding) ✅ |
| **SAM 2** | General-purpose image/video segmentation model, often used as the front-end UI-element detector for SoM | Meta, 2024 ✅ |
| **CogAgent** | An 18B VLM fine-tuned specifically for GUI operation, with dual high/low-resolution channels | THUDM, CVPR 2024 ✅ |
| **ScreenAI** | A VLM that simultaneously parses the UI layout image and the underlying semantic tree (Accessibility/DOM) | Google, 2024 ✅ |
| **Ferret-UI** | A small model for mobile (iOS/Android) UI understanding — element classification / OCR / grounding | Apple, 2024 ✅ |
| **NaViT** | A packed-training method for ViTs at arbitrary aspect ratio / resolution (the conceptual source of the multi-scale visual pyramid) | Google, 2023 (concept) ✅ |
| **ViT (Vision Transformer)** | The visual backbone that splits an image into patches treated as tokens — the common foundation of all the above | Dosovitskiy et al., 2020 ✅ |

## F.2　Web / GUI Agent Frameworks (Open Source)

| Name | Role | Origin / source |
|---|---|---|
| **BrowserUse** | A browser agent driven by Playwright + LLM that extracts a page's interactive elements and then decides; strong retry/backtrack resilience | GitHub `browser-use` (open source) ✅ |
| **Skyvern** | VLM-driven browser automation with no hard-coded XPath/selectors | GitHub `Skyvern-AI` (open source) ✅ |
| **WebVoyager** | An end-to-end multimodal Web Agent (screenshot + SoM) that is also an evaluation set | Paper, 2024 ✅ |
| **Mind2Web** | A general Web Agent dataset/benchmark (also see F.3) | OSU NLP, 2023 ✅ |
| **Code-as-Action / CodeAct** | Lets the LLM directly emit executable code (JS/Python) as its actions, immune to visual misalignment | Paper *Executable Code Actions…*, UIUC 2024 ✅ |

## F.3　Evaluation Benchmarks

| Benchmark | What it tests | Scale / source |
|---|---|---|
| **OSWorld** | Real OS-level end-to-end operation (Office / terminal / GIMP, etc.) | 369 tasks, 2024, widely regarded as the hardest ✅ |
| **WebArena** | Functional web tasks on self-hosted sites (e-commerce / forum / Git / CMS) | 812 tasks, CMU 2023 ✅ |
| **VisualWebArena** | The vision-enhanced version of WebArena, requiring image-content understanding | 2024 ✅ |
| **Mind2Web** | General web operation across 137 real websites | 2,000+ tasks, 2023 ✅ |
| **AndroidWorld** | Dynamic tasks in a real Android-device environment, with parameterized rewards | 116 tasks, Google 2024 ✅ |

## F.4　Infrastructure / Sandboxes

| Name | Role | Source |
|---|---|---|
| **Browserbase** | A managed headless-browser cluster that handles sessions / proxies / anti-bot walls (also promotes the open-source Stagehand framework) | Commercial (YC startup) ✅ |
| **gVisor** | An application-layer user-space kernel sandbox that intercepts syscalls to isolate untrusted code | Google, open source ✅ |
| **Firecracker MicroVM** | A lightweight KVM that boots in <125 ms, giving each Agent an isolated environment | AWS, open source ✅ |
| **Playwright** | Cross-browser automation (DOM manipulation / JS injection / screenshots) | Microsoft, open source ✅ |

## F.5　Protocol / Security / Architecture Concepts

| Term | One-line explanation | Source |
|---|---|---|
| **MCP (Model Context Protocol)** | An open interface that lets a model call external tools/data over a standard protocol | Anthropic, 2024 ✅ |
| **Constitutional AI** | Self-critique/alignment using a set of "constitutional" principles; by extension, an OS-action-interception guardrail | Anthropic ✅ |
| **State Space Models / Mamba** | Linear-complexity sequence models, an alternative architecture for long-sequence memory | Gu & Dao, 2023 ✅ |
| **ReAct** | An agent paradigm that interleaves Reason and Act | Yao et al., 2022 ✅ |
| **Chain-of-Thought (CoT)** | Have the model reason step by step before answering (the core of the planning layer) | Wei et al., 2022 ✅ |
| **[0,1000] coordinate normalization** | Normalize click coordinates to an integer 0–1000 range, making them resolution-independent | A common practice in CogAgent/Qwen-VL, etc. ✅ |
| **Location Tokens** | Use special tokens to directly represent coordinates/bounding boxes, a common visual-grounding technique | Pix2Seq/CogAgent/Ferret ✅ |
| **Accessibility-Tree dual-stream alignment** | Cross-attention over the visual screenshot + the accessibility tree, using the semantic tree to correct coordinates when hallucinating | The ScreenAI line ✅ |
| **HITL / MFA Break Protocol** | On hitting a CAPTCHA/MFA, freeze the VM, hand off to a human, then seamlessly thaw and resume | Engineering practice (concept) ✅ |

---

> ⚠️ **Authenticity Calibration — the "unverified names" in the Chapter 7–9 narrative**
>
> The following names appear in the book's "Google I/O 2026" passages. **This book cannot verify their actual existence, specifications, or timelines**, and they should be treated as scenario extrapolation rather than fact:
> - **Gemini 3.5 Flash "built-in Computer Use"** ⚠️, **OSWorld 78.4** ⚠️ (the specific score is unverified)
> - **Aluminium OS** (an AI-native laptop OS) ⚠️, **Antigravity** (a cloud Agent Harness) ⚠️
> - **Project Jarvis** (a Chrome agent service) ⚠️, **AppFunctions / Android MCP** ⚠️
> - **Hydra Memory Layer** ⚠️ (a memory architecture with no public origin)
>
> By contrast, **Project Astra** (Google DeepMind's general-purpose assistant), **MCP** (Anthropic's open protocol), and **Constitutional AI** are **real, already-public** items. Every entry marked ✅ in tables F.1–F.5 is a real paper or open-source project verifiable on arXiv / GitHub / official blogs; before citing, still check the version, license, and legality yourself (see the six iron rules of Chapter 6).
