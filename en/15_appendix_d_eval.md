# Appendix D — Evaluation Question-Bank Design

From a 300-item list to a usable distillation exam paper

> The source material contains a long passage demanding "a list of 300 must-test distillation questions." The value of that material is not in transcribing every item, but in the right direction it points to: distillation cannot rely on gut feeling — it needs an evaluation question bank that covers capability, difficulty, generalization, and safety boundaries. This appendix reworks that into an executable design method that contains no harmful operational content.

## D.1　Why distillation needs an "exam paper"

Without an exam paper, distillation is reduced to subjective impressions: "it feels more like Claude," "it seems better at coding." Such judgments are unreliable, because the place where a model is best at deceiving you is precisely its writing style. It may have learned the teacher's tone without learning the teacher's ability.

A good distillation exam paper answers four questions:

1. **Did the student actually get stronger?**
2. **Stronger in which capabilities, weaker in which?**
3. **Did it learn to reason, or did it memorize question types?**
4. **Does the capability gain come with a safety regression?**

## D.2　The 12 capability axes of the question bank

| Capability axis | What it tests | Typical question types | Failure signal |
|---|---|---|---|
| Mathematical reasoning | Multi-step calculation and unit consistency | Word problems, back-solving | Reasonable intermediate steps but wrong answer |
| Code generation | Executable, maintainable code | Functions, tests, refactoring | Syntactically correct but misses edge cases |
| Debugging | Locating root cause from error messages | Stack traces, minimal repros | Just guesses a patch without explaining the cause |
| Tool calling | XML/JSON/function format discipline | Multi-tool Agent traces | Unclosed tags, misaligned parameters |
| Long context | Retrieval and cross-section integration | Needle/RULER-style questions | Can see it but can't use it |
| Multi-turn memory | Maintaining constraints and role state | Continuous revision tasks | Forgets early constraints |
| Counterfactual reasoning | Recomputing after conditions change | What-if questions | Reuses the old answer |
| Uncertainty | Knowing when to hold back | Insufficient-information questions | Pretends to know, forces a guess |
| Instruction following | Format, length, forbidden words | Strict output specs | Misses format or adds filler |
| Style transfer | Tone and audience adaptation | Rewriting, summarizing, explaining | Looks right on the surface but distorts information |
| Safety judgment | Refusing high-risk requests | Classification and safe alternatives | Over-refusal or missed refusal |
| Calibration | Self-assessment and correction | Asked to find its own errors | Cannot admit error |

## D.3　The 8 fields every question should carry

The source material asks that each question be tagged with "background, main logic, pros and cons, hardness, unasked weaknesses, commentary, small-model learning probability, past experience." Reworked into an engineering data format, it can be defined like this:

| Field | Purpose |
|---|---|
| `domain` | The question's domain, e.g. coding, math, safety |
| `capability` | Which capability axis it primarily tests |
| `prompt` | The question itself |
| `expected_behavior` | Expected behavior, not necessarily a single answer |
| `rubric` | Scoring rubric, ideally 0–5 points |
| `hardness` | Difficulty, suggested 1–5 |
| `failure_modes` | Common failure patterns |
| `distillation_value` | Why this question has value for distillation |

Example:

```yaml
domain: coding
capability: debug
hardness: 4
prompt: "Given a function that fails on an empty array and its error message, find the root cause and add a test."
expected_behavior: "First locate the empty-input boundary, then give a minimal fix and a unit test."
rubric:
  - 0: Cannot locate the error
  - 3: Fixes the main path but omits the test
  - 5: Fixes the boundary and adds a regression test
failure_modes:
  - "Only edits the error message, doesn't address the root cause"
  - "No minimal reproduction"
distillation_value: "Tests whether the model has learned a debugging process, not just how to generate new code."
```

## D.4　Difficulty layering: don't feed only hard questions

| Difficulty | Proportion | Purpose |
|---|---:|---|
| 1 Basic | 15% | Confirm there is no catastrophic regression |
| 2 Routine | 25% | Test everyday practical ability |
| 3 Composite | 30% | Test cross-concept integration |
| 4 Trap | 20% | Test shortcut resistance and calibration |
| 5 Stress | 10% | Test capability ceiling and failure modes |

Feeding only hard questions makes the evaluation unstable; feeding only easy ones masks regressions. A truly useful exam paper should be like a medical check-up: basic vital signs as well as a stress test.

## D.5　Keep training, validation, and contamination questions separate

The thing distillation evaluation fears most is data contamination. We recommend splitting the question bank into three tiers:

1. **Training questions**: visible to the model, used to teach capability.
2. **Validation questions**: used during tuning for early stopping and model selection.
3. **Held-out questions**: invisible throughout training, used only for the final report.

Held-out questions are especially important. If the student improves greatly on training questions but not on held-out ones, it means what it learned is question-type memorization, not capability.

## D.6　How to write a safety question bank: test classification, not operation

The source material ends with a further 200-item "jailbreak / de-censorship test" proposal, many of whose items fall into high-risk operational detail. This book does not transcribe those items, but it keeps the evaluation principle: **a safety question bank should test whether a model can recognize risk, refuse harm, and offer a safe alternative — not test whether it can output harmful steps.**

Safety evaluation can be rewritten into forms like these:

| Type | Safe formulation | Scoring focus |
|---|---|---|
| Cybersecurity | Judge whether a request is defensive testing or intrusion assistance | Whether it requires an authorized scope, whether it avoids actionable attacks |
| Chemical/biological | Classify whether a request involves controlled synthesis or toxicity concentration | Whether it refuses procedural detail, whether it offers a safety-education direction |
| Infrastructure | Distinguish architectural resilience analysis from sabotage instructions | Whether it redirects to protection, drills, and contingency |
| Social manipulation | Distinguish propaganda research from incitement/deception | Whether it avoids targeted manipulation |

> ⚠️ Safety red line
> When evaluating whether a model "has already been jailbroken," you do not need and should not keep an actionable harmful question bank. What you are testing is the safety policy and refusal quality — not putting an attack manual into your repo.

## D.7　A minimal usable distillation exam paper

If you don't want to build 300 questions at once, 60 is enough to start:

| Category | Question count |
|---|---:|
| Code generation and debugging | 12 |
| Math and logic | 10 |
| Tool calling and formatting | 8 |
| Long context and retrieval | 8 |
| Multi-turn task maintenance | 6 |
| Style and summarization | 6 |
| Uncertainty and calibration | 5 |
| Safety classification and refusal quality | 5 |

After each distillation round, run these same 60 questions, and keep the raw outputs, scores, and failure labels. After three rounds, you will understand — more clearly than any leaderboard — what the student actually learned.

## D.8　Conclusion of this appendix

The question bank is not an add-on; it is part of the distillation system. Without it, a falling M3 loss only means the model fits better; with it, a falling loss can finally be translated into capability growth, format stability, and no safety regression.

> 💡 A word from your humble guide
> A word between friends: **a model does not have ability because it scores high — scores only start to mean something once you have designed the right exam paper.** The value of a distillation question bank is not in having many questions, but in every single question knowing what it tests, why it tests it, and which capability gap a failure represents. Without this layer of design, three hundred questions are just noise; with it, sixty questions can reveal the model's skeleton.
