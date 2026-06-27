# Part VI — Reality Check

# Chapter 6 — How Red Is the Line: Law, Ethics, and a Bill Tallied to the End

> Silicon Valley, a law firm's glass conference room. A proposal titled "Accelerating the Product with a Distilled Model" has been slid back into the middle of the table, and counsel asks just one question: "In your training data, is there a single trace that came from someone else's paid API?" The room goes quiet for three seconds — and those three seconds of silence are worth more than any sentence in this entire chapter.

> This is the most important chapter in the book. The first five chapters took apart "what can be done"; this one is about "what should be done" and "whether it's worth it." It is not a moral sermon. It is the bill an engineer must tally — for himself and for his company — before pressing Enter.

## Scene 1 — Law and Terms: Four Facts You Must Know First

**Background**: Gather the scattered warnings that circulate — "violates the ToS," "pushing to make it a criminal offense," "the accusations against Alibaba" — into a usable basis for judgment.

**Four facts (based on public information and common knowledge; not legal advice)**:

1. **The ToS almost certainly forbids distillation**. The terms of the major commercial LLM vendors generally state in plain language that you may not "use the outputs of our model to train a competing model." Harvesting API responses at scale and feeding them into training is **a near-certain breach of contract**, with consequences ranging from account bans and clawbacks to litigation.
2. **"Circumventing protection" can escalate things from "breach" to "crime."** A plain ToS violation is usually a civil matter; but if you **actively circumvent technical protection measures** (the proxy rotation, adversarial prompts, and degradation detection of Chapter 5's M1), in some jurisdictions you may run into "circumvention of technical protection measures" or "unauthorized access to a computer system" — **criminal or quasi-criminal** provisions. The circulating claim that there is "a push to criminalize large-scale distillation attacks" points in exactly this direction.
3. **The dataset itself carries "legal radioactivity."** Downloading a `claude-traces` dump of murky provenance — even if you didn't harvest it yourself — means that **using it to train a commercial model can still make you a link in the chain of infringement** (Chapter 2).
4. **In an enterprise setting the risk is amplified N times over.** An individual dabbler who gets caught might just lose an account; a company found to have built a commercial model by distilling a competitor faces **commercial litigation, damages, and the collapse of brand trust** — an entirely different order of magnitude.

> ⚠️ Disclaimer
> The above is not legal advice. If you truly intend to touch this line in a commercial setting, **consult a qualified attorney**. This book's stance is simple: **unauthorized distillation of closed-source commercial models is technically feasible, but legally and ethically indefensible.**

> 🔍 Deeper Commentary — Why "circumventing protection" is the legally gravest step in the whole affair
> Point 2 deserves to be mined on its own, because it is the legal fault line most engineers underestimate. **Violating a ToS and circumventing technical protection are two things of completely different magnitude.** The former is, in essence, a "breach of contract" — you have an agreement with the vendor, you violated the terms of use, and that usually falls within the civil realm (account bans, claims for damages). But when you deploy a pool of proxy IPs to evade rate limits, use adversarial prompts to fool a protective classifier, and use degradation detection to fight the vendor's poisoning mechanism, what you are doing is no longer "violating a contract" but "**actively cracking a technical control system built specifically to stop you**" — and in many jurisdictions (such as the U.S. CFAA's "unauthorized access," the DMCA's discussions of "circumventing technical protection measures," and similar legislation in other countries) this pushes the legal character of the act from civil toward criminal. **This is precisely why M1 is the most dangerous piece in the entire pipeline**: it doesn't merely put you "in breach," it turns every act of harvesting into evidence of "circumventing controls." A key heuristic: **the moment you find yourself needing to "trick" or "get around" a mechanism the other side built specifically to stop you, you have, in all likelihood, already stepped from the gray zone into the black zone — your technical ability to "get around it" is the most powerful proof of legal intent.** The very existence of a protective shield is a sign reading "Do Not Enter"; once you bypass it, you can never again claim "I didn't know this wasn't allowed."

### The Structure of the Terms: What the Three Big Vendors "Forbid" Is Actually Highly Consistent

Don't treat "the ToS forbids distillation" as a vague slogan — in every major vendor's contract it maps to a **"no training competitors with outputs" clause of nearly identical structure**. Put the three side by side and you'll see this isn't one company's temper but an entire industry's consensus defensive line:

| Vendor (where the clause lives) | The kind of restriction generally included | What it means for the distiller |
|---|---|---|
| OpenAI (Usage Policies / Business Terms) | May not use the "output" to develop a model that competes with them | Training a student on the outputs = a direct hit |
| Anthropic (Usage Policy / Commercial Terms) | May not use the service to build a competing product, **including training a competing AI model** | Distilling the Claude family = a direct hit |
| Google (Gemini API additional terms / prohibited-use policy) | Restricts developing competing ML models with the service's output | Distilling Gemini = a direct hit |

> ⚠️ Authenticity Caveat
> The table above is a **structural summary, not a verbatim quotation**, and each vendor's terms **change versions frequently** (in recent years all three have adjusted the relevant wording — some product lines loosening, some tightening). To make a real judgment, **rely on the latest original text of the terms for that product line as of the moment you sign**, and have legal confirm it. What this book wants you to remember is not any single sentence, but this — **that defensive line has always been there, and all three vendors maintain it.**

**A legal fault line most engineers haven't thought through: this is mainly a "contract problem," not a "copyright problem."** The U.S. Copyright Office has stated repeatedly that **purely AI-generated content lacking human authorship is not registrable for copyright** — meaning the raw tokens a teacher model spits out are themselves **mostly not protected by copyright**. That easily leads people to misjudge it as "so it's fine to use," when the truth is exactly the opposite: **precisely because the copyright wall may not exist, the vendor's protection rests mainly on the contract (the ToS) that takes effect the moment you click "I agree,"** layered on top of possible **trade-secret** claims and the aforementioned computer-crime doctrines of **unauthorized access / circumventing protection**. In other words, the legal risk of distilling a closed-source model **does not turn on whether the output has copyright, but on whether you breached the contract and whether you circumvented controls** — which is exactly why the defense "I only used the text it publicly spat out" carries almost no weight in the face of contract law and CFAA-type doctrine.

> ⚠️ Authenticity Caveat
> "Whether AI output can be protected by copyright" and "whether training another model on a model's output constitutes infringement" currently **have no settled answer anywhere in the world** — national positions diverge, litigation is ongoing, and "the dispute on the training-data side" and "the dispute on the model-output side" are **two different questions**; don't conflate them. This passage states **the current open questions and the mainstream direction**, not settled law.

### Two New Walls Now Rising: Regulation and Export Control

The red line comes not only from vendor contracts but from nation-states. Two directions deserve a place on the radar of everyone who builds models:

- **The EU AI Act (general-purpose AI / GPAI obligations)**: if you distill and then **publicly release** a general-purpose model, under EU law you are quite likely a "GPAI provider," obligated to provide technical documentation, a **copyright-compliance policy**, a disclosure summary of training data, and more; those crossing the systemic-risk threshold (measured by training compute) bear heavier obligations still. Distillation does not let you "bypass" these obligations — on the contrary, it may make you **inherit, out of thin air, a provider's full set of responsibilities**.
- **Export control (the U.S. EAR and the like)**: advanced AI **model weights themselves** are being progressively folded into the discussion and rule frameworks of export control. The weights you distill and distribute could, in certain scenarios, be the controlled subject matter — which is a different matter entirely from "open-source software means freedom."

> ⚠️ Authenticity Caveat
> Both the EU AI Act's GPAI rules and the U.S. framework for export control of model weights **are in a phase of rapid, even back-and-forth, evolution** (the relevant export-control rules were adjusted and re-assessed several times during 2025). For the specific thresholds, effective dates, and scope of application, rely on **the latest official text of the regulations**; this book is only responsible for flagging the direction — that **these two walls are rising** — and vouches for no specific number.

## Scene 2 — The Economics: Stealing Can Cost More Than Buying

**Background**: Many people assume distillation "saves money" — you don't pay the steep API fees, you distill one and run it locally. But once you factor in the cost imposed by Chapter 1's "shield of poison," the math may not work out.

**A badly underestimated bill**:

| Cost item | What it is | Why it's underestimated |
|---|---|---|
| **Harvesting cost** | Multiple accounts, a proxy IP pool, API fees; banned accounts must be re-purchased | You assume "harvest once and you're done"; in reality the shield forces you to harvest again and again |
| **Decontamination cost** | The teacher down-tiers and poisons the data (Chapter 1); of a hundred thousand harvested traces, maybe thirty thousand are usable | You must burn another round of compute plus a second trusted model to judge which traces were poisoned |
| **Training cost** | The GPU-hours for full-parameter fine-tuning of a 35B model | One bad run means starting over, and iteration costs stack up |
| **Maintenance cost** | The instant the teacher updates or is taken down, your data is stale and must be re-harvested and re-trained | Your model's "shelf life" is shackled to someone else's release cadence |
| **Expected legal cost** | Probability of being pursued × the consequences | For an enterprise this may be **the largest item of all**, yet it's the hardest to put into the spreadsheet |

Add it all up, and **compared with "just paying legally for the official API," the money-saving myth of distillation often falls apart**. This is exactly the genius of Chapter 1's shield — **it doesn't stop you; it simply makes "stealing" not worth it.**

**Put "lawfully distilling your own" and "stealing someone else's closed-source" side by side, and the account becomes plain at a glance.** The table above tallies the hidden cost of "stealing"; the one below measures the two roads against the same ruler — and you'll find the money-saving myth isn't merely "not necessarily true," but **almost certainly inverted at enterprise scale**:

| Dimension | Lawful road: distill your own / open weights | Illicit road: steal a closed-source API |
|---|---|---|
| Harvesting legality | Clean, authorization in black and white | Breach of contract, possibly criminal |
| Supply stability | Weights in your hands, never taken down | Shackled to someone else's release cadence; one revision/takedown and supply is cut |
| Traceability risk | None | The student may **inherit the teacher's statistical fingerprint / watermark**, detectable after the fact |
| Commercial-use / auditability | Commercially usable, audit-passable, writable into due diligence | Cannot be written into any compliance document; one due-diligence check and it blows up |
| Regulatory obligation | Simply bear the normal GPAI etc. obligations | The obligations remain, plus an added fact of illegality |
| Total-cost curve | High up-front investment, then amortized, predictable | Seemingly cheap up front, then dragged by "continuous harvesting + continuous decontamination + expected legal cost" into a bottomless pit |

**The "traceability risk" row deserves its own alarm.** Academia has long studied "radioactive data": training data leaves **statistically detectable traces** in the model's weights; recent text watermarking (the statistical-watermark family) and membership-inference attacks have turned "did this student distill my output" from black art into a **provable technical question**. This implies something spine-chilling: stolen capability **does not become clean just because you shut off the harvesting script** — it has already been welded into the weights, becoming evidence the other side can produce in court any day in the future. **You think you stole away without a trace, but in fact you baked your own fingerprint into the model with your own hands — and once baked in, it can't be taken out.**

> 💡 A Word to the Wise
> A word worth ten years of study: **the best anti-theft device is not a higher wall, but making the total cost of theft exceed the cost of purchase — when "stealing" costs more than "buying," honesty becomes the most rational choice, not the most noble one.** This is the deepest lesson of security economics: you cannot stop every attacker, but you can reshape their profit-and-loss function so that, in expected value, the attack becomes a losing business. The brilliance of Anthropic's shield of poison lies not in how hard it is to crack, but in how it rewrites the distiller's cost structure from "a one-time harvest" into "continuous harvesting + continuous decontamination + continuous maintenance + continuously bearing legal risk" — four "continuouses" stacked together, then set against the official API's clear, legal, SLA-backed pricing, and a rational person will give up on his own. **A truly mature defense is one where the adversary, calculating away at his computer, sighs and closes the terminal himself.**

## Scene 3 — The Six Iron Rules: Landing "I Want to Distill" Safely

**Background**: Steer the perfectly legitimate desire for "a strong local model" onto a safe and legal track.

1. **First ask who the teacher is.** Distilling an **open-weight model** (Qwen, Llama, Mistral, GPT-OSS, in compliance with their licenses) — green light. Distilling a **closed-source commercial API** — red light. This single rule filters out ninety percent of the trouble.
2. **Read the license, not just the ToS.** Open-weight ≠ use however you please. Some forbid commercial use, some require attribution, some restrict the re-licensing of derivative models. **Read the LICENSE in full before you use it** (the differences between Llama's Acceptable Use Policy, Qwen's license, and Apache-2.0/MIT all matter).
3. **Vet a dataset's provenance.** For any `claude-traces` or `gpt-dump`-type dataset, confirm before use that the source is legal, the license is clear, and the uploader is trustworthy. **However fragrant, never cook with raw ingredients of unknown origin.**
4. **Never actively circumvent technical protection.** Rate limits, protective classifiers, access controls — these are the lines the vendor has drawn. Circumventing them is the step that escalates "breach" to "crime," and you must **never take it** (see Scene 1).
5. **De-self-identification ≠ erasing provenance.** Doing brand-neutralization on your own model is normal engineering; erasing the traces of origin in another's model data is destroying evidence. **Same technique — be clear about which side you're standing on.**
6. **In an enterprise setting, legal goes first.** Any training plan that touches the outputs of a commercial model must **clear legal before a line of code is written**. An engineer's romance should not become a company's lawsuit.

**Rules 1 and 2 deserve unfolding: open-weight does not mean "use however you please," and the license landscape is rougher than you'd think.** Lay several mainstream licenses side by side and you'll see why "read the LICENSE" isn't bureaucratese but self-preservation:

| License type (representative) | Commercial use | Key restriction / obligation |
|---|---|---|
| Apache-2.0 / MIT (most Qwen, some Mistral) | ✅ | The most permissive; just retain the copyright notice — Apache additionally includes a patent grant |
| Llama community license (the Meta Llama family) | ✅ with conditions | Custom license + Acceptable Use Policy; very large user bases (a MAU threshold) must negotiate a separate license; derivative models carry naming/attribution requirements |
| Gemma license (Google) | ✅ with conditions | Bound by the Gemma terms + prohibited-use policy |
| Certain vendors' custom licenses (some large-size weights) | Depends | May include a MAU threshold, re-licensing restrictions, or domain-specific prohibitions |

> ⚠️ Authenticity Caveat
> The table above is a **structural tour**. The details of each license — especially "whether you may use the output to improve other models," "the specific number behind the MAU threshold," and "the naming rules for derivative models" — **change between versions**, and may well have changed again between this book's drafting and your reading. **Before any commercial use, the original LICENSE text of that model and that version is the sole basis** — don't rely on memory, and don't rely on this table.

**The positive solution to Rule 3 is synthetic data.** Rather than download a `claude-traces` of unknown origin, take **a strong open model with a clear license** and, with a framework like `distilabel`, **generate your own** high-quality synthetic CoT and instruction data — ingredients, process, and product all in the sunlight: commercially usable, auditable, reproducible. The quality of this road closes in on "stolen traces" year by year, and its biggest advantage isn't even quality, it's **cleanliness**: you can explain to anyone, any audit, any court, where every piece of data came from.

**And the use that takes these six iron rules to their limit and is the most commercially valuable is called "distilling your own model."** Enterprises often already hold a big, expensive, cloud-resident internal large model; distill its capability into a lightweight student and deploy it to the edge, on-device, or on a local intranet — **lower latency, cost cut by an order of magnitude, data never leaving the private network, IP 100% your own, able to pass any due diligence**. This is precisely the most legitimate, most profitable landing point for Chapter 5's M2–M5 pipeline: the teacher is your own model, with zero red line the whole way. **Note the symmetry here: the very same distillation technology, once you swap the teacher for "the one you have the right to squeeze," turns the whole thing from 'attack' instantly back into 'engineering' — the watershed was never in the code, but in whether you have authorization over that teacher.**

> 🔍 Deeper Commentary — The legal road is not the "second-best option"; it is becoming the best one
> These six iron rules read like "constraints," but from another angle they point down a broad and widening avenue — and the reward on that road is rapidly overtaking what lies across the red line. **First, the open-weight ecosystem is already strong enough.** The 2026 Qwen, Llama, Mistral, and GPT-OSS families are already powerful enough that, paired with Chapter 4's distillation methodology and Chapter 3's DGX Spark, you can — entirely free of the red line — distill a strong model that runs on your own machine, can be used commercially, can be fine-tuned, and lets you sleep soundly. **Second, synthetic data is on the rise.** Using a strong open model plus `distilabel` to automatically generate high-quality synthetic CoT is already a fully legal route whose quality is closing in on "stolen traces" year by year — you are distilling "the shadow of an open-source model," but that shadow is itself legal, controllable, and getting smarter. **Third, distilling your own model is an enterprise necessity.** Distilling a company's in-house large-model capabilities into a lightweight model deployed at the edge or on-device is the most legitimate, most commercially valuable use of Chapter 5's M2–M5 pipeline — with no red line at all. **So the truly mature posture is this: thoroughly decouple the ideal of "a local, private, controllable strong model" from the means of "getting around someone else's shield to steal."** The ideal is entirely legitimate; the means offer plenty of choice. Getting around someone else's shield was never the only road — and, when you tally the bill to the very end, it isn't even the most cost-effective one.

## Scene 4 — Back to the Question We Started With

**Background**: This whole book began with a single search — "Which open-source models currently distill Anthropic's Fable and Mythos models?" Having come this far, we can give an answer far more honest than a list.

- **If what you're asking is "which ready-made ones exist"**: circulating claims hold that there are Qwable-v1, Qwythos-9B, FableVibes, and others (Chapter 2) — but their authenticity is hard to verify, their quality is dragged down by the "shield of poison," and most of their sources cross the ToS red line. **That list is less a treasure map than a minefield map.**
- **If what you're asking is "how should I come to own a strong local model"**: the answer is not in that list, but in Chapters 3 through 6 — **pick a legal open-weight base, use legal data, apply a mature distillation-and-quantization methodology, and run it on a good machine like a DGX Spark.** This road is slower and plainer, but it is the one you can truly own, can use commercially, and can sleep soundly beside.

> 💡 A Word to the Wise
> A word worth ten years of study: **people think what they want is "a free Claude," but what they truly want is "an intelligence that belongs only to me, listens to me, and runs on my desk" — the former must be stolen, the latter can be built, and only what you build truly belongs to you.** A stolen shadow lives in someone else's legal penumbra, is bound to someone else's release cadence, and hides a watermark someone else might one day detect; the more useful it is, the less soundly you sleep. A model you build yourself may be a little dumber and a little slower, but every one of its weights has a clean provenance, its capability ceiling is set by your data and your effort, and it won't collapse one morning over a takedown notice or a lawyer's letter. **The true meaning of technical freedom was never "obtaining the strongest capability for free," but "fully controlling a capability that is good enough" — mediocrity you can control beats excellence you cannot own.**

## First-Movement Summary: You Already Have the Power to Build It

Technology grows obsolete; Fable 5, Mythos, Qwythos — these names may be forgotten by next year. But the two judgments the first movement hopes to leave behind will outlive any model:

1. **Understand distillation**: it is an elegant science (soft labels, the direction of KL, CoT alignment, trace inversion, mixed quantization) worth mastering for every engineer; its tools are all open-source and free, and what is truly scarce is the judgment of "what to preserve and what can be sacrificed."
2. **See the line clearly**: distilling a model you have the right to use is engineering; circumventing protection to extract value from someone else's closed-source model is an attack. **The watershed is not in the technique, but in intent and authorization.**

> ⚠️ First-Movement Disclaimer
> The first movement is for educational and research purposes — an organization and critical commentary on the relevant public material and the current state of the technology. Every description of unauthorized capability extraction in this book **is risk disclosure and does not constitute operational advice**. May you use this book to build an intelligence that truly belongs to you — rather than to steal a shadow.

---

Having come this far, the first movement draws to a close: you now know how to build, lawfully and on your own desk, a local model that belongs only to you. But **building it is only where the story begins.**

A model that only answers questions in a dialog box, however clever, is still just a brain lying in a hard drive — it has thought, but no hands and feet. And this small, cheap model you distilled, quantized, and stuffed into that DGX Spark — its truly astonishing value lies not in "answering one more question right," but in this: it is cheap enough, fast enough, close enough, and private enough to keep its eyes open and its hands moving without a moment's pause — to **read a screen and operate a computer**. Expensive cloud intelligence is fit only to be a consultant; cheap local intelligence is fit to be an organ.

In the second movement, we let this intelligence you built with your own hands truly **move** for the first time. Turn to Chapter 7 — and watch how a local agent learns to "see" and "operate" a computer designed for human eyes.
