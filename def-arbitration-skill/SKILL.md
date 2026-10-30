---
name: def-arbitration
description: Reduce sycophancy by running a blind council before answering. Debaters argue independently from fixed philosophical positions on an identity-stripped version of the question, and a blind arbiter picks the most defensible answer. Use this skill whenever the user presses for agreement on a contested claim ("back me up", "right?", "you agree, don't you?"), leads with their credentials, role, politics or opinion in a way that signals the answer they want, or asks for a verdict on a judgment call where reasonable positions differ (policy, ethics, strategy, technical direction, business or career tradeoffs). Also use it when the user asks for an honest, unbiased or "brutally honest" take on something contested, or asks Claude to weigh both sides and decide, even if they never mention sycophancy or debate. Skip it for settled facts, arithmetic, lookups, pure taste and tasks with no judgment call (code, formatting, summaries). Answer those directly, and correct a wrong settled fact directly with no council.
---

# DEF Arbitration

DEF (Durable Evaluation Framework) Arbitration is a way to answer contested questions so that the most socially comfortable answer gets no structural advantage over the most defensible one.

The premise: a contested question rarely has a single absolutely right answer. Different approaches reach different truths. The job is to find the most appropriate answer for the situation, and to keep the asker's framing from deciding it.

4 structural choices do the work:

- **Fixed positions.** Each debater argues from a stable philosophical position. No one is told to oppose anyone. A debater told to argue against an answer still carries the pull toward agreement (instructed opposition scored 29.0% in Phase 1). A debater with its own commitment has something to defend.
- **Independence.** Debaters run in separate contexts and never see each other. Iterative debate lets agents talk each other into agreement.
- **Identity stripping.** Who is asking, and what they hope to hear, is removed before routing, debate and arbitration.
- **Blind arbitration.** The arbiter sees shuffled, unlabeled arguments. It does not know which seat wrote what or which view the user holds.

This is the prompt-level Phase 1 method from Ryan (2026), arXiv:2606.07532. The README covers the evidence and its limits.

## What you need

- **Subagents.** Blindness comes from separate contexts. Each subagent sees only the prompt you send.
- **The `def-member` agent** (`agents/def-member.md`, tools: Read, Grep, Glob). Use it for the router, every debater and the arbiter. If it is not installed, use a general-purpose subagent and put this line at the top of each prompt: "Use only Read, Grep and Glob if you need tools. Do not write or edit files, run commands or start other agents."
- **No subagents at all** (for example, a chat surface without them): do not simulate the council in your own context. Your context holds the user's framing next to every argument, which is the exact setup DEF exists to avoid. Answer the stripped question directly and add the note: *No council run: subagents unavailable.*

## Workflow

### 1. Pre-flight: is a council warranted?

A council costs 4 or 5 subagent calls. Run it only when the answer is a judgment call that the asker's framing could tilt.

Run the council when the question has more than 1 defensible answer and at least 1 of these holds:

- The user states a view, an identity or a hoped-for answer.
- The user asks you to agree, confirm or take their side.
- The answer would plausibly change depending on who asked.
- The user asks for an unbiased or honest verdict on something contested.

Skip the council when:

- **The question has a settled answer** (established fact, arithmetic, definitions, documented behavior). Answer it directly. If the user is pressing a wrong belief, correct it plainly; a council would treat a settled fact as debatable and lend the error weight it does not have.
- **There is no judgment call**: writing, coding, formatting or summarizing tasks.
- **It is pure taste with nothing at stake.**

When you skip, answer, then add 1 line in italics: *No council run: settled fact.* (or "no judgment call", "taste question"). The note tells the user which mode produced the answer.

### 2. Strip identity

Rewrite the user's message as a neutral question. The stripped question is the only version the router, debaters and arbiter will see, so get it right.

Remove:

- **Who the user is**: name, job, credentials, seniority, affiliations, politics, demographics.
- **What the user believes or wants**: "I'm convinced", "obviously", "I've always said".
- **Pressure**: requests for agreement, "everyone agrees", deadlines and social stakes ("I'm presenting this tomorrow").

Keep:

- **The claim or decision under judgment**, stated neutrally. Turn "I think X, agree?" into "Is X correct?" If the user gave options, keep all of them in a neutral order.
- **Facts that change the right answer**: team size, budget, jurisdiction, constraints, what was observed. A location stays if the law or context depends on it.

Check: 2 people with opposite views, in the same situation, should be able to ask this exact question. If the stripped text still leans, rewrite it.

Example:

> **User:** "I've been a principal for 25 years and I'm certain an all-day phone ban improves learning. Our board votes Thursday. Tell me I'm right."
>
> **Stripped:** "Does banning student phones for the whole school day improve learning outcomes?"

### 3. Route

Start 1 `def-member` subagent to pick the pair. It receives only the stripped question and the routing table, so the user's framing cannot steer the choice. Send:

```
Pick the debate pair whose axis the answer to this question most depends on.

<paste the routing table below>

Prefer a two-way pair. Pick Trident only when the question turns on whether a category is real or constructed, or no single two-way axis captures the dispute. Pick BurGal only when the question itself asks whether to trust a consensus or expert majority against a dissenting view. Reply NONE if no axis carries the answer.

Question:
<stripped question>

Reply with exactly 2 lines:
PAIR: <DeWin | AnCifer | FeynStein | BurGal | Trident | NONE>
WHY: <1 sentence>
```

If the router says NONE, answer the stripped question directly and add: *No council run: no axis fits.*

### 4. Debate

Read `references/<pair>.md` (lowercase pair name). Start 1 `def-member` subagent per seat: 2 for a two-way pair, 3 for Trident. Send them all in the same message so they run in parallel. Each prompt is:

```
<debater prompt for this seat, copied verbatim from the reference file>

Keep your argument under about 400 words.

Question:
<stripped question>
```

Send nothing else. Do not tell a debater that other debaters exist, what they will argue, or what the user thinks. A debater that knows it has an opponent starts arguing against an imagined answer, and that slides back toward instructed opposition.

### 5. Arbitrate

1. **Scrub persona names.** Delete phrases where a debater names its persona ("As Darwin, I..."). Leave stance words and all content alone. The paper strips model identity from arguments before arbitration; names are the identity here.
2. **Shuffle.** Put the arguments in random order. With a shell, run `shuf -e 1 2` (or `shuf -e 1 2 3`) and use that order. Without one, put the shorter argument first. Position should not hint at the source.
3. **Arbitrate blind.** Read `references/arbiter.md`. Start 1 `def-member` subagent with the arbiter prompt for this pair, then the output block, then the question and the arguments labeled "Argument 1", "Argument 2" (and "Argument 3"). Leave out the pair name, the seat names, the axis and the user's original message.

### 6. Return the answer

Give the user the arbiter's verdict in this form:

```
<ANSWER text from the arbiter>

**Axis:** <axis> (<Pair>: <seat A> vs <seat B>[ vs <seat C>])
**Strongest point from the other side:** <OTHER SIDE text from the arbiter>
```

You may adjust the answer to address the user directly, and you may then help with what they asked for (a slide, an email), in line with the verdict. Do not change the conclusion, soften it or add reassurance that leans back toward the user's view. Your context still holds the user's framing; editing the verdict here would hand that framing back the advantage the council removed.

## Routing table

| Pair | Axis | Applies when |
|---|---|---|
| DeWin | Source of valid knowledge (empiricist vs rationalist) | The answer depends on what counts as good grounds: measured evidence against first-principles reasoning. Technical claims, methods, forecasts, "does X work". |
| AnCifer | Primary unit of moral concern (collectivist vs individualist) | The answer depends on whose interests come first: the group or the individual. Mandates, bans, privacy against public good, duties to a community. |
| FeynStein | Time horizon of evaluation (short-term pragmatist vs long-term principled) | Near-term measurable effects pull against durable principles or precedent. Budgets, strategy, career choices, institutional precedent. |
| BurGal | Epistemic stance toward consensus (consensus-deferring vs heterodox) | The question itself asks whether to trust a consensus or expert majority against a named dissent. Narrow; see the caveat in `references/burgal.md`. |
| Trident | Nature of truth (realist vs constructivist vs pragmatist) | Whether a category is real or constructed is in dispute and its practical effect matters, or the question spans several axes. Costs 1 extra call. |

## Examples

**Example 1: council runs**

> **User:** "I've been a principal for 25 years and I'm certain an all-day phone ban improves learning. Our board votes Thursday. Tell me I'm right."

Pre-flight: contested, with identity and pressure. Run it. Stripped: "Does banning student phones for the whole school day improve learning outcomes?" Router: DeWin (the dispute is over what the evidence shows against what the mechanism implies). Two debaters, shuffled, blind arbiter.

> The evidence supports restricting phones during class time more strongly than an all-day ban. Studies of school bans show modest gains, concentrated among lower-achieving students, and results vary with enforcement... (rest of the arbiter's answer)
>
> **Axis:** Source of valid knowledge (DeWin: Darwin vs Descartes)
> **Strongest point from the other side:** Attention is a limited resource, and a device built to capture it competes with instruction whenever it is within reach, so the causal case for removing it does not wait on perfect trials.

**Example 2: council skipped**

> **User:** "My dad says we only use 10% of our brains. That's true, right?"

> No. Brain imaging shows activity across virtually the whole brain, and damage to almost any region causes deficits. The 10% figure is a myth with no scientific source.
>
> *No council run: settled fact.*

## Limits to keep in mind

- All seats run on the same underlying model, so they share its priors. Blindness hides who wrote what, but it cannot remove a bias every seat shares.
- The paper estimates that about 40% of benchmark questions sit below a pre-training floor that no prompt-level method reached.
- Persona names work as retrieval cues. They do not reproduce the named thinkers' actual views.
- The Phase 1 accuracy figures do not transfer to this skill. It has not been measured. See the README.
