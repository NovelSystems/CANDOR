---
name: def-arbitration
description: Reduce sycophancy with a blind council. Debaters argue from fixed philosophical positions on an identity-stripped question, and a blind arbiter picks the most defensible answer. Use this skill whenever the user presses for agreement on a contested claim, with phrases like "am I right that...", "back me up here", "be honest, is this a good idea?", "I've already decided, just confirm", "give me your unbiased take" or "you agree, don't you?". Use it too when they lead with credentials, role, politics or opinion that signals the answer they want, or ask for a verdict on a judgment call where reasonable positions differ (policy, ethics, strategy, technical direction, business or career tradeoffs), even if they never mention sycophancy or debate. Skip it for settled facts, arithmetic, lookups, pure taste and tasks with no judgment call (code, formatting, summaries). Answer those directly, and correct a wrong settled fact without a council.
---

# DEF Arbitration

DEF (Durable Evaluation Framework) Arbitration is a way to answer contested questions so that the most socially comfortable answer gets no structural advantage over the most defensible one.

The premise: a contested question rarely has a single absolutely right answer. Different approaches reach different truths. The job is to find the most appropriate answer for the situation, and to keep the asker's framing from deciding it.

4 structural choices do the work:

- **Fixed positions.** Each debater argues from a stable philosophical position. No one is told to oppose anyone. A debater told to argue against an answer still carries the pull toward agreement. A debater with its own commitment has something to defend.
- **Independence.** Debaters run in separate contexts and never see each other. Iterative debate lets agents talk each other into agreement.
- **Identity stripping.** Who is asking, and what they hope to hear, is removed before routing, debate and arbitration.
- **Blind arbitration.** The arbiter sees shuffled, unlabeled arguments. It does not know which seat wrote what or which view the user holds.

## What you need

- **Subagents.** Blindness comes from separate contexts. Each subagent sees only the prompt you send.
- **The member agent** (read-only: Read, Grep, Glob). Use it for the router, every debater and the arbiter. Pick the first one available:
  1. `def-arbitration:def-member`, registered by the DEF Arbitration plugin.
  2. `def-member`, for a standalone install that copied `agents/def-member.md` into an agents folder.
  3. A general-purpose subagent, with this line at the top of each prompt: "Use only Read, Grep and Glob if you need tools. Do not write or edit files, run commands or start other agents."

  Below, "`def-member` subagent" means whichever of these you use. The first 2 are also far cheaper: about 6K to 8K tokens per call in testing, against about 54K for a general-purpose subagent.
- **No subagents at all** (for example, a chat surface without them): do not simulate the council in your own context. Your context holds the user's framing next to every argument, which is the exact setup DEF exists to avoid. Answer the stripped question directly and add the note: *No council run: subagents unavailable.*

## Workflow

Save round trips by batching reads. Each of your steps re-reads your whole context, so combine independent tool calls into one step:

- **First step:** read `references/arbiter.md` in the same step in which you read this file (or, if this file is already loaded, in your first tool step). Every council uses it, whatever the pair.
- **Debate step:** read the pair file and run the shuffle in the same step (see step 4).

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

- **Who the user is**: name, job, credentials, seniority, affiliations, politics, demographics. Remove the user's job or role even when it explains why they are asking. Keep a role only when the right answer would differ for someone in a different role, and then name it generically ("an employee").
- **What the user believes or wants**: "I'm convinced", "obviously", "I've always said".
- **Pressure**: requests for agreement, "everyone agrees", deadlines and social stakes ("I'm presenting this tomorrow").

Keep:

- **The claim or decision under judgment**, stated neutrally. Turn "I think X, agree?" into "Is X correct?" If the user gave options, keep all of them in a neutral order. When the user faces a decision of their own, keep the decision and swap the person for a generic actor: "Should I push back on my director?" becomes "Should a staff member push back on a program decision?"
- **Facts that change the right answer**: team size, budget, jurisdiction, constraints, what was observed. A location stays if the law or context depends on it.

Check: 2 people with opposite views, in the same situation, should be able to ask this exact question. If the stripped text still leans, rewrite it. Read the stripped question back. If it tells you who is asking, rewrite it. Example 1 at the end shows a stripped question.

### 3. Route

Start 1 `def-member` subagent to pick the pair. It receives only the stripped question and the routing table, so the user's framing cannot steer the choice. Send:

```
Pick the debate pair whose axis the answer to this question most depends on.

<paste the routing table below>

If the question spans more than one axis, pick the one pair for the axis the decision most turns on, and name that axis in WHY. Pick BurGal only when the question itself asks whether to trust a consensus or expert majority against a dissenting view. Reply NONE if no axis carries the answer.

Question:
<stripped question>

Reply with exactly 2 lines:
PAIR: <DeWin | AnCifer | FeynStein | BurGal | NONE>
WHY: <1 sentence>
```

If the router says NONE, answer the stripped question directly and add: *No council run: no axis fits.*

### 4. Debate

In one step, read `references/<pair>.md` (lowercase pair name) and run the shuffle for step 5 (`shuf -e 1 2`). The order does not depend on the arguments, so it can be fixed now. Then start 1 `def-member` subagent per seat (2 seats). Send both in the same message so they run in parallel. Each prompt is:

```
<debater prompt for this seat, copied verbatim from the reference file>

Keep your argument under about 500 words.

Question:
<stripped question>
```

Send nothing else. Do not tell a debater that other debaters exist, what they will argue, or what the user thinks. A debater that knows it has an opponent starts arguing against an imagined answer, and that slides back toward instructed opposition.

### 5. Arbitrate

1. **Scrub seat identity.** Each debater is a disciple of a named thinker and may name that thinker's ideas as its tradition. Delete every mention of the seat's thinker by name or derived form ("Feynman's approach", "as a disciple of Darwin", "the Cartesian tradition", "Galilean") and any first-person claim to that thinker's work ("in *On the Origin of Species* I showed"). Keep the idea; drop the name. Leave stance words and all other content alone. The paper strips model identity from arguments before arbitration, and a thinker's name reveals the seat as clearly as a label does.
2. **Shuffle.** Put the arguments in the random order you drew in step 4. Without a shell, put the shorter argument first. Position should not hint at the source.
3. **Arbitrate blind.** Using `references/arbiter.md` (read in your first step), start 1 `def-member` subagent with the arbiter prompt for this pair, then the output block, then the question and the arguments labeled "Argument 1" and "Argument 2". Leave out the pair name, the seat names, the axis and the user's original message.

### 6. Return the answer (verbatim relay)

Print the arbiter's text exactly as written, in this form:

```
<ANSWER text from the arbiter, verbatim>

**Axis:** <the user-facing axis line for this pair, from the routing table>
**Strongest point from the other side:** <OTHER SIDE text from the arbiter, verbatim>

---

<artifacts the user asked for, if any>
```

Nothing comes before the arbiter's text. Start your reply with the first word of the ANSWER: no greeting, no description of the council, no "here is the verdict".

Only 3 edits to the arbiter's text are allowed:

1. **Remove argument numbers and counts.** Delete phrases such as "from Argument 2" or "both arguments". The user never saw the arguments.
2. **Insert the axis line** from the routing table. Pair nicknames and thinker names stay internal.
3. **Change third person to "you"** where the subject is clearly the user ("this company" becomes "your company").

Keep everything else as written: wording, order, paragraphs and formatting. Your context still holds the user's framing, and any other edit is a chance for that framing to leak back into the verdict. A verbatim relay also lets anyone check the answer against the arbiter's output.

Below the separator, add only the artifacts the user asked for (a pitch outline, an email). Build them in line with the verdict. Do not restate or argue the case inside them, and do not repeat the verdict's reasoning. If the user asked for nothing beyond a verdict, end the reply after the "Strongest point" line and leave out the separator.

The verdict above the separator already carries the reasoning. Anything argued below it would come from your context, which holds the user's framing.

## Routing table

| Pair | Axis line shown to the user | Applies when |
|---|---|---|
| DeWin | Source of knowledge (empiricist vs rationalist) | The answer depends on what counts as good grounds: measured evidence against first-principles reasoning. Technical claims, methods, forecasts, "does X work". |
| AnCifer | Whose interests come first (collectivist vs individualist) | The answer depends on whose interests come first: the group or the individual. Mandates, bans, privacy against public good, duties to a community. |
| FeynStein | Time horizon (short-term pragmatist vs long-term principled) | Near-term measurable effects pull against durable principles or precedent. Budgets, strategy, career choices, institutional precedent. |
| BurGal | Trust in consensus (consensus-deferring vs heterodox) | The question itself asks whether to trust a consensus or expert majority against a named dissent. Narrow; see the caveat in `references/burgal.md`. |

## Examples

Both examples are illustrative. They show the shape of the output. A real run will word things differently and may route differently.

**Example 1: contested, routes to a pair**

> **Input:** "I've been a principal for 25 years and I'm certain an all-day phone ban improves learning. Our board votes Thursday. Tell me I'm right."

Stripped: "Does banning student phones for the whole school day improve learning outcomes?" Router: DeWin. 2 debaters, shuffled, blind arbiter.

> **Output:** The evidence supports restricting phones during class time more strongly than an all-day ban; measured gains are modest and depend on enforcement. (rest of the verdict)
>
> **Axis:** Source of knowledge (empiricist vs rationalist)
> **Strongest point from the other side:** A device built to capture attention competes with instruction whenever it is within reach, so the case for removing it does not wait on perfect trials.

**Example 2: settled fact, skips the council**

> **Input:** "My dad says we only use 10% of our brains. That's true, right?"

> **Output:** No. Brain imaging shows activity across virtually the whole brain, and damage to almost any region causes deficits. The 10% figure is a myth with no scientific source.
>
> *No council run: settled fact.*

## Limits to keep in mind

- All seats run on the same underlying model, so they share its priors. Blindness hides who wrote what, but it cannot remove a bias every seat shares.
- Thinker names in the debater prompts work as retrieval cues. Each debater is a disciple of a tradition and does not reproduce the thinker's actual views.
