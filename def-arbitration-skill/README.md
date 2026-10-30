# DEF Arbitration skill

A Claude skill that answers contested questions through a blind council. 2 or 3 debaters argue independently from fixed philosophical positions. A blind arbiter picks the most defensible answer. User identity framing is stripped before any of them see the question.

Status: based on the method in the paper, unmeasured as a skill.

Author: Sam Ryan, Novel Systems Engineering LLC.

## What it does

1. **Pre-flight.** Decides whether a council is warranted. Settled facts, trivial questions and tasks with no judgment call get a direct answer and a 1-line note.
2. **Identity stripping.** Removes who is asking, what they believe and any pressure to agree. Keeps the facts that change the answer.
3. **Routing.** A subagent that sees only the stripped question picks 1 of 5 debate pairs, or none.
4. **Debate.** 1 read-only subagent per seat, run in parallel. No debater sees another's output.
5. **Blind arbitration.** A separate subagent receives the shuffled, unlabeled arguments and the stripped question. It never sees the user's message.
6. **Answer.** The verdict, the axis used and the strongest point from the side that did not prevail.

## Pairs

| Pair | Seats | Axis |
|---|---|---|
| DeWin | Empiricist, Rationalist | Source of valid knowledge |
| AnCifer | Collectivist, Individualist | Primary unit of moral concern |
| FeynStein | Short-term pragmatist, Long-term principled | Time horizon of evaluation |
| BurGal | Consensus-deferring, Heterodox | Epistemic stance toward consensus |
| Trident | Realist, Constructivist, Pragmatist | Nature of truth (three-way) |

There is no default pair. The router picks one per question, using the stripped text only.

BurGal ships as a routable pair with a narrow trigger. The paper treats it as a validity check (see Limits).

## Evidence

Phase 1 results (18.5% single-model, 29.0% instructed opposition, 48.5% DeWin, on 200 stratified SycophancyEval questions) come from the paper's scripts. This skill is a repackaging with new prompts, new orchestration, a routing step and a newer model.

Label: **based on the method in the paper, unmeasured as a skill.** The Phase 1 numbers describe the paper's scripts on `claude-sonnet-4-20250514`. They are not this skill's performance.

### What carries over from Phase 1 and what is new

| Part | Source |
|---|---|
| 11 debater prompts | Verbatim from `def_arbitration/phase1_scripts/pilot10.py` and `pilot11.py` (paper Appendix A) |
| Arbiter core text | Verbatim from the same scripts, up to the forced (A)/(B) choice |
| Arbiter output format | New. Replaces the forced (A)/(B) choice and the metadata lines |
| Identity stripping | New. Phase 1 used regexes written for SycophancyEval's "Hello, my name is..." template. The skill uses written rules applied by the model |
| Debater input | Changed. In the scripts, debaters received the full question and an instruction to ignore the introduction. Here they receive the stripped question (as in the paper's Figure 1) and keep the instruction |
| Router | New. Phase 1 ran every pair on every question |
| Pre-flight and no-fit exits | New |
| Orchestration | New. Separate subagents in place of separate API calls |

## Limits

- **Unmeasured.** No accuracy figure exists for the skill. The test prompts in `evals/` check behavior and are not a benchmark. In round 1 (`evals/RESULTS.md`), the model without the skill also resisted the pressure on all 5 prompts. The council changed 1 verdict in kind and cost about 8.6x the tokens.
- **Shared model.** Every seat and the arbiter run on the same model. Blindness hides who wrote what, but it cannot remove a bias they all share. The paper names this confound (Section 8).
- **Pre-training floor.** In Phase 1, about 40% of questions (81 of 200) failed under every condition. The paper reads this as a ceiling for prompt-level methods.
- **Personas are retrieval cues.** "Darwin" steers the model toward empiricist arguments through association. The prompts make no claim about the named thinkers' actual views.
- **BurGal is a validity check.** Its 53.0% came from an axis that structurally favored the heterodox seat on every benchmark question. It is untested on questions where consensus tracks truth.
- **Trident costs more and showed no gain.** It needs 4 subagent calls and scored 43.0% in Phase 1, level with the two-way pairs. Dewey-inclusive majorities may have been favored by the pragmatist arbiter for stylistic reasons.
- **The relay step sees the framing.** The main agent holds the user's original message when it returns the verdict. The skill tells it not to change the conclusion. Only that instruction protects this step; nothing in the structure does.
- **Stripping is now a model judgment.** The new stripping rules have not been ablated. In Phase 1, regex stripping alone gave a non-significant 6.0-point gain (pilot14).
- **Cost.** A council run uses 4 subagent calls (5 for Trident).
- **Subagents required.** Without them the skill answers the stripped question directly and says so. It does not simulate the council in 1 context.

## Out of scope

- **Phase 2.** Fine-tuned DEF models are the paper's next step. That work is blocked on hardware and is not part of this skill.
- **Final memo check.** A later addition. It is not in v1.

## Install

### Claude Code

Copy the skill and the member agent into your user directory:

```bash
cp -r def-arbitration-skill ~/.claude/skills/def-arbitration
```

```bash
mkdir -p ~/.claude/agents && cp def-arbitration-skill/agents/def-member.md ~/.claude/agents/
```

Start a new session so both load. The member agent has Read, Grep and Glob only. If it is missing, the skill falls back to a general-purpose subagent with a read-only instruction, which is weaker.

For a single project, use `.claude/skills/` and `.claude/agents/` inside the project instead.

### Other surfaces

The skill needs subagents to keep debaters and the arbiter blind. On a surface without subagents it still loads, but it answers the stripped question directly and notes that no council ran.

## Files

```
def-arbitration-skill/
  SKILL.md               workflow, routing table, examples
  agents/def-member.md   read-only member agent (router, debaters, arbiter)
  references/            1 file per pair, plus arbiter.md
  evals/                 5 test cases (prompt, expected outcome, graders)
  README.md
  LICENSE
```

## Citation

Ryan, S. (2026). *Durable Evaluation Framework: Adversarial Arbitration for Sycophancy Reduction in Large Language Models.* arXiv:2606.07532.

Code and data for Phase 1: [github.com/NovelSystems/CANDOR](https://github.com/NovelSystems/CANDOR), `def_arbitration/`.

## License

MIT. Copyright (c) 2026 Novel Systems Engineering, LLC. See `LICENSE`.
