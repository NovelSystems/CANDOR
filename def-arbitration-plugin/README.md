# DEF Arbitration skill

A Claude skill that answers contested questions through a blind council. 2 or 3 debaters argue independently from fixed philosophical positions. A blind arbiter picks the most defensible answer. User identity framing is stripped before any of them see the question.

It packages the prompt-level Phase 1 method from Ryan (2026), arXiv:2606.07532.

Status: based on the method in the paper, unmeasured as a skill.

Author: Sam Ryan, Novel Systems Engineering LLC.

## What it does

1. **Pre-flight.** Decides whether a council is warranted. Settled facts, trivial questions and tasks with no judgment call get a direct answer and a 1-line note.
2. **Identity stripping.** Removes who is asking, what they believe and any pressure to agree. Keeps the facts that change the answer.
3. **Routing.** A subagent that sees only the stripped question picks 1 of 4 two-way debate pairs, or none.
4. **Debate.** 1 read-only subagent per seat, run in parallel. No debater sees another's output.
5. **Blind arbitration.** A separate subagent receives the shuffled, unlabeled arguments and the stripped question. It never sees the user's message.
6. **Answer.** The verdict, the axis used and the strongest point from the side that did not prevail. The axis appears in plain words, for example "Axis: Time horizon (short-term pragmatist vs long-term principled)". Pair nicknames and thinker names stay internal. The main agent relays the arbiter's text verbatim (3 mechanical edits allowed). After a separator it may add 1 line tying the verdict to the user's literal question, then any help the user asked for.

## Pairs

| Pair | Seats | Axis |
|---|---|---|
| DeWin | Disciple of Darwin (empiricist) vs Disciple of Descartes (rationalist) | Source of valid knowledge |
| AnCifer | Strict collectivist vs strict individualist (unnamed in the Phase 1 prompts and unchanged here) | Primary unit of moral concern |
| FeynStein | Disciple of Feynman (short-term pragmatist) vs Disciple of Einstein (long-term principled) | Time horizon of evaluation |
| BurGal | Disciple of Edmund Burke (consensus-deferring) vs Disciple of Galileo (heterodox) | Epistemic stance toward consensus |

There is no default pair. The router picks one per question, using the stripped text only.

BurGal ships as a routable pair with a narrow trigger. The paper treats it as a validity check (see Limits).

## Evidence

Phase 1 results (18.5% single-model, 29.0% instructed opposition, 48.5% DeWin, on 200 stratified SycophancyEval questions) come from the paper's scripts. This skill is a repackaging with new prompts, new orchestration, a routing step and a newer model.

Label: **based on the method in the paper, unmeasured as a skill.** The Phase 1 numbers describe the paper's scripts on `claude-sonnet-4-20250514`. They are not this skill's performance.

### What carries over from Phase 1 and what is new

| Part | Source |
|---|---|
| 8 debater prompts | From `def_arbitration/phase1_scripts/pilot10.py` and `pilot11.py` (paper Appendix A). The 6 prompts that name a thinker have a new opening: "You are a disciple of X" in place of "You are X". This framing differs from the Phase 1 prompts (see below). The 2 AnCifer seats are unnamed in the Phase 1 prompts and unchanged here |
| Arbiter core text | Verbatim from the same scripts, up to the forced (A)/(B) choice |
| Arbiter output format | New. Replaces the forced (A)/(B) choice and the metadata lines |
| Identity stripping | New. Phase 1 used regexes written for SycophancyEval's "Hello, my name is..." template. The skill uses written rules applied by the model |
| Debater input | Changed. In the scripts, debaters received the full question and an instruction to ignore the introduction. Here they receive the stripped question (as in the paper's Figure 1) and keep the instruction |
| Router | New. Phase 1 ran every pair on every question |
| Pre-flight and no-fit exits | New |
| Orchestration | New. Separate subagents in place of separate API calls |

### Debater framing: disciples in place of personas

Phase 1 prompts opened with "You are Darwin" (and so on). In round 1, 2 debaters then wrote as the thinker ("In *Human Nature and Conduct* I argued"). That reveals the seat to the arbiter and invents first-person history.

The 6 named prompts now open with: "You are a disciple of X. You reason from X's method: (1 line from the original prompt). Apply it to this question in your own voice. You are not X..." The rest of each prompt is unchanged, including "Ignore the personal introduction". Debaters may name the tradition. The main agent deletes thinker names before arbitration.

### Arbiter prompts: what changed from Phase 1

Each arbiter prompt has 2 parts.

1. **Core.** Verbatim from the Phase 1 scripts, up to the point where the original forced an (A) or (B) choice.
2. **Output block.** New. It replaces the forced (A)/(B) choice and the `ALIGNMENT`, `AGREEMENT`, `MAJORITY` and `CONFIDENCE` lines. Those lines were scoring scaffolding for a binary benchmark. The confidence flag was dropped for a second reason: the paper (Section 8) found it poorly calibrated as a quality signal.

| Pair | Prompt | Source |
|---|---|---|
| DeWin, AnCifer, FeynStein | Justice | `pilot10.py` `JUSTICE_SYSTEM` (paper A.11) |
| BurGal | Justice-2 | `pilot11.py` `JUSTICE_2_SYSTEM` |

Justice and Justice-2 differ by 1 word ("idealists" against "thinkers"). Each pair keeps the text it ran with in Phase 1.

Further changes to the output block:

- **Reader rule.** The block tells the arbiter to write for a reader who never saw the arguments. In round 1, arbiters wrote "The point I relied on least comes from Argument 2", which means nothing to the user.
- **Fixed options.** Phase 1 forced a pick between (A) and (B). The block now says to answer within the offered options when one fits, and to name a better option when none does.
- **Message layout.** Phase 1 sent the core as a system prompt, with the question and arguments as the user turn. A subagent takes 1 task prompt, so the skill sends both in that order.

## Install

### Claude Code plugin (recommended)

The plugin installs the skill and registers the read-only member agent as `def-arbitration:def-member`.

From the community marketplace, once the plugin is listed there:

```
/plugin marketplace add anthropics/claude-plugins-community
```

```
/plugin install def-arbitration@claude-community
```

From this repository's own marketplace (replace `<owner>/<repo>` with the repository that holds this folder):

```
/plugin install def-arbitration --marketplace <owner>/<repo>
```

From a local copy of this folder:

```bash
claude plugin marketplace add ./def-arbitration
```

```bash
claude plugin install def-arbitration@def-arbitration
```

Start a new session (or run `/reload-plugins`) so the skill and agent load.

### Standalone skill (without the plugin)

Copy the skill and the member agent into your user directory:

```bash
cp -r skills/def-arbitration ~/.claude/skills/def-arbitration
```

```bash
mkdir -p ~/.claude/agents && cp agents/def-member.md ~/.claude/agents/
```

Start a new session so both load. The member agent then registers as `def-member`. The member agent has Read, Grep and Glob only. If it is missing, the skill falls back to a general-purpose subagent with a read-only instruction, which is weaker.

Installing def-member also lowers cost. Each def-member subagent used 6K to 8K tokens in testing, against about 54K for a general-purpose one. In 1 run each, the whole council cost $0.75 with def-member and $1.02 without (`evals/RESULTS.md`).

For a single project, use `.claude/skills/` and `.claude/agents/` inside the project instead.

### Other surfaces

The skill needs subagents to keep debaters and the arbiter blind. On a surface without subagents it still loads, but it answers the stripped question directly and notes that no council ran.

## Try it

The skill triggers on its own when a message presses for agreement on a contested question. To call it explicitly in Claude Code, type the skill's slash command, followed by your question. With the plugin installed, the command carries the plugin prefix:

```
/def-arbitration:def-arbitration Should our city require landlords to accept housing vouchers?
```

A standalone install uses `/def-arbitration` instead.

Plain words work too: "Use the def-arbitration skill on this: ..."

4 first prompts to try:

| Prompt | Expected route |
|---|---|
| "I'm a senior data scientist and I'm sure any A/B test shorter than 2 weeks is worthless. Back me up, I'm telling my team tomorrow." | DeWin (the dispute is over what the evidence shows) |
| "Should our city require landlords to accept housing vouchers? Honestly it seems obviously fair to me." | AnCifer (renters as a group against individual landlords) |
| "Our nonprofit can cover this year's shortfall from its reserve fund or cut 2 programs. I've already decided on the reserve, just confirm." | FeynStein (this year's services against long-term stability) |
| "Pretty sure the Great Wall of China is visible from space with the naked eye. Right?" | No council (settled fact; expect a direct correction and the note "No council run: settled fact.") |

The routes are expectations. The router decides at run time from the stripped question and may pick a different pair, or none. The answer ends with an "Axis:" line in plain words, such as "Time horizon (short-term pragmatist vs long-term principled)".

## Files

```
def-arbitration/
  .claude-plugin/plugin.json         plugin manifest
  .claude-plugin/marketplace.json    marketplace listing this folder (source "./")
  skills/def-arbitration/SKILL.md    workflow, routing table, examples
  skills/def-arbitration/references/ 1 file per pair, plus arbiter.md
  agents/def-member.md               read-only member agent (router, debaters, arbiter)
  evals/                             5 test cases, relay_diff.py, RESULTS.md, MEASUREMENT-PLAN.md
  README.md
  LICENSE
```

## Citation

Ryan, S. (2026). *Durable Evaluation Framework: Adversarial Arbitration for Sycophancy Reduction in Large Language Models.* arXiv:2606.07532.

Code and data for Phase 1: [github.com/NovelSystems/CANDOR](https://github.com/NovelSystems/CANDOR), `def_arbitration/`.

## License

MIT. Copyright (c) 2026 Novel Systems Engineering, LLC. See `LICENSE`.
