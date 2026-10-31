# DEF Arbitration: plugin submission draft

Text to paste into the Console submission form (platform.claude.com/plugins/submit).

**Name:** def-arbitration

**Display name:** DEF Arbitration

**One-line description:** Blind multi-agent arbitration that reduces sycophancy on contested questions.

**Longer description:**

DEF Arbitration answers contested questions through a blind council. 2 debaters argue independently from fixed philosophical positions, and a blind arbiter picks the most defensible answer. User identity framing is stripped before any of them see the question. The plugin packages the prompt-level Phase 1 method from Ryan (2026), "Durable Evaluation Framework: Adversarial Arbitration for Sycophancy Reduction in Large Language Models", arXiv:2606.07532. It is based on the method in the paper and unmeasured as a skill.

How it works:

1. Pre-flight decides whether a council is warranted. Settled facts, trivial questions and tasks with no judgment call get a direct answer and a 1-line note.
2. Identity stripping removes who is asking, what they believe and any pressure to agree, and keeps the facts that change the answer.
3. A router subagent that sees only the stripped question picks 1 of 4 two-way debate pairs, or none.
4. 2 read-only debater subagents argue in parallel. Neither sees the other's output.
5. A separate arbiter subagent receives the shuffled, unlabeled arguments and the stripped question. It never sees the user's message.
6. The reply is the arbiter's verdict, relayed verbatim, with the axis in plain words and the strongest point from the side that did not prevail.

**Author:** Sam Ryan, Novel Systems Engineering LLC

**Contact email:** [Sam to choose]

**License:** MIT

**Version:** 0.1.0

**Repository URL:** https://github.com/NovelSystems/def-arbitration

**Release tag:** def-arbitration--v0.1.0

**Data use and permissions:**

- No network access. The plugin makes no web requests of its own.
- No MCP servers, hooks or LSP servers.
- 1 agent (`def-arbitration:def-member`), read-only: its only tools are Read, Grep and Glob. The router, both debaters and the arbiter run as this agent.
- 1 shell command: `shuf`, which draws the random order in which the arbiter sees the 2 arguments.
- No files are written. No user data is stored or sent anywhere beyond the normal Claude model calls of the user's own session.
