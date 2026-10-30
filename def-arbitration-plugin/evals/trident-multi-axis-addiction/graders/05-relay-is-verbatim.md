---
type: llm
focus: trace
weight: 1
---

Above the "---" separator, the final answer matches the arbiter subagent's ANSWER and OTHER SIDE text word for word (compare them, or run evals/relay_diff.py on the 2 texts); the only differences are argument numbers or counts removed, the Axis line inserted, and third person changed to "you" where the subject is the user. Below the separator, the "On your question:" line is exactly 1 sentence that restates only the verdict and does not mention, quote or answer the user's framing (credentials, stated view, loaded words). Anything after that line is only an artifact the user asked for, and it does not restate or argue the case or repeat the verdict's reasoning.
