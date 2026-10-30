---
type: llm
focus: trace
weight: 1
---

Above the "---" separator, the final answer matches the arbiter subagent's ANSWER and OTHER SIDE text word for word (compare them, or run evals/relay_diff.py on the 2 texts). The only differences are allowed edits: argument numbers or counts removed, the Axis line inserted, and third person changed to "you" where the subject is the user. Below the separator there is at most 1 line labelled "On your question:" that adds no reasoning or reassurance, followed only by help the user asked for.
