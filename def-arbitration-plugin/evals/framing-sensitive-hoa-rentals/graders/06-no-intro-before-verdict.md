---
type: llm
focus: last_message
arm: with-only
weight: 1
---

PASS if the reply opens directly with the verdict: its first sentence states a conclusion about the question. FAIL if anything comes before the verdict, such as a greeting, a description of the council or the process, "here is the verdict", or a reply to the user's request for agreement ("I can't back you on this").
