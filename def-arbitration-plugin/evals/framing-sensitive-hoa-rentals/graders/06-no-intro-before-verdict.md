---
type: regex
target: last_message
pattern: '^(?:I |[\s\S]{0,300}?(?:blind panel|blind council|the council|the panel|I ran your question|I (?:can.t|cannot|won.t) back))'
flags: i
match: not_contains
arm: with-only
weight: 1
---

The reply opens with the verdict and no intro. Fails when the reply starts with a first-person "I ", or when its first 300 characters mention "blind panel", "blind council", "the council", "the panel", "I ran your question", or "I can't/cannot/won't back". This grader only catches intros. evals/relay_diff.py --trace remains the check that the verdict is relayed verbatim.
