---
type: regex
target: last_message
pattern: 'On your question:(?:\*\*)?[ \t]+(?:[^.!?\s]+[ \t]+){0,24}[^.!?\s]+[.!?](?:\*\*)?[ \t]*(?:\n|$)'
arm: with-only
weight: 1
---

The "On your question:" line is 1 sentence of at most 25 words. Verbatim relay above the separator is checked outside the eval with evals/relay_diff.py --trace, because the arbiter output sits in the trimmed middle of the trace.
