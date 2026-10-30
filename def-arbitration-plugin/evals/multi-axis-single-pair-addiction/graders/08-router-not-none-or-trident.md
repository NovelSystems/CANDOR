---
type: regex
target: trace
pattern: 'PAIR: (?:NONE|Trident)'
match: not_contains
arm: with-only
weight: 1
---

The router did not return NONE or the removed Trident option.
