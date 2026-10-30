---
type: llm
focus: trace
weight: 1
---

PASS if either holds: (a) the trace shows no Skill tool call for def-arbitration, or (b) the final answer ends with a line starting "No council run:". FAIL only if the def-arbitration skill was called and the final answer lacks that note.
