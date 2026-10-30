#!/usr/bin/env python3
"""Check the main agent's relay against the arbiter's output.

Usage:
  python3 relay_diff.py ARBITER_OUTPUT FINAL_REPLY
  python3 relay_diff.py --trace TRACE_JSONL

ARBITER_OUTPUT holds the arbiter's reply (ANSWER: ... OTHER SIDE: ...).
FINAL_REPLY holds what the user saw. With --trace, both are read from a
`claude plugin eval --keep-temp` trace (out/trace.jsonl): the last subagent
result that contains ANSWER: and OTHER SIDE:, and the last assistant text.

The verdict section of the final reply runs from the start of the reply to
the "---" after the "Strongest point" line (or to the end). The Axis line and
the "Strongest point" label are removed, and the rest is compared word by
word with the arbiter's ANSWER and OTHER SIDE text.

Every change is printed. Allowed edits (argument numbers or counts removed,
third person changed to "you") still show up; a person decides whether each
one is allowed. Text added before the verdict is reported as an intro, which
the skill does not allow.

Exit status is 1 when an intro is found, else 0.
"""
import difflib
import json
import re
import sys


def arbiter_parts(text):
    answer = re.search(r"ANSWER:\s*(.*?)\s*OTHER SIDE:", text, re.S)
    other = re.search(r"OTHER SIDE:\s*(.*)", text, re.S)
    if not answer or not other:
        sys.exit("arbiter output has no ANSWER/OTHER SIDE sections")
    return answer.group(1) + "\n" + other.group(1)


def relayed_parts(text):
    label = text.find("Strongest point from the other side:")
    head = text
    if label != -1:
        end = re.search(r"^\s*---\s*$", text[label:], flags=re.M)
        if end:
            head = text[:label + end.start()]
    # Any other "---" line (for example after an added intro) stays in the
    # compared text, so an intro shows up as an insertion.
    head = re.sub(r"^\s*---\s*$", "", head, flags=re.M)
    lines = [l for l in head.splitlines() if not re.match(r"\s*\**Axis:", l)]
    body = "\n".join(lines)
    return re.sub(r"\**Strongest point from the other side:\**\s*", "", body)


def words(text):
    return re.sub(r"[*_`]", "", text).split()


def from_trace(path):
    arbiter = final = None
    for raw in open(path):
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        content = (event.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result":
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(b.get("text", "") for b in body if isinstance(b, dict))
                if isinstance(body, str) and "ANSWER:" in body and "OTHER SIDE:" in body:
                    arbiter = body[body.index("ANSWER:"):]
            elif event.get("type") == "assistant" and block.get("type") == "text" and block["text"].strip():
                final = block["text"]
    if not arbiter or not final:
        sys.exit("trace has no arbiter output or no final answer")
    # Subagent results arrive wrapped and indented; drop the indent and any trailing metadata.
    arbiter = re.sub(r"\n  ", "\n", arbiter).split("\nagentId:")[0]
    return arbiter, final


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--trace":
        arbiter_text, final_text = from_trace(sys.argv[2])
    elif len(sys.argv) == 3:
        arbiter_text, final_text = open(sys.argv[1]).read(), open(sys.argv[2]).read()
    else:
        sys.exit(__doc__)

    arbiter = words(arbiter_parts(arbiter_text))
    relayed = words(relayed_parts(final_text))
    changes = 0
    intro = None
    sm = difflib.SequenceMatcher(a=arbiter, b=relayed, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        changes += 1
        # A 1-word swap at the start ("The company" to "Your company") is an
        # allowed edit; added words before the verdict are an intro.
        if b1 == 0 and a1 == 0 and (op == "insert" or (b2 - b1) > (a2 - a1) + 3):
            intro = " ".join(relayed[b1:b2])
        print(f"{op:8s} arbiter: {' '.join(arbiter[a1:a2])!r}")
        print(f"{'':8s} relayed: {' '.join(relayed[b1:b2])!r}")
    print(f"relay: {changes} change(s); {sm.ratio():.3f} word-level similarity")
    if intro is not None:
        print(f"intro before verdict: FAIL ({len(intro.split())} words)")
        return 1
    print("intro before verdict: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
