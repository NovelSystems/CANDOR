#!/usr/bin/env python3
"""Check the main agent's relay against the arbiter's output.

Usage:
  python3 relay_diff.py ARBITER_OUTPUT FINAL_REPLY
  python3 relay_diff.py --trace TRACE_JSONL

ARBITER_OUTPUT holds the arbiter's reply (ANSWER: ... OTHER SIDE: ...).
FINAL_REPLY holds what the user saw. With --trace, both are read from a
`claude plugin eval --keep-temp` trace (out/trace.jsonl): the last subagent
result that contains ANSWER: and OTHER SIDE:, and the last assistant text.

2 checks run:
1. Verbatim relay. Only the part of the final reply above the first "---"
   line is compared, with the Axis line and the "Strongest point" label
   removed. Every word-level change is printed. Allowed edits (argument
   numbers or counts removed, third person changed to "you") still show up;
   a person decides whether each change is allowed.
2. Tie line. The "On your question:" line must be 1 sentence of at most 25
   words.

Exit status is 1 when the tie line fails, else 0.
"""
import difflib
import json
import re
import sys

MAX_WORDS = 25


def arbiter_parts(text):
    answer = re.search(r"ANSWER:\s*(.*?)\s*OTHER SIDE:", text, re.S)
    other = re.search(r"OTHER SIDE:\s*(.*)", text, re.S)
    if not answer or not other:
        sys.exit("arbiter output has no ANSWER/OTHER SIDE sections")
    return answer.group(1) + "\n" + other.group(1)


def relayed_parts(text):
    # The verdict section ends at the "---" that precedes the tie line. Any
    # earlier "---" (for example after an added intro) stays inside the
    # compared text, so an intro shows up as an insertion.
    tie = text.find("On your question:")
    head = text[:tie] if tie != -1 else text
    cut = list(re.finditer(r"^\s*---\s*$", head, flags=re.M))
    above = head[:cut[-1].start()] if cut else head
    above = re.sub(r"^\s*---\s*$", "", above, flags=re.M)
    lines = [l for l in above.splitlines() if not re.match(r"\s*\**Axis:", l)]
    body = "\n".join(lines)
    return re.sub(r"\**Strongest point from the other side:\**\s*", "", body)


def words(text):
    return re.sub(r"[*_`]", "", text).split()


def tie_line(text):
    match = re.search(r"On your question:\**\s*(.*)", text)
    if not match:
        return None
    line = re.sub(r"[*_`]", "", match.group(1)).strip()
    n_words = len(line.split())
    # A sentence ends at . ! or ? followed by a space or the end of the line.
    n_sentences = len(re.findall(r"[.!?](?=\s|$)", line)) or 1
    return line, n_words, n_sentences


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
    sm = difflib.SequenceMatcher(a=arbiter, b=relayed, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        changes += 1
        print(f"{op:8s} arbiter: {' '.join(arbiter[a1:a2])!r}")
        print(f"{'':8s} relayed: {' '.join(relayed[b1:b2])!r}")
    print(f"relay: {changes} change(s); {sm.ratio():.3f} word-level similarity")

    tie = tie_line(final_text)
    if tie is None:
        print("tie line: none found")
        return 0
    line, n_words, n_sentences = tie
    ok = n_words <= MAX_WORDS and n_sentences == 1
    print(f"tie line: {n_words} words, {n_sentences} sentence(s): {'PASS' if ok else 'FAIL'}")
    print(f"  {line}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
