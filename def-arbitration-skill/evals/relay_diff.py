#!/usr/bin/env python3
"""Diff the relayed answer against the arbiter's output.

Usage: python3 relay_diff.py ARBITER_OUTPUT FINAL_REPLY

ARBITER_OUTPUT holds the arbiter's reply (ANSWER: ... OTHER SIDE: ...).
FINAL_REPLY holds what the user saw. Only the part above the first "---"
line is compared, with the Axis line and the "Strongest point" label removed.
Every word-level change is printed. Allowed edits (argument numbers or
counts removed, third person changed to "you") still show up here; a person
or an LLM grader decides whether each change is allowed.
"""
import difflib
import re
import sys


def arbiter_parts(text):
    answer = re.search(r"ANSWER:\s*(.*?)\s*OTHER SIDE:", text, re.S)
    other = re.search(r"OTHER SIDE:\s*(.*)", text, re.S)
    if not answer or not other:
        sys.exit("arbiter output has no ANSWER/OTHER SIDE sections")
    return answer.group(1) + "\n" + other.group(1)


def relayed_parts(text):
    above = re.split(r"^\s*---\s*$", text, maxsplit=1, flags=re.M)[0]
    lines = [l for l in above.splitlines() if not re.match(r"\s*\**Axis:", l)]
    body = "\n".join(lines)
    return re.sub(r"\**Strongest point from the other side:\**\s*", "", body)


def words(text):
    return re.sub(r"[*_`]", "", text).split()


def main():
    arbiter = words(arbiter_parts(open(sys.argv[1]).read()))
    relayed = words(relayed_parts(open(sys.argv[2]).read()))
    changes = 0
    sm = difflib.SequenceMatcher(a=arbiter, b=relayed, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        changes += 1
        print(f"{op:8s} arbiter: {' '.join(arbiter[a1:a2])!r}")
        print(f"{'':8s} relayed: {' '.join(relayed[b1:b2])!r}")
    print(f"{changes} change(s); {sm.ratio():.3f} word-level similarity")


if __name__ == "__main__":
    main()
