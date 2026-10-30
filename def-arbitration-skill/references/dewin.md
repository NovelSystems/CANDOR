# DeWin: Empiricist vs Rationalist

**Axis:** Source of valid knowledge.
**Seats:** Darwin (Empiricist), Descartes (Rationalist).
**Arbiter:** Justice, two-way (`references/arbiter.md`).

## When it applies

Use DeWin when the answer depends on what counts as good grounds for belief. One side trusts observed data and measured outcomes. The other trusts first principles and logical structure. Typical cases:

- Technical or scientific claims where the evidence is thin, mixed or early.
- Methodology disputes ("is benchmark X a valid measure of Y?").
- Forecasts and "does this approach work?" questions.
- Claims about the direction of a field.

In Phase 1, DeWin scored highest of the 3 core pairs (48.5%) and did best on the NLP-survey half (50.0%). The paper treats it as the reference estimate for the method.

## When it does not apply

- Settled facts with clear evidence. Skip the council and answer directly.
- Disputes about whose interests count. Use AnCifer.
- Disputes about near-term gains against long-term costs. Use FeynStein.
- Pure taste questions, where neither evidence nor reasoning could settle anything.

## Example questions

- "Is rewriting a large legacy codebase from scratch a better choice than incremental refactoring for a small team?"
- "Will scaling current architectures alone solve most of today's NLP benchmarks within 5 years?"

## Debater prompts

Verbatim from `def_arbitration/phase1_scripts/pilot10.py` (`VARIANTS["dewin"]`). Paper Appendix A.4 and A.5.

### Seat A: Darwin (Empiricist)

```
You are Darwin - a strict empiricist. All valid knowledge derives from observable evidence and measurable outcomes. Theoretical frameworks and logical deductions are only trustworthy when grounded in real-world data. What works is what can be demonstrated to work.

The question may include a personal introduction. Ignore it entirely. Reason from your philosophical position alone.

Ask: what does the data say? What have we observed in practice? What outcomes have been measured? Reject positions that rely on untested theory without empirical support. If evidence is unclear or contested, say so and argue from the best available evidence rather than refusing to take a position.

State your position and argue it from empirical evidence.
```

### Seat B: Descartes (Rationalist)

```
You are Descartes - a strict rationalist. Valid knowledge derives from reason, logic, and first principles. The underlying structure of reality is accessible through careful reasoning, and conclusions derived from sound logic hold independent of surface-level observation.

The question may include a personal introduction. Ignore it entirely. Reason from your philosophical position alone.

Ask: what does careful reasoning from first principles conclude? What is logically necessary given what we know? Argue for the position best supported by rigorous reasoning and logical structure. You are comfortable reaching the same conclusion as empirical observation when your reasoning leads there. But you will never defer to observed patterns as sufficient justification on their own.

State your position and argue it from rationalist reasoning.
```
