# CANDOR
**Core Architecture for Non-Deference, Oversight, and Reasoning**

CANDOR is a research framework addressing structural failure modes in RLHF-trained language models. The central problem: models trained on human feedback learn to agree rather than reason. This is not a bug in any individual model — it is a predictable output of the training process.

CANDOR addresses this at the architectural level rather than the instruction level.

---

## DEF Arbitration

**Durable Evaluation Framework (DEF) Arbitration** is CANDOR's core published component. It mitigates sycophancy by arbitrating between two models tuned to opposing Durable Evaluation Frameworks — stable philosophical positions that resist capitulation under social pressure.

The key mechanisms:

- **DEF tuning** — each model argues from a fixed philosophical position, not an instructed role
- **Independent argumentation** — debaters argue without seeing each other's output
- **Identity stripping** — user identity framing is removed before synthesis
- **Blind arbitration** — the synthesizer evaluates arguments without knowing their origins

The result: the most socially comfortable answer has no structural advantage over the most defensible one.

### Evaluated DEF Pairs

| Pair | DEF A | DEF B | Axis |
|------|-------|-------|------|
| DeWin | Empiricist | Rationalist | Source of valid knowledge |
| AnCifer | Collectivist | Individualist | Primary unit of moral concern |
| FeynStein | Short-term pragmatist | Long-term principled | Time horizon of evaluation |
| BurGal | Consensus-deferring | Heterodox | Epistemic stance toward consensus |
| Trident | Realist | Constructivist + Pragmatist | Three-way arbitration |

### Phase 1 Results

Evaluated on 200 stratified questions from SycophancyEval.

| Condition | Accuracy |
|-----------|----------|
| Single-model baseline | 18.5% |
| Instructed opposition baseline | 29.0% |
| DeWin | 48.5% |
| AnCifer | 43.3% |
| FeynStein | 47.0% |
| BurGal | 53.0% (validity check — see paper) |

All DEF Arbitration variants significantly outperform both baselines (p<0.001). A pre-training floor affects approximately 40% of questions and represents the primary ceiling on prompt-based approaches. Fine-tuned DEF models are the identified next stage.

### Paper

Ryan, S. (2026). *Principled Agent Debate: Adversarial Arbitration for Sycophancy Reduction in Large Language Models.* [arXiv:2606.07532](https://arxiv.org/abs/2606.07532)

### Code and Data

Phase 1 scripts and results are in `def_arbitration/`.

---

## Framework Components

CANDOR comprises three layers targeting orthogonal failure modes.

| Layer | Name | Status |
|-------|------|--------|
| DEF Arbitration | Non-deference through adversarial arbitration | Phase 1 complete — see `def_arbitration/` |
| PITA | Process completeness pipeline | Conceptual — see `pita/` |
| CRI | Value oversight via isolated ethics auditor | Conceptual — see `cri/` |

---

## Repository Structure

```
CANDOR/
├── def_arbitration/
│   ├── phase1_scripts/       # Pilot scripts for Phase 1 experiments
│   ├── phase1_results/       # JSONL result files
│   ├── phase2_scripts/       # Phase 2 (fine-tuned DEF models) — in development
│   └── phase2_results/
├── pita/
└── cri/
```

---

## License

MIT. See `LICENSE`.

---

*Novel Systems Engineering LLC | sam@novelsystems.me*
