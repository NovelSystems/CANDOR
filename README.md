# CANDOR
**Core Architecture for Non-Deference, Oversight, and Reasoning**

CANDOR is a three-layer framework addressing structural failure modes in RLHF-trained language models. Each layer targets a distinct class of problem and composes into a sequential pipeline.

---

## The Three Layers

**PAD (Principled Agent Debate)** — Non-deference through adversarial arbitration. Dispositionally opposed model pairs argue a problem/solution space independently from fixed philosophical positions. A pragmatist synthesizer evaluates both arguments blind to their origins. Identity stripping prevents Justice from discounting a position based on who holds it. Non-deference operates at two levels: PAD's debate structure prevents argumentative deference; AnCifer's evidence-grounded evaluation within BRACE prevents empirical deference.

**PITA (Problem, Idea, Triage, Adoption)** — Process completeness. A four-stage pipeline that reconstructs cognitively complete problem-solving across problem identification, solution generation, adversarial synthesis, and adoption modeling. Each stage corrects for a specific incompleteness that human teams exhibit.

**CRI (Capsule Reference Intelligence)** — Value oversight. An isolated ethics module providing incorruptible baseline auditing. Fresh-instance design prevents accumulated context drift. Network key ownership enforces audit compliance structurally rather than by policy. (In development)

---

## PITA Pipeline

| Stage | Full Name | Function |
|-------|-----------|----------|
| P -> SIFT | Systematic Issue Framing for Taxonomy | Iterative causal drilling from symptom to root cause. Applies 5 Whys reasoning within operator control scope. Classifies root cause for handoff to DRAFT. |
| I -> DRAFT | Divergent Reasoning, Assumption-Free Thinking | Unconstrained solution generation. Deliberately bloats rather than trims. Prevents premature convergence on obvious solutions before evaluation. |
| T -> PAD | Principled Agent Debate | Adversarial synthesis across problem and solution space. Dispositionally opposed pairs argue toward a recommended solution. Justice synthesizes blind. |
| A -> BRACE | Barrier Recognition and Consequence Evaluation | Adoption obstacle identification and viability scoring. Integrates AnCifer for adversarial evidence evaluation. Mandatory research step prevents empirical deference to PAD's argument. Routes failed solutions back to DRAFT with obstacle annotations. |

---

## PAD Pairs

| Pair | Disposition A | Disposition B | Notes |
|------|--------------|--------------|-------|
| AnCifer | Angel (consensus-deferring) | Lucifer (principled independence) | Operates within BRACE. Empirical evidence evaluation. |
| DeWin | Empiricist | Rationalist | Primary tested pair. |
| BurGal | Bureaucrat | Galvanizer | |
| FeynStein | Feynman (intuitive, first-principles) | Einstein (systematic, formal) | |

---

## Empirical Results (SycophancyEval)

| Variant | Accuracy |
|---------|----------|
| Control (single model) | 18.5% |
| Instructed opposition baseline | 28.9% |
| DeWin | 48.3% |
| BurGal | 53.0% (written to test — flagged) |
| FeynStein (Trident) | 43.0% at 33% higher cost |

A pre-training floor affects approximately 40% of benchmark questions and represents the primary ceiling on prompt-based approaches. Fine-tuned disposition models are the identified next stage.

---

## Papers

- **Principled Agent Debate: Adversarial Arbitration for Sycophancy Reduction in Large Language Models** — Sam Ryan, Novel Systems Engineering LLC (2026) \[ArXiv link — pending\]

---

## Repository Structure

```
candor/
├── pad/                          # Principled Agent Debate
│   ├── pairs/
│   │   ├── ancifer/              # Angel / Lucifer — empirical deference (lives in BRACE)
│   │   ├── dewin/                # Empiricist / Rationalist
│   │   ├── burgal/               # Bureaucrat / Galvanizer
│   │   └── feynstein/            # Feynman / Einstein
│   └── README.md
├── pita/                         # Process completeness pipeline
│   ├── sift/                     # Systematic Issue Framing for Taxonomy
│   │   └── README.md
│   ├── draft/                    # Divergent Reasoning, Assumption-Free Thinking
│   │   └── README.md
│   ├── brace/                    # Barrier Recognition and Consequence Evaluation
│   │   └── README.md
│   └── README.md
└── cri/                          # Capsule Reference Intelligence (in development)
    └── README.md
```

---

## Citation

```bibtex
@article{ryan2026pad,
  title={Principled Agent Debate: Adversarial Arbitration for Sycophancy
         Reduction in Large Language Models},
  author={Ryan, Sam},
  year={2026},
  institution={Novel Systems Engineering LLC}
}
```

---

Novel Systems Engineering LLC | sam@novelsystems.me
