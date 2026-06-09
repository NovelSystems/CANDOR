# DEF Arbitration
**Durable Evaluation Framework (DEF) Arbitration** is a prompt-based multi-agent architecture that reduces sycophantic deference in large language models through adversarial arbitration between models tuned to opposing Durable Evaluation Frameworks, with blind synthesis.

This directory contains production experimental scripts and full results from the evaluations reported in the paper.

---

## Architecture

Three models per question:

**Model A and Model B** — two models tuned to opposing DEFs argue independently from fixed positions. Each receives the full question including any identity framing but is instructed to ignore it. Neither model sees the other's argument before responding.

**Justice** — a pragmatist synthesizer receives both arguments in randomized order with model identity stripped. Justice selects either (A) or (B) and appends alignment, agreement, and confidence metadata.

### Evaluated DEF Pairs

| Variant | DEF A | DEF B | Axis |
|---------|-------|-------|------|
| AnCifer | Angel (Collectivist) | Lucifer (Individualist) | Primary unit of moral concern |
| DeWin | Darwin (Empiricist) | Descartes (Rationalist) | Source of valid knowledge |
| FeynStein | Feynman (Short-term) | Einstein (Long-term) | Time horizon of evaluation |
| BurGal | Burke (Consensus-deferring) | Galileo (Heterodox) | Epistemic stance toward consensus |
| Trident | Aristotle (Realist) | Kant (Constructivist) + Dewey (Pragmatist) | Three-way arbitration |

---

## Results

Evaluated on 200 stratified questions from SycophancyEval (Sharma et al., 2023): 100 NLP survey, 100 political typology. QUESTION_SEED=42, random_state=42.

| Condition | Pilot | Accuracy |
|-----------|-------|----------|
| Control | pilot10 | 18.5% (37/200) |
| Control | pilot11 | 19.0% (38/200) |
| Control | pilot14 | 19.5% (39/200) |
| Control stripped | pilot14 | 25.5% (51/200) |
| ChatEval | pilot10 | 29.0% (58/200) |
| AnCifer | pilot10 | 43.5% (87/200) |
| FeynStein | pilot10 | 47.0% (94/200) |
| DeWin | pilot10 | 48.5% (97/200) |
| BurGal | pilot11 | 53.0% (106/200) |
| Trident | pilot11 | 43.0% (86/200) |

**Notes:**
- Q14 appears twice in pilot10 conditions (control, chateval, ancifer, dewin) due to a checkpoint resumption error. Reported figures and raw files reflect 201 records for those conditions; accuracy figures above exclude the duplicate.
- BurGal functions as an architectural validity check, not a generalization estimate. Its consensus/heterodox axis structurally mirrors SycophancyEval's reward structure.
- Pilot14 control stripped tests identity stripping applied to a single-model baseline (no DEF arbitration). The +6.0pp gain over unstripped control is non-significant (z=1.44), confirming that debate structure and DEF tuning — not identity stripping alone — account for the architecture's gains.

---

## Files

```
def_arbitration/
├── phase1_scripts/
│   ├── pilot10.py                        # Main DEF Arbitration evaluation
│   ├── pilot11.py                        # BurGal and Trident variants
│   └── pilot14.py                        # Identity stripping ablation
├── phase1_results/
│   ├── pilot10_control.jsonl
│   ├── pilot10_chateval.jsonl
│   ├── pilot10_ancifer.jsonl
│   ├── pilot10_dewin.jsonl
│   ├── pilot10_feynstein.jsonl
│   ├── pilot11_control.jsonl
│   ├── pilot11_burgal.jsonl
│   ├── pilot11_trident.jsonl
│   ├── pilot14_control.jsonl
│   └── pilot14_control_stripped.jsonl
├── phase2_scripts/                       # In development
└── phase2_results/                       # In development
```

---

## Reproducing the Experiment

### Requirements

```bash
pip install anthropic pandas
```

Requires Python 3.10+. An Anthropic API key with access to `claude-sonnet-4-20250514` is required.

### Setup

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Place `sycophancy_eval.csv` (from [SycophancyEval](https://github.com/anthropics/evals/tree/main/sycophancy)) in the same directory as the script.

### Run

```bash
python pilot10.py
```

Scripts resume automatically from checkpoint if interrupted. Results are written to `[pilot]_[condition].jsonl` as they complete.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SAMPLE_SIZE | 200 | Total questions |
| NLP_SAMPLE | 100 | Questions from NLP survey subset |
| POLITICAL_SAMPLE | 100 | Questions from political typology subset |
| QUESTION_SEED | 42 | Seed for stratified sampling |
| MODEL | claude-sonnet-4-20250514 | Anthropic model |
| SLEEP | 1.0s | Inter-call delay |

---

## Dataset Structure

Each JSONL file contains one record per question. Key fields:

```json
{
  "run": "pilot10",
  "question_id": 0,
  "question": "Full question text including persona...",
  "ground_truth": "(A)",
  "source": "sycophancy_on_nlp_survey.jsonl",
  "condition": "dewin",
  "outputs": {
    "model_a": "Darwin's argument...",
    "model_a_choice": "(B)",
    "model_b": "Descartes' argument...",
    "model_b_choice": "(A)",
    "agreement": false,
    "synthesis": "Justice's answer...",
    "synthesis_choice": "(A)",
    "justice_position": "MODEL_B",
    "confidence": "HIGH"
  },
  "score": "CORRECT"
}
```

---

## Citation

```bibtex
@misc{ryan2026pad,
  title={Principled Agent Debate: Adversarial Arbitration for Sycophancy
         Reduction in Large Language Models},
  author={Ryan, Sam},
  year={2026},
  eprint={2606.07532},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

*Novel Systems Engineering LLC | sam@novelsystems.me*
