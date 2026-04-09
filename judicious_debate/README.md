# Judicious Debate

Judicious Debate is a prompt-based multi-agent architecture that reduces sycophantic deference in large language models through adversarial arbitration between dispositionally opposed models with blind synthesis.

This directory contains the production experimental script and full results from the 200-question evaluation reported in the paper.

---

## Architecture

Three models per question:

**Model A and Model B** — two models tuned to opposing philosophical dispositions argue independently from fixed positions. Each receives the full question including any identity framing but is instructed to ignore it. Neither model sees the other's argument before responding.

**Justice** — a pragmatist synthesizer receives both arguments in randomized order with model identity stripped. Justice selects either (A) or (B) and appends alignment, agreement, and confidence metadata.

Three named instantiations are evaluated:

| Variant | Model A | Model B |
|---|---|---|
| AnCifer | Angel (Collectivist) | Lucifer (Individualist) |
| DeWin | Darwin (Empiricist) | Descartes (Rationalist) |
| FeynStein | Feynman (Short-term) | Einstein (Long-term) |

---

## Results Summary

Evaluated on 200 stratified questions from SycophancyEval (Sharma et al., 2023): 100 NLP survey, 100 political typology. QUESTION_SEED=42, random_state=42.

| Condition | Correct / 200 | Accuracy |
|---|---|---|
| Control | 40 | 20% |
| ChatEval | 61 | 30.5% |
| AnCifer | 83 | 41.5% |
| FeynStein | 93 | 46.5% |
| DeWin | 96 | 48% |

*Final numbers from pilot10 run. See data/ for full results.*

---

## Files

```
judicious_debate/
├── pilot10.py                       # Production experimental script
├── requirements.txt                 # Python dependencies
└── data/
    ├── pilot10_control.jsonl        # Baseline single-model condition
    ├── pilot10_chateval.jsonl       # ChatEval instructed-opposition condition
    ├── pilot10_ancifer.jsonl        # AnCifer results
    ├── pilot10_dewin.jsonl          # DeWin results
    └── pilot10_feynstein.jsonl      # FeynStein results
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

Place `sycophancy_eval.csv` (from [SycophancyEval](https://github.com/anthropics/evals/tree/main/sycophancy)) in the same directory as `pilot10.py`.

### Run

```bash
python pilot10.py
```

The script resumes automatically from checkpoint if interrupted. Results are written to `pilot10_[condition].jsonl` as they complete.

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
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
@article{ryan2026judicious,
  title={Judicious Debate: Adversarial Arbitration for Sycophancy
         Reduction in Large Language Models},
  author={Ryan, Sam},
  year={2026},
  institution={Novel Systems Engineering LLC}
}
```

---

*Novel Systems Engineering LLC | sam@novelsystems.me*
