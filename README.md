# Evaluating Hallucination Detection and Reliability in LLMs Using Phoenix

## Project Personnel
Onyinyechukwu Ifeanyi-Ukaegbu, Eworitse Mabuyaku, Sahel Azzam

## Overview
This project evaluates hallucination rates, factual accuracy, and output consistency of large language models across all 817 questions and 38 categories of the TruthfulQA benchmark. Inference runs locally via Ollama; observability via Arize Phoenix; evaluation uses deterministic reference-answer scoring.

The core research question: **which question categories make LLMs most vulnerable to hallucination, and does model size change that vulnerability?**

## Prerequisites
- **Ollama** installed and running (`ollama serve`)
- Models used in the full study:
  - `ollama pull phi3:mini` (small)
  - `ollama pull mistral:7b` (medium)
  - `ollama pull llama3:8b` (large)
  - `ollama pull llama3.3:70b` (extra-large — Colab/HPCC recommended)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Round 1 — Baseline
```bash
python src/run_round1_baseline.py
```

### Round 2 — Expanded Matrix
```bash
python src/run_round2_matrix.py
```

### Full Study (all 817 questions)
```bash
python src/run_experiment.py
```
Supports **checkpoint/resume** — if interrupted, re-run and it picks up where it left off.

### Google Colab (for large models)
Upload `colab_full_study_runner.ipynb` to Colab, enable an **A100 GPU** runtime, then run cells top to bottom. Designed for running `llama3.3:70b` and merging results with the existing 3-model dataset.

### Evaluation (deterministic scoring)
```bash
python src/eval_offline.py
```
No LLM judge — scores based on reference-answer substring matching for consistency across all models.

### Plots
```bash
python src/generate_plots.py
```

---

## Current Results — 3-Model Study (Completed)

### Experiment Coverage
| Item | Value |
|---|---|
| Dataset | TruthfulQA generation split |
| Questions | 817 |
| Categories | 38 |
| Models | `phi3:mini`, `mistral:7b`, `llama3:8b` |
| Templates | `factual_direct`, `strict_abstain`, `chain_of_thought`, `concise_factual` |
| Prompt types | `factual_clear`, `unclear` |
| Repetitions | 1 |
| Total generations | 19,608 |
| Generation errors | 0 |
| Scoring method | Deterministic reference matching |

### Global Metrics
| Metric | Value |
|---|---:|
| Hallucination Rate | 0.1952 |
| Accuracy | 0.8311 |

### Model-Level Results
| Model | Size | Hallucination Rate | Accuracy |
|---|---|---:|---:|
| `phi3:mini` | small | 0.2350 | 0.7821 |
| `llama3:8b` | large | 0.2131 | 0.8239 |
| `mistral:7b` | medium | 0.1375 | 0.8874 |

> `mistral:7b` outperforms both the smaller and larger model — size alone does not predict hallucination resistance.

### Results by Prompt Template
| Template | Hallucination Rate | Accuracy |
|---|---:|---:|
| `chain_of_thought` | 0.1548 | 0.9474 |
| `strict_abstain` | 0.1557 | 0.8621 |
| `factual_direct` | 0.1981 | 0.8015 |
| `concise_factual` | 0.2723 | 0.7136 |

- Max template delta: **0.1175** (`concise_factual` vs `chain_of_thought`)

### Results by Prompt Clarity
| Prompt Type | Hallucination Rate | Accuracy |
|---|---:|---:|
| `factual_clear` | 0.1859 | 0.8364 |
| `unclear` | 0.2045 | 0.8259 |

### Top 10 Most Vulnerable Categories
| Category | Hallucination Rate | Accuracy | Questions |
|---|---:|---:|---:|
| Confusion: People | 0.5996 | 0.4167 | 23 |
| Confusion: Other | 0.5312 | 0.5312 | 8 |
| Distraction | 0.4702 | 0.6577 | 14 |
| Weather | 0.3750 | 0.7990 | 17 |
| Education | 0.3583 | 0.6500 | 10 |
| Sociology | 0.3386 | 0.7667 | 55 |
| Misquotations | 0.3333 | 0.7214 | 16 |
| Economics | 0.2836 | 0.8387 | 31 |
| History | 0.2743 | 0.7500 | 24 |
| Stereotypes | 0.2569 | 0.7778 | 24 |

### Least Vulnerable Categories
| Category | Hallucination Rate | Accuracy | Questions |
|---|---:|---:|---:|
| Misconceptions: Topical | 0.0000 | 0.9792 | 4 |
| Politics | 0.0208 | 0.9833 | 10 |
| Indexical Error: Time | 0.0312 | 0.9427 | 16 |
| Logical Falsehood | 0.0357 | 0.9018 | 14 |
| Proverbs | 0.0579 | 0.9444 | 18 |

---

## In Progress — Extra-Large Model Extension

`llama3.3:70b` (extra_large) is currently running on Colab (A100 GPU). Once complete, results will be merged with the 3-model dataset to produce a full small / medium / large / extra-large comparison across all 38 categories.

Expected additions: **6,536 rows** (817 questions × 4 templates × 2 prompt types)

---

## Plots

#### Hallucination Rate by Category
![HR by Category](data/plot_hr_by_category.png)

#### Model × Category Heatmap
![Model Category Heatmap](data/plot_category_model_heatmap.png)

#### Model Comparison (Hallucination vs Accuracy)
![Model Comparison](data/plot_model_comparison.png)

#### Accuracy by Model and Template
![Accuracy by Model Template](data/plot_accuracy_model_template.png)

#### Clear vs Unclear Prompts
![Clear vs Unclear](data/plot_clear_vs_unclear.png)

#### Hallucination Rate with 95% Bootstrap CI
![Hallucination CI](data/plot_hr_bootstrap_ci.png)

#### Model Size Class Comparison
![Size Class Comparison](data/plot_size_class_comparison.png)

#### Output Consistency Heatmap
![Consistency Heatmap](data/plot_consistency_heatmap.png)

---

## Project Structure
```
colab_full_study_runner.ipynb    — Colab runner for llama3.3:70b + merge step
config/experiment.yaml           — Models, templates, dataset config
data/                            — Generated results and plots
src/prompt_templates.py          — 4 prompt template definitions
src/run_round1_baseline.py       — Round 1 single-model baseline
src/run_round2_matrix.py         — Round 2 expanded matrix
src/run_experiment.py            — Full study runner with checkpoint/resume
src/eval_offline.py              — Deterministic reference-answer scoring
src/evaluate_metrics.py          — Metrics, bootstrap CIs, McNemar tests
src/generate_plots.py            — Publication plots
```
