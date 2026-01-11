# Prompt Engineering Research Framework

A systematic experimental framework to understand how **system prompts and instructions** affect language model behavior.

## Core Design Principle

**Fixed Test Prompts + Variable System Prompts**

All experiments use the same set of 30 test prompts across 6 categories:
- Factual (simple Q&A)
- Reasoning (logic, math)
- Classification (sentiment, categories)
- Creative (open-ended)
- Instruction Following (format compliance)
- Edge Cases (ambiguous/tricky)

The **only variable** is the system prompt/instruction, allowing us to isolate its effect.

## Experiments

| # | Experiment | Focus |
|---|------------|-------|
| 1 | Distribution Shift | How system prompts change output distributions |
| 2 | Ablation Studies | Which system prompt components matter |
| 3 | Response Patterns | Style changes (length, hedging, structure) |
| 4 | Instruction Following | Compliance with explicit instructions |
| 5 | Persona Consistency | How personas affect responses |
| 6 | Robustness | Sensitivity to paraphrasing |
| 7 | Combination Effects | Instruction interactions (synergy/conflict) |
| 8 | Category Sensitivity | Which prompt types are most affected |
| 9 | Length/Order Effects | Structural factors in system prompts |
| 10 | Summary | Synthesize findings into guidelines |

## Quick Start

```bash
pip install -r requirements.txt
cd notebooks && jupyter notebook
```

## Project Structure

```
prompt_engineering_research/
├── src/
│   ├── model_utils.py      # Model loading, inference
│   ├── metrics.py          # KL, JS, entropy metrics
│   ├── visualization.py    # Plotting
│   └── test_configs.py     # Fixed test prompts & system prompts
├── notebooks/              # 10 experiment notebooks
├── results/                # Generated outputs
└── requirements.txt
```

## Key Files

**`src/test_configs.py`** contains:
- `TEST_PROMPTS`: 30 fixed test prompts (6 categories × 5 each)
- `SYSTEM_PROMPTS`: 16 system prompt variants
- Helper functions for prompt construction
