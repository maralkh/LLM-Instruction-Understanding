# Understanding Prompt Engineering: An Experimental Research Framework

A systematic investigation into **why prompt engineering works** by analyzing model internals and output distributions.

## Research Questions

1. **What changes in the output distribution** when we modify prompts?
2. **Which prompt components** carry the most signal?
3. **Are there identifiable patterns** in what makes prompts effective?
4. **How do base vs instruction-tuned models** respond differently?
5. **Can we predict prompt effectiveness** without full evaluation?

## Project Structure

```
prompt_engineering_research/
├── src/                          # Core utilities
│   ├── __init__.py
│   ├── model_utils.py           # Model loading & probability extraction
│   ├── prompt_utils.py          # Prompt generation & manipulation
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting utilities
├── notebooks/                    # Experiments
│   ├── 01_distribution_shift_analysis.ipynb
│   ├── 02_ablation_studies.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_fewshot_analysis.ipynb
│   └── 05_minimal_effective_prompt.ipynb
├── data/                         # Datasets (if any)
├── results/                      # Experiment outputs
├── configs/                      # Experiment configurations
├── requirements.txt
└── README.md
```

## Experiments

### Experiment 1: Distribution Shift Analysis
**Goal:** Quantify how prompt variations change output probability distributions.
- Measures KL divergence, Jensen-Shannon divergence, entropy changes
- Tests across multiple prompt dimensions (specificity, format, persona)
- Analyzes consistency across question types

### Experiment 2: Ablation Studies
**Goal:** Identify which prompt components carry the most causal weight.
- Systematic removal of prompt components
- Word-level ablation analysis
- Surface-level modifications (punctuation, case, order)

### Experiment 3: Base vs Instruction-Tuned Comparison
**Goal:** Understand how instruction tuning changes prompt sensitivity.
- Compares base model vs chat-tuned model
- Tests assistant prefix effects
- Analyzes chat template impact

### Experiment 4: Few-Shot Analysis
**Goal:** Understand how few-shot examples affect model behavior.
- N-shot scaling analysis
- Example order effects
- Format vs content disentanglement

### Experiment 5: Minimal Effective Prompt
**Goal:** Find the smallest prompt modification that produces performance gains.
- Single word/phrase addition testing
- Combination search
- Feature correlation analysis

### Experiment 6: Paraphrase Sensitivity
**Goal:** Test whether semantically equivalent prompts produce equivalent outputs.
- Paraphrase consistency measurement
- Surface form vs meaning analysis
- Identifies which paraphrases break prompt effectiveness

### Experiment 7: Training Distribution Hypothesis
**Goal:** Understand if prompt effectiveness relates to training data patterns.
- Tests prompts mimicking web formats (StackOverflow, Wikipedia, etc.)
- Measures prompt "naturalness" via perplexity
- Source-specific trigger analysis

### Experiment 8: Attention Pattern Analysis
**Goal:** Understand where the model "looks" under different prompts.
- Extracts and visualizes attention patterns
- Layer-wise entropy analysis
- Few-shot attention pattern comparison
- Head specialization analysis

### Experiment 9: Prompt-Task Interaction Matrix
**Goal:** Understand whether prompt strategies are task-specific or general.
- Builds strategy × task performance matrix
- Identifies universal vs specialized strategies
- Analyzes interaction effects

### Experiment 10: Predictive Framework
**Goal:** Build a model to predict prompt effectiveness without full evaluation.
- Feature engineering for prompts
- Trains Ridge, Lasso, and Random Forest models
- Identifies most predictive features
- Creates simple scoring heuristic

## Setup

```bash
# Clone and setup
cd prompt_engineering_research
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/
```

## Models

Default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (resource-efficient, single GPU)

To use a different model:
```python
from src.model_utils import load_model

model = load_model("your-model-name", device="cuda")
```

## Key Utilities

### Model Utilities
```python
from src.model_utils import load_model

model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Get next token distribution
dist = model.get_next_token_distribution("What is 2+2?")
print(dist["top_tokens"][:5])  # Top 5 predictions
print(dist["entropy"])          # Output entropy

# Get completion probability
probs = model.get_sequence_log_probs("Question: 2+2?\nAnswer:", " 4")
print(probs["total_log_prob"])

# Generate with probability tracking
output = model.generate_with_probs("Complete: The sky is", max_new_tokens=10)
print(output.text, output.entropy)
```

### Prompt Utilities
```python
from src.prompt_utils import PromptVariantGenerator

# Generate systematic prompt variants
variants = PromptVariantGenerator.create_variants(
    question="What is the capital of France?",
    dimensions=['specificity', 'format', 'persona']
)

for v in variants[:3]:
    print(v['config'], v['prompt'][:100])
```

### Metrics
```python
from src.metrics import DistributionMetrics, compute_all_metrics

# Compare distributions
metrics = compute_all_metrics(
    baseline_probs, variant_probs,
    baseline_top_k, variant_top_k
)
print(metrics["kl_divergence"], metrics["jensen_shannon"])
```

## Extending the Framework

### Adding New Experiments

1. Create a new notebook in `notebooks/`
2. Import utilities: `from src import load_model, PromptVariantGenerator, ...`
3. Define your experimental setup
4. Save results to `results/`

### Adding Custom Prompt Dimensions

```python
# In your notebook or add to prompt_utils.py
CUSTOM_DIMENSION = {
    "option_a": "First option text",
    "option_b": "Second option text",
}

variants = PromptVariantGenerator.create_variants(
    question="Your question",
    dimensions=['specificity'],
    custom_dimensions={'custom': CUSTOM_DIMENSION}
)
```

### Supporting New Models

The framework works with any HuggingFace causal LM:

```python
model = load_model("mistralai/Mistral-7B-v0.1")
model = load_model("meta-llama/Llama-2-7b-hf")
```

For larger models, enable 8-bit quantization:
```python
from src.model_utils import ModelConfig, PromptEngineeringModel

config = ModelConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    load_in_8bit=True
)
model = PromptEngineeringModel(config)
```

## Expected Insights

After running all experiments, you should be able to answer:

1. **Which prompt elements matter most?**
   - Is it semantic content or structural cues?
   - How much does formatting contribute vs. instructions?

2. **How do effects vary by model type?**
   - Base vs instruction-tuned sensitivity
   - Role of chat templates

3. **Are there universal strategies?**
   - Do certain additions always help?
   - Task-specific vs. general techniques

4. **What predicts effectiveness?**
   - Simple features that correlate with success
   - Minimal modifications for maximum impact

## Citation

If you use this framework in your research:

```bibtex
@software{prompt_engineering_research,
  title={Understanding Prompt Engineering: An Experimental Research Framework},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

MIT License
