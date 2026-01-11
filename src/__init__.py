"""
Prompt Engineering Research Package

Tools for understanding why prompt engineering works by analyzing
model internals and output distributions.
"""

from .model_utils import (
    ModelConfig,
    GenerationOutput,
    PromptEngineeringModel,
    load_model
)

from .prompt_utils import (
    PromptTemplate,
    PromptVariantGenerator,
    FewShotExample,
    FewShotPromptBuilder,
    ExperimentConfig,
    INSTRUCTION_SPECIFICITY,
    FORMATTING_STYLES,
    PERSONAS,
    THINKING_STYLES,
    ASSISTANT_PREFIXES
)

from .metrics import (
    DistributionMetrics,
    SequenceMetrics,
    ComparisonMetrics,
    ExperimentResults,
    compute_all_metrics
)

from .visualization import (
    set_style,
    plot_distribution_comparison,
    plot_entropy_comparison,
    plot_log_prob_comparison,
    plot_dimension_heatmap,
    plot_ablation_results,
    plot_entropy_trajectory,
    plot_model_comparison,
    create_summary_dashboard
)

__version__ = "0.1.0"
