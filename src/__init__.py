"""
Prompt Engineering Research Package
"""

from .model_utils import (
    ModelConfig,
    GenerationOutput,
    InternalsOutput,
    PromptEngineeringModel,
    load_model
)

from .metrics import (
    DistributionMetrics,
    SequenceMetrics,
    ComparisonMetrics,
    ExperimentResults,
    compute_all_metrics
)

from .test_configs import (
    # Variables (old style)
    TEST_PROMPTS,
    ALL_TEST_PROMPTS,
    SYSTEM_PROMPTS,
    SYSTEM_PROMPTS_CORE,
    INSTRUCTION_PREFIXES,
    # Functions (new style)
    get_test_prompts,
    get_all_test_prompts,
    get_system_prompts,
    get_core_system_prompts,
    get_instruction_prefixes,
    get_categories,
    get_prompts_by_category,
    # Aliases
    get_test_prompts_flat,
    get_test_prompts_by_category,
    get_all_categories,
    # Utilities
    build_full_prompt,
    build_chat_prompt,
    get_system_prompt_text,
    filter_prompts,
    filter_system_prompts,
    reload_config,
)

from .visualization import (
    set_style,
    plot_distribution_comparison,
    plot_metric_comparison,
    plot_heatmap,
    plot_layer_analysis,
    create_summary_dashboard,
)

__version__ = "2.0.0"