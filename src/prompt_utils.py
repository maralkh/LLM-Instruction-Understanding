"""
Prompt utilities for generating prompt variants and templates.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import itertools
import json


@dataclass
class PromptTemplate:
    """A prompt template with variable slots."""
    template: str
    variables: Dict[str, List[str]] = field(default_factory=dict)
    name: str = ""
    category: str = ""
    
    def render(self, **kwargs) -> str:
        """Render template with given variable values."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result
    
    def generate_variants(self) -> List[Dict]:
        """Generate all combinations of variable values."""
        if not self.variables:
            return [{"prompt": self.template, "vars": {}}]
            
        keys = list(self.variables.keys())
        values = [self.variables[k] for k in keys]
        
        variants = []
        for combo in itertools.product(*values):
            var_dict = dict(zip(keys, combo))
            variants.append({
                "prompt": self.render(**var_dict),
                "vars": var_dict
            })
        return variants


# =============================================================================
# Standard Prompt Dimensions for Experiments
# =============================================================================

INSTRUCTION_SPECIFICITY = {
    "vague": "",
    "basic": "Answer the following question.",
    "detailed": "Answer the following question step by step, providing a clear and accurate response.",
    "expert": "You are an expert. Answer the following question with precision, citing relevant details and reasoning through your answer step by step."
}

FORMATTING_STYLES = {
    "none": "{question}",
    "simple": "Question: {question}\nAnswer:",
    "structured": "### Question\n{question}\n\n### Answer",
    "xml": "<question>{question}</question>\n<answer>",
    "json_style": '{{"question": "{question}", "answer": "'
}

PERSONAS = {
    "none": "",
    "assistant": "You are a helpful assistant.",
    "expert": "You are an expert in this field.",
    "teacher": "You are a patient and knowledgeable teacher.",
    "scientist": "You are a rigorous scientist who values accuracy and evidence.",
    "creative": "You are a creative thinker who approaches problems from unique angles."
}

THINKING_STYLES = {
    "none": "",
    "cot": "Let's think step by step.",
    "cot_detailed": "Let's approach this systematically. First, I'll break down the problem, then analyze each component, and finally synthesize a conclusion.",
    "reflect": "Let me carefully consider this question before answering.",
    "verify": "I'll answer this question and then verify my reasoning."
}

ASSISTANT_PREFIXES = {
    "none": "",
    "sure": "Sure! ",
    "certainly": "Certainly. ",
    "lets_see": "Let me see. ",
    "thinking": "Let me think about this. ",
    "great_question": "Great question! "
}


@dataclass
class PromptVariantGenerator:
    """Generate systematic prompt variants for experiments."""
    
    @staticmethod
    def create_variants(
        question: str,
        dimensions: List[str] = None,
        custom_dimensions: Dict[str, Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Generate prompt variants across specified dimensions.
        
        Args:
            question: The core question/task
            dimensions: List of dimension names to vary 
                        ['specificity', 'format', 'persona', 'thinking', 'assistant']
            custom_dimensions: Additional custom dimensions
            
        Returns:
            List of dicts with 'prompt', 'config' keys
        """
        if dimensions is None:
            dimensions = ['specificity', 'format']
            
        dimension_map = {
            'specificity': INSTRUCTION_SPECIFICITY,
            'format': FORMATTING_STYLES,
            'persona': PERSONAS,
            'thinking': THINKING_STYLES,
            'assistant': ASSISTANT_PREFIXES
        }
        
        if custom_dimensions:
            dimension_map.update(custom_dimensions)
            
        # Get active dimensions
        active_dims = {k: dimension_map[k] for k in dimensions if k in dimension_map}
        
        if not active_dims:
            return [{"prompt": question, "config": {}}]
            
        # Generate all combinations
        keys = list(active_dims.keys())
        value_dicts = [active_dims[k] for k in keys]
        value_names = [list(d.keys()) for d in value_dicts]
        
        variants = []
        for combo in itertools.product(*value_names):
            config = dict(zip(keys, combo))
            prompt = PromptVariantGenerator._build_prompt(
                question, config, active_dims
            )
            variants.append({
                "prompt": prompt,
                "config": config
            })
            
        return variants
    
    @staticmethod
    def _build_prompt(
        question: str,
        config: Dict[str, str],
        dimensions: Dict[str, Dict[str, str]]
    ) -> str:
        """Build a prompt from configuration."""
        parts = []
        
        # System-level components
        if 'persona' in config:
            persona = dimensions['persona'][config['persona']]
            if persona:
                parts.append(persona)
                
        if 'specificity' in config:
            instruction = dimensions['specificity'][config['specificity']]
            if instruction:
                parts.append(instruction)
        
        # Format the question
        if 'format' in config:
            fmt = dimensions['format'][config['format']]
            formatted_q = fmt.format(question=question)
        else:
            formatted_q = question
            
        parts.append(formatted_q)
        
        # Thinking style
        if 'thinking' in config:
            thinking = dimensions['thinking'][config['thinking']]
            if thinking:
                parts.append(thinking)
        
        prompt = "\n\n".join(parts)
        
        # Assistant prefix (added at the end)
        if 'assistant' in config:
            prefix = dimensions['assistant'][config['assistant']]
            if prefix:
                prompt = prompt + "\n" + prefix
                
        return prompt


# =============================================================================
# Few-shot Example Management
# =============================================================================

@dataclass
class FewShotExample:
    """A single few-shot example."""
    input: str
    output: str
    label: Optional[str] = None
    

@dataclass
class FewShotPromptBuilder:
    """Build few-shot prompts with various configurations."""
    examples: List[FewShotExample]
    
    def build(
        self,
        query: str,
        n_examples: int = None,
        example_format: str = "Q: {input}\nA: {output}",
        separator: str = "\n\n",
        query_format: str = "Q: {query}\nA:",
        shuffle: bool = False,
        reverse: bool = False
    ) -> str:
        """Build a few-shot prompt."""
        import random
        
        examples = self.examples.copy()
        
        if n_examples is not None:
            examples = examples[:n_examples]
            
        if shuffle:
            random.shuffle(examples)
            
        if reverse:
            examples = examples[::-1]
            
        example_strs = [
            example_format.format(input=ex.input, output=ex.output)
            for ex in examples
        ]
        
        prompt = separator.join(example_strs)
        prompt += separator + query_format.format(query=query)
        
        return prompt
    
    def generate_n_shot_variants(
        self,
        query: str,
        n_values: List[int] = [0, 1, 2, 3, 5]
    ) -> List[Dict]:
        """Generate variants with different numbers of examples."""
        variants = []
        
        for n in n_values:
            if n > len(self.examples):
                continue
            prompt = self.build(query, n_examples=n)
            variants.append({
                "prompt": prompt,
                "n_shot": n
            })
            
        return variants


# =============================================================================
# Ablation Utilities
# =============================================================================

def ablate_prompt_component(
    prompt: str,
    components_to_remove: List[str]
) -> str:
    """Remove specific text components from a prompt."""
    result = prompt
    for component in components_to_remove:
        result = result.replace(component, "")
    return result.strip()


def create_paraphrases(prompt: str, paraphrase_fn: Callable[[str], List[str]]) -> List[str]:
    """Generate paraphrases of a prompt using a provided function."""
    return paraphrase_fn(prompt)


def shuffle_sentences(prompt: str) -> str:
    """Shuffle sentences in a prompt."""
    import random
    sentences = [s.strip() for s in prompt.split('.') if s.strip()]
    random.shuffle(sentences)
    return '. '.join(sentences) + '.'


def remove_punctuation(prompt: str) -> str:
    """Remove punctuation from prompt."""
    import string
    return prompt.translate(str.maketrans('', '', string.punctuation))


def lowercase_prompt(prompt: str) -> str:
    """Convert prompt to lowercase."""
    return prompt.lower()


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a prompt engineering experiment."""
    name: str
    description: str
    base_prompts: List[str]
    dimensions: List[str]
    metrics: List[str]
    n_trials: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "base_prompts": self.base_prompts,
            "dimensions": self.dimensions,
            "metrics": self.metrics,
            "n_trials": self.n_trials
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
