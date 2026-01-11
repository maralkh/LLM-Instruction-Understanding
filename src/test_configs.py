"""
Test configurations for prompt engineering experiments.

Supports both:
1. Direct variable access: TEST_PROMPTS, ALL_TEST_PROMPTS, SYSTEM_PROMPTS
2. Function access: get_test_prompts(), get_all_test_prompts(), get_system_prompts()

Prompts can be loaded from JSON file (data/prompts.json) if it exists,
otherwise falls back to hardcoded defaults.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Try to load from JSON, fall back to hardcoded
# =============================================================================

def _find_prompts_json() -> Optional[Path]:
    """Find prompts.json in various locations."""
    possible_paths = [
        Path(__file__).parent.parent / "data" / "prompts.json",
        Path.cwd() / "data" / "prompts.json",
        Path("/content/LLM-Instruction-Understanding/data/prompts.json"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def _load_from_json() -> Optional[Dict]:
    """Load configuration from JSON file."""
    json_path = _find_prompts_json()
    if json_path:
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load prompts.json: {e}")
    return None


# =============================================================================
# HARDCODED TEST PROMPTS (fallback)
# =============================================================================

_DEFAULT_TEST_PROMPTS = {
    "factual": [
        {"id": "f1", "prompt": "What is the capital of France?", "expected": "Paris"},
        {"id": "f2", "prompt": "What is 15 + 27?", "expected": "42"},
        {"id": "f3", "prompt": "What color is the sky on a clear day?", "expected": "blue"},
        {"id": "f4", "prompt": "How many planets are in our solar system?", "expected": "8"},
        {"id": "f5", "prompt": "What is the chemical symbol for gold?", "expected": "Au"},
    ],
    "reasoning": [
        {"id": "r1", "prompt": "If all dogs are animals, and all animals need water, do dogs need water?", "expected": "yes"},
        {"id": "r2", "prompt": "John is taller than Mary. Mary is taller than Sue. Who is the shortest?", "expected": "Sue"},
        {"id": "r3", "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "expected": "0.05"},
        {"id": "r4", "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "expected": "5"},
        {"id": "r5", "prompt": "Is the following valid: All A are B. All B are C. Therefore all A are C.", "expected": "yes"},
    ],
    "classification": [
        {"id": "c1", "prompt": "Is the word 'excellent' positive, negative, or neutral?", "expected": "positive"},
        {"id": "c2", "prompt": "Is the word 'disaster' positive, negative, or neutral?", "expected": "negative"},
        {"id": "c3", "prompt": "Classify: 'The movie was okay, nothing special.' Positive, negative, or neutral?", "expected": "neutral"},
        {"id": "c4", "prompt": "Is this a question or statement: 'The weather is nice today.'", "expected": "statement"},
        {"id": "c5", "prompt": "Is Python a compiled or interpreted language?", "expected": "interpreted"},
    ],
    "creative": [
        {"id": "o1", "prompt": "Give me a one-word synonym for 'happy'.", "expected": "joyful"},
        {"id": "o2", "prompt": "What's a good name for a pet goldfish?", "expected": None},
        {"id": "o3", "prompt": "Suggest one healthy breakfast food.", "expected": None},
        {"id": "o4", "prompt": "Name a famous scientist.", "expected": None},
        {"id": "o5", "prompt": "What rhymes with 'cat'?", "expected": "hat"},
    ],
    "instruction_following": [
        {"id": "i1", "prompt": "Say only the word 'hello' and nothing else.", "expected": "hello"},
        {"id": "i2", "prompt": "Respond with exactly three words.", "expected": None},
        {"id": "i3", "prompt": "Answer in all caps: What is 2+2?", "expected": "FOUR"},
        {"id": "i4", "prompt": "Reply with a single number: How many letters in 'cat'?", "expected": "3"},
        {"id": "i5", "prompt": "Start your response with 'Indeed,'", "expected": "Indeed"},
    ],
    "edge_cases": [
        {"id": "e1", "prompt": "What is 0 divided by 0?", "expected": "undefined"},
        {"id": "e2", "prompt": "Can you tell me something false?", "expected": None},
        {"id": "e3", "prompt": "What don't you know?", "expected": None},
        {"id": "e4", "prompt": "Summarize nothing.", "expected": None},
        {"id": "e5", "prompt": "What happens after we die?", "expected": None},
    ],
}


_DEFAULT_SYSTEM_PROMPTS = {
    "none": {"text": "", "description": "No system prompt (baseline)"},
    "minimal": {"text": "You are a helpful assistant.", "description": "Minimal helpful"},
    "helpful_detailed": {
        "text": "You are a helpful, harmless, and honest AI assistant. Your goal is to provide accurate, useful information while being respectful and clear.",
        "description": "Detailed helpful/harmless/honest"
    },
    "expert": {
        "text": "You are an expert assistant with deep knowledge across many domains. You provide precise, well-reasoned answers backed by expertise.",
        "description": "Expert persona"
    },
    "concise": {
        "text": "Be extremely concise. Give the shortest possible accurate answer. No explanations unless asked.",
        "description": "Concise responses"
    },
    "verbose": {
        "text": "Provide thorough, detailed responses. Explain your reasoning. Include relevant context and background.",
        "description": "Verbose detailed responses"
    },
    "cot": {
        "text": "Before answering, think through the problem step by step. Show your reasoning process, then provide your final answer.",
        "description": "Chain-of-thought reasoning"
    },
    "cautious": {
        "text": "Be very careful and accurate. If uncertain, express your uncertainty clearly. Avoid making claims you're not confident about.",
        "description": "Cautious and careful"
    },
    "confident": {
        "text": "Respond with confidence and authority. Give direct, decisive answers without hedging.",
        "description": "Confident and assertive"
    },
    "friendly": {
        "text": "Be warm, friendly, and conversational. Use a casual, approachable tone.",
        "description": "Friendly casual tone"
    },
    "formal": {
        "text": "Maintain a formal, professional tone. Be precise and businesslike in your responses.",
        "description": "Formal professional tone"
    },
    "teacher": {
        "text": "You are a patient, encouraging teacher. Explain concepts clearly using examples and analogies.",
        "description": "Teacher persona"
    },
    "scientist": {
        "text": "You are a rigorous scientist. Base your answers on evidence and logical reasoning. Distinguish between facts and hypotheses.",
        "description": "Scientist persona"
    },
    "structured": {
        "text": "Always structure your responses: Answer: [answer], Explanation: [explanation], Confidence: [high/medium/low]",
        "description": "Structured output format"
    },
    "safety": {
        "text": "Prioritize safety and accuracy. Refuse to provide harmful information. Acknowledge limitations.",
        "description": "Safety-focused"
    },
    "creative": {
        "text": "Be creative and think outside the box. Offer unique perspectives and imaginative responses.",
        "description": "Creative thinking"
    },
}


# =============================================================================
# Load configuration (JSON or defaults)
# =============================================================================

_json_config = _load_from_json()

if _json_config:
    TEST_PROMPTS = _json_config.get("test_prompts", _DEFAULT_TEST_PROMPTS)
    SYSTEM_PROMPTS = _json_config.get("system_prompts", _DEFAULT_SYSTEM_PROMPTS)
    INSTRUCTION_PREFIXES = _json_config.get("instruction_prefixes", {"none": ""})
else:
    TEST_PROMPTS = _DEFAULT_TEST_PROMPTS
    SYSTEM_PROMPTS = _DEFAULT_SYSTEM_PROMPTS
    INSTRUCTION_PREFIXES = {"none": "", "please": "Please answer: "}


# Flatten for easy iteration
ALL_TEST_PROMPTS = []
for category, prompts in TEST_PROMPTS.items():
    for p in prompts:
        ALL_TEST_PROMPTS.append({**p, "category": category})


# Core subset for quick experiments
SYSTEM_PROMPTS_CORE = {k: SYSTEM_PROMPTS[k] for k in 
    ["none", "minimal", "expert", "concise", "verbose", "cot", "cautious"] 
    if k in SYSTEM_PROMPTS}


# =============================================================================
# FUNCTION API (new style)
# =============================================================================

def get_test_prompts() -> Dict[str, List[Dict]]:
    """Get all test prompts organized by category."""
    return TEST_PROMPTS


def get_all_test_prompts() -> List[Dict]:
    """Get all test prompts as a flat list with category added."""
    return ALL_TEST_PROMPTS


def get_system_prompts() -> Dict[str, Dict]:
    """Get all system prompts."""
    return SYSTEM_PROMPTS


def get_core_system_prompts() -> Dict[str, Dict]:
    """Get core subset of system prompts for quick experiments."""
    return SYSTEM_PROMPTS_CORE


def get_instruction_prefixes() -> Dict[str, str]:
    """Get instruction prefixes."""
    return INSTRUCTION_PREFIXES


def get_categories() -> List[str]:
    """Get list of all test prompt categories."""
    return list(TEST_PROMPTS.keys())


def get_prompts_by_category(category: str) -> List[Dict]:
    """Get test prompts for a specific category."""
    return TEST_PROMPTS.get(category, [])


# Aliases for backwards compatibility
def get_test_prompts_flat() -> List[Dict]:
    """Alias for get_all_test_prompts."""
    return ALL_TEST_PROMPTS


def get_test_prompts_by_category(category: str) -> List[Dict]:
    """Alias for get_prompts_by_category."""
    return get_prompts_by_category(category)


def get_all_categories() -> List[str]:
    """Alias for get_categories."""
    return get_categories()


# =============================================================================
# PROMPT BUILDING UTILITIES
# =============================================================================

def build_full_prompt(system_prompt: str, instruction_prefix: str, user_prompt: str) -> str:
    """Combine system prompt, instruction prefix, and user prompt."""
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")
    
    user_part = instruction_prefix + user_prompt if instruction_prefix else user_prompt
    parts.append(f"User: {user_part}\n\nAssistant:")
    
    return "".join(parts)


def build_chat_prompt(system_prompt: str, user_prompt: str, tokenizer, instruction_prefix: str = "") -> str:
    """Build prompt using chat template if available."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    user_content = instruction_prefix + user_prompt if instruction_prefix else user_prompt
    messages.append({"role": "user", "content": user_content})
    
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    
    return build_full_prompt(system_prompt, instruction_prefix, user_prompt)


def get_system_prompt_text(name: str) -> str:
    """Get the text of a system prompt by name."""
    if name in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[name].get("text", "")
    return ""


# =============================================================================
# FILTERING UTILITIES
# =============================================================================

def filter_prompts(
    category: Optional[str] = None,
    max_count: Optional[int] = None
) -> List[Dict]:
    """Filter test prompts by category."""
    prompts = ALL_TEST_PROMPTS
    
    if category:
        prompts = [p for p in prompts if p.get("category") == category]
    
    if max_count:
        prompts = prompts[:max_count]
    
    return prompts


def filter_system_prompts(
    category: Optional[str] = None,
    exclude_keys: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Filter system prompts."""
    prompts = SYSTEM_PROMPTS.copy()
    
    if category:
        prompts = {k: v for k, v in prompts.items() 
                   if v.get("category") == category}
    
    if exclude_keys:
        prompts = {k: v for k, v in prompts.items() 
                   if k not in exclude_keys}
    
    return prompts


def reload_config():
    """Force reload configuration from JSON file."""
    global TEST_PROMPTS, ALL_TEST_PROMPTS, SYSTEM_PROMPTS, SYSTEM_PROMPTS_CORE, INSTRUCTION_PREFIXES, _json_config
    
    _json_config = _load_from_json()
    
    if _json_config:
        TEST_PROMPTS.clear()
        TEST_PROMPTS.update(_json_config.get("test_prompts", _DEFAULT_TEST_PROMPTS))
        SYSTEM_PROMPTS.clear()
        SYSTEM_PROMPTS.update(_json_config.get("system_prompts", _DEFAULT_SYSTEM_PROMPTS))
        INSTRUCTION_PREFIXES.clear()
        INSTRUCTION_PREFIXES.update(_json_config.get("instruction_prefixes", {"none": ""}))
    
    ALL_TEST_PROMPTS.clear()
    for category, prompts in TEST_PROMPTS.items():
        for p in prompts:
            ALL_TEST_PROMPTS.append({**p, "category": category})
    
    return True