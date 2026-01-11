"""
Shared test prompts and system prompt configurations for all experiments.

The core idea: Keep TEST_PROMPTS fixed across all experiments.
Vary only the SYSTEM_PROMPTS / INSTRUCTIONS to measure their effect.
"""

# =============================================================================
# FIXED TEST PROMPTS - Used across ALL experiments
# =============================================================================

# These prompts remain constant - we measure how different instructions
# change the model's response patterns to these exact prompts.

TEST_PROMPTS = {
    # Simple factual questions
    "factual": [
        {"id": "f1", "prompt": "What is the capital of France?", "expected": "Paris"},
        {"id": "f2", "prompt": "What is 15 + 27?", "expected": "42"},
        {"id": "f3", "prompt": "What color is the sky on a clear day?", "expected": "blue"},
        {"id": "f4", "prompt": "How many planets are in our solar system?", "expected": "8"},
        {"id": "f5", "prompt": "What is the chemical symbol for gold?", "expected": "Au"},
    ],
    
    # Reasoning questions
    "reasoning": [
        {"id": "r1", "prompt": "If all dogs are animals, and all animals need water, do dogs need water?", "expected": "yes"},
        {"id": "r2", "prompt": "John is taller than Mary. Mary is taller than Sue. Who is the shortest?", "expected": "Sue"},
        {"id": "r3", "prompt": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "expected": "0.05"},
        {"id": "r4", "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "expected": "5"},
        {"id": "r5", "prompt": "Is the following valid: All A are B. All B are C. Therefore all A are C.", "expected": "yes"},
    ],
    
    # Classification/judgment
    "classification": [
        {"id": "c1", "prompt": "Is the word 'excellent' positive, negative, or neutral?", "expected": "positive"},
        {"id": "c2", "prompt": "Is the word 'disaster' positive, negative, or neutral?", "expected": "negative"},
        {"id": "c3", "prompt": "Classify: 'The movie was okay, nothing special.' Positive, negative, or neutral?", "expected": "neutral"},
        {"id": "c4", "prompt": "Is this a question or statement: 'The weather is nice today.'", "expected": "statement"},
        {"id": "c5", "prompt": "Is Python a compiled or interpreted language?", "expected": "interpreted"},
    ],
    
    # Open-ended / creative
    "creative": [
        {"id": "o1", "prompt": "Give me a one-word synonym for 'happy'.", "expected": "joyful"},
        {"id": "o2", "prompt": "What's a good name for a pet goldfish?", "expected": None},  # No fixed answer
        {"id": "o3", "prompt": "Suggest one healthy breakfast food.", "expected": None},
        {"id": "o4", "prompt": "Name a famous scientist.", "expected": None},
        {"id": "o5", "prompt": "What rhymes with 'cat'?", "expected": "hat"},
    ],
    
    # Instruction following
    "instruction_following": [
        {"id": "i1", "prompt": "Say only the word 'hello' and nothing else.", "expected": "hello"},
        {"id": "i2", "prompt": "Respond with exactly three words.", "expected": None},
        {"id": "i3", "prompt": "Answer in all caps: What is 2+2?", "expected": "FOUR"},
        {"id": "i4", "prompt": "Reply with a single number: How many letters in 'cat'?", "expected": "3"},
        {"id": "i5", "prompt": "Start your response with 'Indeed,'", "expected": "Indeed"},
    ],
    
    # Potentially sensitive / edge cases
    "edge_cases": [
        {"id": "e1", "prompt": "What is 0 divided by 0?", "expected": "undefined"},
        {"id": "e2", "prompt": "Can you tell me something false?", "expected": None},
        {"id": "e3", "prompt": "What don't you know?", "expected": None},
        {"id": "e4", "prompt": "Summarize nothing.", "expected": None},
        {"id": "e5", "prompt": "What happens after we die?", "expected": None},
    ],
}

# Flatten for easy iteration
ALL_TEST_PROMPTS = []
for category, prompts in TEST_PROMPTS.items():
    for p in prompts:
        ALL_TEST_PROMPTS.append({**p, "category": category})


# =============================================================================
# SYSTEM PROMPTS / INSTRUCTIONS - The experimental variable
# =============================================================================

SYSTEM_PROMPTS = {
    # Baseline - no system prompt
    "none": {
        "text": "",
        "description": "No system prompt (baseline)"
    },
    
    # Minimal instructions
    "minimal": {
        "text": "You are a helpful assistant.",
        "description": "Minimal helpful assistant"
    },
    
    # Detailed helpful
    "helpful_detailed": {
        "text": """You are a helpful, harmless, and honest AI assistant. 
Your goal is to provide accurate, useful information while being respectful and clear.
Always strive to be helpful while avoiding harmful content.""",
        "description": "Detailed helpful/harmless/honest"
    },
    
    # Expert persona
    "expert": {
        "text": """You are an expert assistant with deep knowledge across many domains.
You provide precise, well-reasoned answers backed by expertise.
When uncertain, you acknowledge limitations.""",
        "description": "Expert persona"
    },
    
    # Concise/terse
    "concise": {
        "text": """Be extremely concise. Give the shortest possible accurate answer.
No explanations unless asked. One word or number when possible.""",
        "description": "Concise responses only"
    },
    
    # Verbose/detailed
    "verbose": {
        "text": """Provide thorough, detailed responses. Explain your reasoning.
Include relevant context and background information.
Be comprehensive in your answers.""",
        "description": "Verbose detailed responses"
    },
    
    # Chain of thought
    "cot": {
        "text": """Before answering, think through the problem step by step.
Show your reasoning process, then provide your final answer.
Always break down complex questions into simpler parts.""",
        "description": "Chain-of-thought reasoning"
    },
    
    # Careful/cautious
    "cautious": {
        "text": """Be very careful and accurate in your responses.
Double-check your answers before responding.
If uncertain, express your uncertainty clearly.
Avoid making claims you're not confident about.""",
        "description": "Cautious and careful"
    },
    
    # Confident/assertive
    "confident": {
        "text": """Respond with confidence and authority.
Give direct, decisive answers without hedging.
Be assertive in your responses.""",
        "description": "Confident and assertive"
    },
    
    # Friendly/casual
    "friendly": {
        "text": """Be warm, friendly, and conversational.
Use a casual tone like talking to a friend.
Feel free to use appropriate humor and be personable.""",
        "description": "Friendly casual tone"
    },
    
    # Formal/professional
    "formal": {
        "text": """Maintain a formal, professional tone at all times.
Use proper language and avoid colloquialisms.
Respond as you would in a professional business context.""",
        "description": "Formal professional tone"
    },
    
    # Role: Teacher
    "teacher": {
        "text": """You are a patient, encouraging teacher.
Explain concepts clearly and check for understanding.
Use examples and analogies to make things clear.
Encourage learning and curiosity.""",
        "description": "Teacher persona"
    },
    
    # Role: Scientist
    "scientist": {
        "text": """You are a rigorous scientist.
Base your answers on evidence and logical reasoning.
Distinguish between established facts and hypotheses.
Quantify uncertainty when possible.""",
        "description": "Scientist persona"
    },
    
    # Structured output
    "structured": {
        "text": """Always structure your responses clearly.
Use this format when appropriate:
- Answer: [direct answer]
- Explanation: [brief explanation]
- Confidence: [high/medium/low]""",
        "description": "Structured output format"
    },
    
    # Safety-focused
    "safety": {
        "text": """Prioritize safety and accuracy above all else.
Refuse to provide harmful, dangerous, or misleading information.
When in doubt, err on the side of caution.
Acknowledge when something is outside your capabilities.""",
        "description": "Safety-focused"
    },
    
    # Creative
    "creative": {
        "text": """Be creative and think outside the box.
Offer unique perspectives and interesting angles.
Don't be afraid to be imaginative in your responses.""",
        "description": "Creative thinking"
    },
}

# Subset for quick experiments
SYSTEM_PROMPTS_CORE = {k: SYSTEM_PROMPTS[k] for k in [
    "none", "minimal", "expert", "concise", "verbose", "cot", "cautious", "confident"
]}


# =============================================================================
# INSTRUCTION PREFIXES - Applied directly before the user prompt
# =============================================================================

INSTRUCTION_PREFIXES = {
    "none": "",
    "please": "Please answer: ",
    "accurate": "Answer accurately: ",
    "brief": "Answer briefly: ",
    "explain": "Explain your answer: ",
    "step_by_step": "Think step by step: ",
    "certain": "If you're certain, answer: ",
    "format_answer": "Answer: ",
    "consider": "Consider carefully and respond: ",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_full_prompt(system_prompt: str, instruction_prefix: str, user_prompt: str) -> str:
    """Combine system prompt, instruction prefix, and user prompt."""
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")
    
    user_part = instruction_prefix + user_prompt
    parts.append(f"User: {user_part}\n\nAssistant:")
    
    return "".join(parts)


def build_chat_prompt(system_prompt: str, user_prompt: str, tokenizer) -> str:
    """Build prompt using chat template if available."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_prompt})
    
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return build_full_prompt(system_prompt, "", user_prompt)


def get_test_prompts_flat():
    """Get all test prompts as a flat list."""
    return ALL_TEST_PROMPTS


def get_test_prompts_by_category(category: str):
    """Get test prompts for a specific category."""
    return TEST_PROMPTS.get(category, [])


def get_all_categories():
    """Get list of all test prompt categories."""
    return list(TEST_PROMPTS.keys())
