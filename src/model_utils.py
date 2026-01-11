"""
Model utilities for prompt engineering experiments.
Handles model loading, inference, and probability extraction.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16
    load_in_8bit: bool = False
    

@dataclass 
class GenerationOutput:
    """Container for generation outputs with probabilities."""
    text: str
    tokens: List[int]
    token_strings: List[str]
    log_probs: List[float]
    top_k_tokens: List[List[Tuple[str, float]]]  # [(token, prob), ...]
    entropy: List[float]
    

class PromptEngineeringModel:
    """Wrapper for LLM with probability extraction capabilities."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        load_kwargs = {
            "torch_dtype": self.config.torch_dtype,
            "device_map": self.config.device,
            "trust_remote_code": True,
        }
        
        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )
        self.model.eval()
        print(f"Model loaded on {self.config.device}")
        
    @torch.no_grad()
    def get_next_token_distribution(
        self, 
        prompt: str,
        top_k: int = 50
    ) -> Dict:
        """
        Get the probability distribution over next tokens.
        
        Returns:
            Dict with top_k tokens, their probabilities, and entropy
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model(**inputs)
        
        # Get logits for the last position
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [
            (self.tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return {
            "top_tokens": top_tokens,
            "entropy": entropy,
            "full_probs": probs.cpu().numpy(),
            "logits": logits.cpu().numpy()
        }
    
    @torch.no_grad()
    def get_sequence_log_probs(
        self,
        prompt: str,
        completion: str
    ) -> Dict:
        """
        Get log probabilities for each token in the completion given the prompt.
        
        Returns:
            Dict with per-token log probs and total sequence log prob
        """
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.config.device)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get log probs for completion tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        token_log_probs = []
        completion_tokens = []
        
        for i in range(prompt_len, inputs.input_ids.shape[1]):
            token_id = inputs.input_ids[0, i].item()
            token_log_prob = log_probs[0, i-1, token_id].item()
            token_log_probs.append(token_log_prob)
            completion_tokens.append(self.tokenizer.decode([token_id]))
            
        return {
            "tokens": completion_tokens,
            "log_probs": token_log_probs,
            "total_log_prob": sum(token_log_probs),
            "avg_log_prob": np.mean(token_log_probs) if token_log_probs else 0
        }
    
    @torch.no_grad()
    def generate_with_probs(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k_track: int = 10,
        do_sample: bool = False
    ) -> GenerationOutput:
        """
        Generate text while tracking probabilities at each step.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_len = inputs.input_ids.shape[1]
        
        generated_ids = inputs.input_ids.clone()
        all_log_probs = []
        all_top_k = []
        all_entropies = []
        
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=generated_ids)
            logits = outputs.logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Entropy
            entropy = -torch.sum(probs * log_probs).item()
            all_entropies.append(entropy)
            
            # Top-k tokens
            top_probs, top_indices = torch.topk(probs, top_k_track)
            top_k = [
                (self.tokenizer.decode([idx]), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
            all_top_k.append(top_k)
            
            # Sample or greedy
            if do_sample:
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(probs).unsqueeze(0)
                
            next_log_prob = log_probs[next_token].item()
            all_log_probs.append(next_log_prob)
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Decode
        generated_tokens = generated_ids[0, input_len:].tolist()
        token_strings = [self.tokenizer.decode([t]) for t in generated_tokens]
        full_text = self.tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
        
        return GenerationOutput(
            text=full_text,
            tokens=generated_tokens,
            token_strings=token_strings,
            log_probs=all_log_probs,
            top_k_tokens=all_top_k,
            entropy=all_entropies
        )
    
    @torch.no_grad()
    def compare_prompts(
        self,
        prompts: List[str],
        target_completion: str
    ) -> List[Dict]:
        """
        Compare how different prompts affect the probability of a target completion.
        """
        results = []
        for prompt in prompts:
            seq_probs = self.get_sequence_log_probs(prompt, target_completion)
            next_dist = self.get_next_token_distribution(prompt)
            
            results.append({
                "prompt": prompt,
                "completion_log_prob": seq_probs["total_log_prob"],
                "avg_token_log_prob": seq_probs["avg_log_prob"],
                "next_token_entropy": next_dist["entropy"],
                "top_5_next": next_dist["top_tokens"][:5]
            })
            
        return results
    
    def get_hidden_states(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden states from specified layers.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
            
        hidden_states = outputs.hidden_states
        
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))
            
        return {
            i: hidden_states[i][0].cpu().numpy()
            for i in layer_indices
        }


def load_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = None
) -> PromptEngineeringModel:
    """Convenience function to load a model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = ModelConfig(
        model_name=model_name,
        device=device
    )
    return PromptEngineeringModel(config)
