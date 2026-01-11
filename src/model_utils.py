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


@dataclass
class InternalsOutput:
    """Container for model internal states."""
    hidden_states: Dict[int, np.ndarray]  # layer -> (seq_len, hidden_dim)
    attentions: Dict[int, np.ndarray]  # layer -> (n_heads, seq_len, seq_len)
    residuals: Dict[str, np.ndarray]
    logits: np.ndarray
    
    def attention_to_token(self, layer: int, head: int, from_pos: int, to_pos: int) -> float:
        if layer in self.attentions:
            return float(self.attentions[layer][head, from_pos, to_pos])
        return 0.0
    
    def mean_attention_to_position(self, layer: int, target_pos: int) -> float:
        if layer in self.attentions:
            return float(self.attentions[layer][:, :, target_pos].mean())
        return 0.0


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

    @torch.no_grad()
    def get_internals(self, prompt: str, layers: Optional[List[int]] = None) -> InternalsOutput:
        """Extract model internals: hidden states and attention patterns."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if layers is None:
            layers = list(range(len(hidden_states)))
        hs_dict, attn_dict = {}, {}
        for layer_idx in layers:
            if layer_idx < len(hidden_states):
                hs_dict[layer_idx] = hidden_states[layer_idx][0].cpu().numpy()
            if attentions and layer_idx < len(attentions):
                attn_dict[layer_idx] = attentions[layer_idx][0].cpu().numpy()
        return InternalsOutput(hidden_states=hs_dict, attentions=attn_dict, residuals={}, logits=outputs.logits[0, -1].cpu().numpy())

    @torch.no_grad()
    def get_attention_patterns(self, prompt: str, aggregate: str = "last_token") -> Dict:
        """Get attention patterns for analysis."""
        internals = self.get_internals(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        tokens = [self.tokenizer.decode([t]) for t in input_ids]
        results = {"tokens": tokens, "n_layers": len(internals.attentions), "n_heads": internals.attentions[0].shape[0] if internals.attentions else 0, "seq_len": len(tokens)}
        layer_attention = {}
        for layer_idx, attn in internals.attentions.items():
            if aggregate == "last_token":
                layer_attention[layer_idx] = attn[:, -1, :].mean(axis=0)
            elif aggregate == "mean":
                layer_attention[layer_idx] = attn.mean(axis=(0, 1))
            elif aggregate == "max":
                layer_attention[layer_idx] = attn.max(axis=(0, 1))
        results["layer_attention"] = layer_attention
        attn_entropy = {}
        for layer_idx, attn in internals.attentions.items():
            head_entropies = []
            for head in range(attn.shape[0]):
                for pos in range(attn.shape[1]):
                    probs = attn[head, pos, :]
                    probs = probs / (probs.sum() + 1e-10)
                    ent = -np.sum(probs * np.log(probs + 1e-10))
                    head_entropies.append(ent)
            attn_entropy[layer_idx] = np.mean(head_entropies)
        results["attention_entropy"] = attn_entropy
        return results

    @torch.no_grad()
    def compare_internals(self, prompt1: str, prompt2: str, layers: Optional[List[int]] = None) -> Dict:
        """Compare model internals between two prompts."""
        int1 = self.get_internals(prompt1, layers)
        int2 = self.get_internals(prompt2, layers)
        comparison = {"hidden_state_diff": {}, "attention_diff": {}, "logit_diff": {}}
        for layer in set(int1.hidden_states.keys()) & set(int2.hidden_states.keys()):
            hs1, hs2 = int1.hidden_states[layer][-1], int2.hidden_states[layer][-1]
            comparison["hidden_state_diff"][layer] = {"cosine_sim": float(np.dot(hs1, hs2) / (np.linalg.norm(hs1) * np.linalg.norm(hs2) + 1e-10)), "l2_norm": float(np.linalg.norm(hs1 - hs2)), "mean_abs_diff": float(np.mean(np.abs(hs1 - hs2)))}
        for layer in set(int1.attentions.keys()) & set(int2.attentions.keys()):
            attn1, attn2 = int1.attentions[layer], int2.attentions[layer]
            min_len = min(attn1.shape[-1], attn2.shape[-1])
            a1, a2 = attn1[:, -1, :min_len].flatten(), attn2[:, -1, :min_len].flatten()
            comparison["attention_diff"][layer] = {"cosine_sim": float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-10)), "mean_abs_diff": float(np.mean(np.abs(a1 - a2)))}
        logits1, logits2 = int1.logits, int2.logits
        comparison["logit_diff"] = {"cosine_sim": float(np.dot(logits1, logits2) / (np.linalg.norm(logits1) * np.linalg.norm(logits2) + 1e-10)), "l2_norm": float(np.linalg.norm(logits1 - logits2)), "top_token_same": int(np.argmax(logits1) == np.argmax(logits2))}
        return comparison

    @torch.no_grad()
    def get_head_contributions(self, prompt: str, target_token_idx: int = -1) -> Dict:
        """Analyze contribution of each attention head."""
        internals = self.get_internals(prompt)
        head_importance = {}
        for layer_idx, attn in internals.attentions.items():
            n_heads = attn.shape[0]
            head_stats = []
            for head in range(n_heads):
                head_attn = attn[head, target_token_idx, :]
                probs = head_attn / (head_attn.sum() + 1e-10)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_attn_pos = int(np.argmax(head_attn))
                max_attn_val = float(head_attn.max())
                head_stats.append({"head": head, "entropy": float(entropy), "max_attention_position": max_attn_pos, "max_attention_value": max_attn_val, "attention_to_start": float(head_attn[0]), "attention_to_self": float(head_attn[target_token_idx]) if target_token_idx >= 0 else 0})
            head_importance[layer_idx] = head_stats
        return head_importance

    def get_layer_info(self) -> Dict:
        """Get information about model layers."""
        return {"n_layers": self.model.config.num_hidden_layers, "n_heads": self.model.config.num_attention_heads, "hidden_size": self.model.config.hidden_size, "vocab_size": self.model.config.vocab_size, "model_name": self.config.model_name}


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
