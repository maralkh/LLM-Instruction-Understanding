"""
Metrics utilities for evaluating prompt effectiveness.
"""

import numpy as np
from scipy import stats
from scipy.special import kl_div, rel_entr
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class DistributionMetrics:
    """Metrics for analyzing probability distributions."""
    
    @staticmethod
    def entropy(probs: np.ndarray) -> float:
        """Calculate Shannon entropy of a distribution."""
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log(probs))
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate KL divergence D(P || Q).
        Measures how much P diverges from Q.
        """
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(rel_entr(p, q))
    
    @staticmethod
    def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
        """Symmetric KL divergence (Jensen-Shannon like)."""
        return (DistributionMetrics.kl_divergence(p, q) + 
                DistributionMetrics.kl_divergence(q, p)) / 2
    
    @staticmethod
    def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (symmetric, bounded [0, ln(2)])."""
        m = (p + q) / 2
        return (DistributionMetrics.kl_divergence(p, m) + 
                DistributionMetrics.kl_divergence(q, m)) / 2
    
    @staticmethod
    def top_k_overlap(
        dist1: List[Tuple[str, float]], 
        dist2: List[Tuple[str, float]], 
        k: int = 10
    ) -> float:
        """
        Calculate overlap between top-k tokens from two distributions.
        Returns fraction of overlap (0 to 1).
        """
        tokens1 = set(t[0] for t in dist1[:k])
        tokens2 = set(t[0] for t in dist2[:k])
        return len(tokens1 & tokens2) / k
    
    @staticmethod
    def rank_correlation(
        dist1: List[Tuple[str, float]], 
        dist2: List[Tuple[str, float]]
    ) -> float:
        """
        Calculate Spearman rank correlation between token rankings.
        Only considers tokens present in both top-k lists.
        """
        tokens1 = {t[0]: i for i, t in enumerate(dist1)}
        tokens2 = {t[0]: i for i, t in enumerate(dist2)}
        
        common = set(tokens1.keys()) & set(tokens2.keys())
        if len(common) < 2:
            return 0.0
            
        ranks1 = [tokens1[t] for t in common]
        ranks2 = [tokens2[t] for t in common]
        
        corr, _ = stats.spearmanr(ranks1, ranks2)
        return corr if not np.isnan(corr) else 0.0
    
    @staticmethod
    def probability_mass_on_target(
        distribution: np.ndarray,
        target_ids: List[int]
    ) -> float:
        """Calculate total probability mass on target tokens."""
        return sum(distribution[tid] for tid in target_ids if tid < len(distribution))
    
    @staticmethod
    def confidence_gap(probs: np.ndarray) -> float:
        """Gap between highest and second-highest probability."""
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) < 2:
            return 0.0
        return sorted_probs[0] - sorted_probs[1]


@dataclass
class SequenceMetrics:
    """Metrics for analyzing generated sequences."""
    
    @staticmethod
    def perplexity(log_probs: List[float]) -> float:
        """Calculate perplexity from log probabilities."""
        if not log_probs:
            return float('inf')
        avg_log_prob = np.mean(log_probs)
        return np.exp(-avg_log_prob)
    
    @staticmethod
    def mean_log_prob(log_probs: List[float]) -> float:
        """Average log probability."""
        return np.mean(log_probs) if log_probs else float('-inf')
    
    @staticmethod
    def total_log_prob(log_probs: List[float]) -> float:
        """Total log probability of sequence."""
        return sum(log_probs)
    
    @staticmethod
    def entropy_trajectory(entropies: List[float]) -> Dict:
        """Analyze how entropy changes over generation."""
        if not entropies:
            return {}
        return {
            "mean": np.mean(entropies),
            "std": np.std(entropies),
            "min": np.min(entropies),
            "max": np.max(entropies),
            "trend": np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0
        }
    
    @staticmethod
    def token_diversity(tokens: List[str]) -> float:
        """Ratio of unique tokens to total tokens."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)


@dataclass
class ComparisonMetrics:
    """Metrics for comparing prompts."""
    
    @staticmethod
    def relative_improvement(
        baseline_score: float,
        variant_score: float,
        higher_is_better: bool = True
    ) -> float:
        """Calculate relative improvement over baseline."""
        if baseline_score == 0:
            return float('inf') if variant_score > 0 else 0.0
        
        diff = variant_score - baseline_score
        if not higher_is_better:
            diff = -diff
        return diff / abs(baseline_score)
    
    @staticmethod
    def effect_size_cohens_d(
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        """Calculate Cohen's d effect size between two groups."""
        n1, n2 = len(scores1), len(scores2)
        var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        return (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    @staticmethod
    def statistical_significance(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """Run statistical tests comparing two score distributions."""
        # t-test
        t_stat, t_pval = stats.ttest_ind(scores1, scores2)
        
        # Mann-Whitney U (non-parametric)
        try:
            u_stat, u_pval = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        except ValueError:
            u_stat, u_pval = np.nan, np.nan
        
        return {
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "t_significant": t_pval < alpha,
            "mannwhitney_statistic": u_stat,
            "mannwhitney_pvalue": u_pval,
            "mannwhitney_significant": u_pval < alpha if not np.isnan(u_pval) else False,
            "cohens_d": ComparisonMetrics.effect_size_cohens_d(scores1, scores2)
        }


class ExperimentResults:
    """Container for experiment results with analysis methods."""
    
    def __init__(self):
        self.results: List[Dict] = []
        
    def add_result(self, result: Dict):
        self.results.append(result)
        
    def get_by_config(self, **kwargs) -> List[Dict]:
        """Filter results by configuration values."""
        filtered = self.results
        for key, value in kwargs.items():
            filtered = [r for r in filtered if r.get('config', {}).get(key) == value]
        return filtered
    
    def aggregate_by_dimension(
        self,
        dimension: str,
        metric: str,
        agg_fn: callable = np.mean
    ) -> Dict[str, float]:
        """Aggregate a metric across values of a dimension."""
        groups = {}
        for result in self.results:
            dim_value = result.get('config', {}).get(dimension)
            if dim_value is not None:
                if dim_value not in groups:
                    groups[dim_value] = []
                groups[dim_value].append(result.get(metric, 0))
        
        return {k: agg_fn(v) for k, v in groups.items()}
    
    def rank_variants(
        self,
        metric: str,
        higher_is_better: bool = True
    ) -> List[Dict]:
        """Rank all variants by a metric."""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get(metric, float('-inf') if higher_is_better else float('inf')),
            reverse=higher_is_better
        )
        return sorted_results
    
    def summary_statistics(self, metric: str) -> Dict:
        """Calculate summary statistics for a metric."""
        values = [r.get(metric) for r in self.results if r.get(metric) is not None]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75)
        }
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        
        rows = []
        for r in self.results:
            row = {**r}
            # Flatten config
            if 'config' in row:
                for k, v in row['config'].items():
                    row[f'config_{k}'] = v
                del row['config']
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentResults":
        """Load results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        results = cls()
        results.results = data
        return results


def compute_all_metrics(
    baseline_dist: np.ndarray,
    variant_dist: np.ndarray,
    baseline_top_k: List[Tuple[str, float]],
    variant_top_k: List[Tuple[str, float]]
) -> Dict:
    """Compute all distribution comparison metrics."""
    return {
        "kl_divergence": DistributionMetrics.kl_divergence(baseline_dist, variant_dist),
        "symmetric_kl": DistributionMetrics.symmetric_kl(baseline_dist, variant_dist),
        "jensen_shannon": DistributionMetrics.jensen_shannon(baseline_dist, variant_dist),
        "baseline_entropy": DistributionMetrics.entropy(baseline_dist),
        "variant_entropy": DistributionMetrics.entropy(variant_dist),
        "entropy_diff": DistributionMetrics.entropy(variant_dist) - DistributionMetrics.entropy(baseline_dist),
        "top_10_overlap": DistributionMetrics.top_k_overlap(baseline_top_k, variant_top_k, k=10),
        "top_5_overlap": DistributionMetrics.top_k_overlap(baseline_top_k, variant_top_k, k=5),
        "rank_correlation": DistributionMetrics.rank_correlation(baseline_top_k, variant_top_k),
        "baseline_confidence": DistributionMetrics.confidence_gap(baseline_dist),
        "variant_confidence": DistributionMetrics.confidence_gap(variant_dist)
    }
