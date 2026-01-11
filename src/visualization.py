"""
Visualization utilities for prompt engineering experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import pandas as pd


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_distribution_comparison(
    distributions: List[Dict],
    labels: List[str],
    top_k: int = 20,
    title: str = "Token Distribution Comparison",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Compare token probability distributions from different prompts.
    
    Args:
        distributions: List of dicts with 'top_tokens' key
        labels: Labels for each distribution
        top_k: Number of tokens to show
    """
    fig, axes = plt.subplots(1, len(distributions), figsize=figsize, sharey=True)
    if len(distributions) == 1:
        axes = [axes]
    
    for ax, dist, label in zip(axes, distributions, labels):
        tokens = [t[0] for t in dist['top_tokens'][:top_k]]
        probs = [t[1] for t in dist['top_tokens'][:top_k]]
        
        # Clean token strings for display
        tokens = [repr(t)[1:-1] if t.strip() != t else t for t in tokens]
        
        bars = ax.barh(range(len(tokens)), probs, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel('Probability')
        ax.set_title(f'{label}\nEntropy: {dist.get("entropy", 0):.3f}')
        ax.invert_yaxis()
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_entropy_comparison(
    results: List[Dict],
    group_by: str = 'config',
    title: str = "Entropy by Prompt Configuration"
) -> plt.Figure:
    """Plot entropy values grouped by configuration."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group results
    if group_by == 'config':
        labels = [str(r.get('config', {})) for r in results]
    else:
        labels = [r.get(group_by, str(i)) for i, r in enumerate(results)]
    
    entropies = [r.get('next_token_entropy', r.get('entropy', 0)) for r in results]
    
    # Sort by entropy
    sorted_pairs = sorted(zip(labels, entropies), key=lambda x: x[1], reverse=True)
    labels, entropies = zip(*sorted_pairs)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(entropies)))[::-1]
    bars = ax.barh(range(len(labels)), entropies, color=colors)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Entropy (nats)')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_log_prob_comparison(
    results: List[Dict],
    title: str = "Completion Log Probability by Prompt"
) -> plt.Figure:
    """Compare completion log probabilities across prompts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [r.get('prompt', '')[:50] + '...' for r in results]
    log_probs = [r.get('completion_log_prob', 0) for r in results]
    
    # Sort by log prob
    sorted_pairs = sorted(zip(labels, log_probs), key=lambda x: x[1], reverse=True)
    labels, log_probs = zip(*sorted_pairs)
    
    colors = ['green' if lp > np.median(log_probs) else 'red' for lp in log_probs]
    bars = ax.barh(range(len(labels)), log_probs, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Log Probability')
    ax.set_title(title)
    ax.axvline(x=np.median(log_probs), color='gray', linestyle='--', label='Median')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_dimension_heatmap(
    df: pd.DataFrame,
    dim1: str,
    dim2: str,
    metric: str,
    title: str = None,
    cmap: str = 'RdYlGn'
) -> plt.Figure:
    """
    Create heatmap showing metric values across two dimensions.
    
    Args:
        df: DataFrame with experiment results
        dim1, dim2: Column names for the two dimensions
        metric: Column name for the metric to plot
    """
    pivot = df.pivot_table(values=metric, index=dim1, columns=dim2, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax, 
                cbar_kws={'label': metric})
    
    ax.set_title(title or f'{metric} by {dim1} and {dim2}')
    plt.tight_layout()
    return fig


def plot_ablation_results(
    baseline_score: float,
    ablation_scores: Dict[str, float],
    metric_name: str = "Performance",
    title: str = "Ablation Study Results"
) -> plt.Figure:
    """Plot results of ablation study."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Baseline'] + list(ablation_scores.keys())
    scores = [baseline_score] + list(ablation_scores.values())
    
    colors = ['green'] + ['red' if s < baseline_score else 'blue' for s in scores[1:]]
    bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.7)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.axhline(y=baseline_score, color='green', linestyle='--', alpha=0.5)
    
    # Add percentage change labels
    for i, (score, bar) in enumerate(zip(scores, bars)):
        if i > 0:
            pct_change = ((score - baseline_score) / abs(baseline_score)) * 100
            ax.annotate(f'{pct_change:+.1f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_entropy_trajectory(
    entropies: List[float],
    token_strings: List[str] = None,
    title: str = "Entropy During Generation"
) -> plt.Figure:
    """Plot how entropy changes during generation."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    positions = range(len(entropies))
    ax.plot(positions, entropies, 'b-o', markersize=4, linewidth=1.5)
    ax.fill_between(positions, entropies, alpha=0.3)
    
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Entropy (nats)')
    ax.set_title(title)
    
    if token_strings:
        # Add token labels at intervals
        step = max(1, len(token_strings) // 15)
        for i in range(0, len(token_strings), step):
            ax.annotate(repr(token_strings[i])[:10], 
                       xy=(i, entropies[i]),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=7, rotation=45, ha='left')
    
    # Add trend line
    if len(entropies) > 1:
        z = np.polyfit(positions, entropies, 1)
        p = np.poly1d(z)
        ax.plot(positions, p(positions), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.4f})')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    results_by_model: Dict[str, List[Dict]],
    metric: str,
    title: str = "Model Comparison"
) -> plt.Figure:
    """Compare metrics across different models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results_by_model.keys())
    positions = np.arange(len(model_names))
    
    metric_values = []
    for model in model_names:
        values = [r.get(metric, 0) for r in results_by_model[model]]
        metric_values.append(values)
    
    # Box plot
    bp = ax.boxplot(metric_values, positions=positions, widths=0.6, patch_artist=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=30, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_prompt_strategy_matrix(
    df: pd.DataFrame,
    strategy_col: str,
    task_col: str,
    metric_col: str,
    title: str = "Prompt Strategy × Task Matrix"
) -> plt.Figure:
    """Create matrix showing which strategies work best for which tasks."""
    # Pivot and normalize within each task
    pivot = df.pivot_table(values=metric_col, index=strategy_col, columns=task_col, aggfunc='mean')
    normalized = pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(normalized, annot=pivot.round(3), fmt='', cmap='RdYlGn', 
                ax=ax, cbar_kws={'label': 'Normalized Score'})
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


def create_summary_dashboard(
    results: List[Dict],
    dimensions: List[str],
    metric: str = 'completion_log_prob'
) -> plt.Figure:
    """Create a multi-panel summary dashboard."""
    n_dims = len(dimensions)
    fig, axes = plt.subplots(2, max(2, n_dims), figsize=(6*max(2, n_dims), 10))
    
    # Convert to DataFrame
    rows = []
    for r in results:
        row = {metric: r.get(metric, 0)}
        for dim in dimensions:
            row[dim] = r.get('config', {}).get(dim, 'N/A')
        rows.append(row)
    df = pd.DataFrame(rows)
    
    # Top row: metric distribution by each dimension
    for i, dim in enumerate(dimensions):
        ax = axes[0, i] if n_dims > 1 else axes[0, 0]
        grouped = df.groupby(dim)[metric].mean().sort_values(ascending=False)
        grouped.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_title(f'{metric} by {dim}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Fill remaining top row axes
    for i in range(n_dims, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Bottom left: overall distribution
    ax = axes[1, 0]
    df[metric].hist(bins=20, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(df[metric].mean(), color='red', linestyle='--', label=f'Mean: {df[metric].mean():.3f}')
    ax.set_title(f'Distribution of {metric}')
    ax.legend()
    
    # Bottom right: top 10 configurations
    ax = axes[1, 1]
    top_10 = df.nlargest(10, metric)
    config_labels = [str({d: row[d] for d in dimensions})[:40] for _, row in top_10.iterrows()]
    ax.barh(range(10), top_10[metric].values, color='green', alpha=0.7)
    ax.set_yticks(range(10))
    ax.set_yticklabels(config_labels, fontsize=8)
    ax.set_title('Top 10 Configurations')
    ax.invert_yaxis()
    
    # Fill remaining bottom row axes
    for i in range(2, axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.suptitle('Experiment Summary Dashboard', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig
