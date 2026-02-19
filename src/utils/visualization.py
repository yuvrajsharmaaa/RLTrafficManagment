"""
Visualization Utilities for Traffic Signal Control Training and Evaluation.

Provides plots for:
- Training curves (rewards, losses)
- Evaluation comparison charts
- Metric distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import os


def plot_training_curves(
    metrics_tracker,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training curves from metrics tracker.
    
    Creates a 2x2 subplot with:
    - Episode rewards
    - Average waiting time
    - Queue length
    - Throughput
    
    Args:
        metrics_tracker: MetricsTracker instance with recorded data
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    
    episodes = range(1, len(metrics_tracker.episode_rewards) + 1)
    
    # Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, metrics_tracker.episode_rewards, alpha=0.3, color='blue')
    # Moving average
    window = min(50, len(metrics_tracker.episode_rewards))
    if window > 1:
        ma = np.convolve(
            metrics_tracker.episode_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        ax1.plot(range(window, len(metrics_tracker.episode_rewards) + 1), 
                 ma, color='blue', linewidth=2, label=f'{window}-ep MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average Waiting Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, metrics_tracker.episode_waiting_times, 
             alpha=0.3, color='red')
    if window > 1:
        ma = np.convolve(
            metrics_tracker.episode_waiting_times,
            np.ones(window)/window,
            mode='valid'
        )
        ax2.plot(range(window, len(metrics_tracker.episode_waiting_times) + 1),
                 ma, color='red', linewidth=2, label=f'{window}-ep MA')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Waiting Time (s)')
    ax2.set_title('Average Waiting Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Queue Length
    ax3 = axes[1, 0]
    ax3.plot(episodes, metrics_tracker.episode_queue_lengths,
             alpha=0.3, color='green')
    if window > 1:
        ma = np.convolve(
            metrics_tracker.episode_queue_lengths,
            np.ones(window)/window,
            mode='valid'
        )
        ax3.plot(range(window, len(metrics_tracker.episode_queue_lengths) + 1),
                 ma, color='green', linewidth=2, label=f'{window}-ep MA')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Avg Queue Length')
    ax3.set_title('Average Queue Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Throughput
    ax4 = axes[1, 1]
    ax4.plot(episodes, metrics_tracker.episode_throughputs,
             alpha=0.3, color='purple')
    if window > 1:
        ma = np.convolve(
            metrics_tracker.episode_throughputs,
            np.ones(window)/window,
            mode='valid'
        )
        ax4.plot(range(window, len(metrics_tracker.episode_throughputs) + 1),
                 ma, color='purple', linewidth=2, label=f'{window}-ep MA')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Vehicles Completed')
    ax4.set_title('Throughput')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_evaluation_results(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot evaluation results comparing different policies.
    
    Args:
        results: Dict mapping policy names to their metrics
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Policy Comparison', fontsize=14, fontweight='bold')
    
    policies = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))
    
    # Average Waiting Time
    ax1 = axes[0, 0]
    waiting_times = [results[p]['avg_waiting_time'] for p in policies]
    waiting_stds = [results[p].get('std_waiting_time', 0) for p in policies]
    bars1 = ax1.bar(policies, waiting_times, yerr=waiting_stds, 
                    color=colors, capsize=5)
    ax1.set_ylabel('Average Waiting Time (s)')
    ax1.set_title('Average Waiting Time (lower is better)')
    ax1.tick_params(axis='x', rotation=15)
    
    # Add value labels
    for bar, val in zip(bars1, waiting_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Average Queue Length
    ax2 = axes[0, 1]
    queue_lengths = [results[p]['avg_queue_length'] for p in policies]
    bars2 = ax2.bar(policies, queue_lengths, color=colors)
    ax2.set_ylabel('Average Queue Length')
    ax2.set_title('Average Queue Length (lower is better)')
    ax2.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars2, queue_lengths):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Throughput
    ax3 = axes[1, 0]
    throughputs = [results[p]['avg_throughput'] for p in policies]
    bars3 = ax3.bar(policies, throughputs, color=colors)
    ax3.set_ylabel('Total Throughput')
    ax3.set_title('Throughput (higher is better)')
    ax3.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars3, throughputs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Improvement Over Baseline
    ax4 = axes[1, 1]
    if 'Fixed-Time' in results:
        baseline = results['Fixed-Time']
        improvements = []
        policy_names = []
        
        for p in policies:
            if p != 'Fixed-Time':
                imp = ((baseline['avg_waiting_time'] - results[p]['avg_waiting_time']) /
                       baseline['avg_waiting_time'] * 100)
                improvements.append(imp)
                policy_names.append(p)
        
        if improvements:
            colors_subset = colors[1:len(improvements)+1] if len(policies) > 1 else colors
            bars4 = ax4.bar(policy_names, improvements, color=colors_subset)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_ylabel('Improvement (%)')
            ax4.set_title('Waiting Time Improvement vs Fixed-Time')
            ax4.tick_params(axis='x', rotation=15)
            
            for bar, val in zip(bars4, improvements):
                color = 'green' if val > 0 else 'red'
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:+.1f}%', ha='center', va='bottom', 
                         fontsize=9, color=color)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Evaluation results saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    comparison_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot detailed comparison between policies.
    
    Args:
        comparison_results: Output from compare_policies()
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Policy Comparison vs Baseline', fontsize=14, fontweight='bold')
    
    baseline = comparison_results.get('baseline', {})
    policies = [k for k in comparison_results.keys() if k != 'baseline']
    
    metrics = ['waiting_time_improvement', 'queue_length_improvement', 'throughput_improvement']
    titles = ['Waiting Time', 'Queue Length', 'Throughput']
    colors = plt.cm.Paired(np.linspace(0, 1, len(policies)))
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [comparison_results[p][metric] for p in policies]
        bars = ax.bar(policies, values, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement (%)')
        ax.set_title(f'{title} Improvement')
        ax.tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars, values):
            color = 'green' if val > 0 else 'red'
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:+.1f}%', ha='center', va=va, fontsize=10, color=color)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_episode_metrics(
    waiting_times: List[float],
    queue_lengths: List[int],
    episode_num: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot metrics for a single episode.
    
    Args:
        waiting_times: Waiting time at each step
        queue_lengths: Queue length at each step
        episode_num: Episode number
        save_path: Path to save figure
        show: Whether to display
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Episode {episode_num} Metrics', fontsize=14, fontweight='bold')
    
    steps = range(len(waiting_times))
    
    # Waiting Time
    ax1.plot(steps, waiting_times, color='red', alpha=0.7)
    ax1.fill_between(steps, waiting_times, alpha=0.3, color='red')
    ax1.set_ylabel('Total Waiting Time (s)')
    ax1.set_title('Waiting Time Over Episode')
    ax1.grid(True, alpha=0.3)
    
    # Queue Length
    ax2.plot(steps, queue_lengths, color='blue', alpha=0.7)
    ax2.fill_between(steps, queue_lengths, alpha=0.3, color='blue')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Total Queue Length')
    ax2.set_title('Queue Length Over Episode')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_training_animation(
    metrics_history: List[Dict],
    output_path: str,
    fps: int = 2
) -> None:
    """
    Create an animated GIF of training progress.
    
    Args:
        metrics_history: List of metrics dicts at different training points
        output_path: Path to save animation
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Skipping animation creation.")
        return
    
    frames = []
    
    for i, metrics in enumerate(metrics_history):
        fig = plot_single_frame(metrics, frame_num=i)
        
        # Save frame to buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Animation saved to {output_path}")


def plot_single_frame(metrics: Dict, frame_num: int) -> plt.Figure:
    """Create a single frame for animation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.text(0.5, 0.7, f"Episode: {metrics.get('episode', frame_num)}", 
            fontsize=20, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f"Avg Waiting: {metrics.get('avg_waiting_time', 0):.1f}s",
            fontsize=16, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f"Reward: {metrics.get('reward', 0):.1f}",
            fontsize=16, ha='center', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig


def print_summary_table(results: Dict[str, Dict]) -> None:
    """
    Print a formatted summary table of results.
    
    Args:
        results: Dict mapping policy names to metrics
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Policy':<20} {'Avg Wait (s)':<15} {'Avg Queue':<12} {'Throughput':<12} {'Reward':<12}")
    print("-"*80)
    
    # Find baseline for comparison
    baseline_wait = None
    if 'Fixed-Time' in results:
        baseline_wait = results['Fixed-Time']['avg_waiting_time']
    
    # Data rows
    for policy, metrics in results.items():
        wait = metrics['avg_waiting_time']
        queue = metrics['avg_queue_length']
        through = metrics['avg_throughput']
        reward = metrics.get('avg_reward', 0)
        
        # Calculate improvement
        if baseline_wait and policy != 'Fixed-Time':
            imp = ((baseline_wait - wait) / baseline_wait * 100)
            wait_str = f"{wait:.1f} ({imp:+.1f}%)"
        else:
            wait_str = f"{wait:.1f}"
        
        print(f"{policy:<20} {wait_str:<15} {queue:<12.1f} {through:<12.0f} {reward:<12.1f}")
    
    print("="*80 + "\n")
