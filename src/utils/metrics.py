"""
Metrics Tracking and Computation for Traffic Signal Control.

Tracks and computes:
- Average waiting time per vehicle
- Queue lengths (average and maximum)
- Total throughput (vehicles passed)
- Reward statistics
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class MetricsTracker:
    """
    Tracks training and evaluation metrics across episodes.
    
    Provides logging, aggregation, and export functionality.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name for this experiment
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_waiting_times = []
        self.episode_queue_lengths = []
        self.episode_throughputs = []
        self.episode_phase_changes = []
        
        # Step-level metrics (for current episode)
        self.step_rewards = []
        self.step_waiting_times = []
        self.step_queue_lengths = []
        self.step_losses = []
        
        # Training stats
        self.train_steps = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.best_waiting_time = float('inf')
    
    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self.step_rewards = []
        self.step_waiting_times = []
        self.step_queue_lengths = []
    
    def log_step(
        self,
        reward: float,
        waiting_time: float,
        queue_length: int,
        loss: Optional[float] = None
    ) -> None:
        """
        Log metrics for a single step.
        
        Args:
            reward: Step reward
            waiting_time: Total waiting time at step
            queue_length: Total queue length at step
            loss: Training loss (if training)
        """
        self.step_rewards.append(reward)
        self.step_waiting_times.append(waiting_time)
        self.step_queue_lengths.append(queue_length)
        
        if loss is not None:
            self.step_losses.append(loss)
        
        self.total_steps += 1
    
    def end_episode(
        self,
        throughput: int = 0,
        phase_changes: int = 0
    ) -> Dict:
        """
        Finalize metrics for completed episode.
        
        Args:
            throughput: Total vehicles that completed trips
            phase_changes: Number of phase changes in episode
            
        Returns:
            Episode summary metrics
        """
        ep_reward = sum(self.step_rewards)
        ep_length = len(self.step_rewards)
        avg_waiting = np.mean(self.step_waiting_times) if self.step_waiting_times else 0
        avg_queue = np.mean(self.step_queue_lengths) if self.step_queue_lengths else 0
        max_queue = np.max(self.step_queue_lengths) if self.step_queue_lengths else 0
        
        # Store episode metrics
        self.episode_rewards.append(ep_reward)
        self.episode_lengths.append(ep_length)
        self.episode_waiting_times.append(avg_waiting)
        self.episode_queue_lengths.append(avg_queue)
        self.episode_throughputs.append(throughput)
        self.episode_phase_changes.append(phase_changes)
        
        # Update best metrics
        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
        if avg_waiting < self.best_waiting_time:
            self.best_waiting_time = avg_waiting
        
        # Return summary
        return {
            'episode': len(self.episode_rewards),
            'reward': ep_reward,
            'length': ep_length,
            'avg_waiting_time': avg_waiting,
            'avg_queue_length': avg_queue,
            'max_queue_length': max_queue,
            'throughput': throughput,
            'phase_changes': phase_changes
        }
    
    def get_recent_stats(self, window: int = 100) -> Dict:
        """Get statistics over recent episodes."""
        if not self.episode_rewards:
            return {}
        
        window = min(window, len(self.episode_rewards))
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-window:]),
            'std_reward': np.std(self.episode_rewards[-window:]),
            'avg_waiting_time': np.mean(self.episode_waiting_times[-window:]),
            'avg_queue_length': np.mean(self.episode_queue_lengths[-window:]),
            'avg_throughput': np.mean(self.episode_throughputs[-window:]),
            'avg_loss': np.mean(self.step_losses[-1000:]) if self.step_losses else 0,
            'best_reward': self.best_reward,
            'best_waiting_time': self.best_waiting_time,
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save all metrics to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        
        data = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_waiting_times': self.episode_waiting_times,
            'episode_queue_lengths': self.episode_queue_lengths,
            'episode_throughputs': self.episode_throughputs,
            'episode_phase_changes': self.episode_phase_changes,
            'best_reward': float(self.best_reward),
            'best_waiting_time': float(self.best_waiting_time)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.experiment_name = data.get('experiment_name', self.experiment_name)
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.episode_waiting_times = data.get('episode_waiting_times', [])
        self.episode_queue_lengths = data.get('episode_queue_lengths', [])
        self.episode_throughputs = data.get('episode_throughputs', [])
        self.episode_phase_changes = data.get('episode_phase_changes', [])
        self.total_steps = data.get('total_steps', 0)
        self.best_reward = data.get('best_reward', float('-inf'))
        self.best_waiting_time = data.get('best_waiting_time', float('inf'))


def compute_metrics(
    waiting_times: List[float],
    queue_lengths: List[int],
    throughput: int,
    phase_changes: int
) -> Dict:
    """
    Compute comprehensive metrics from episode data.
    
    Args:
        waiting_times: List of waiting times per step
        queue_lengths: List of queue lengths per step
        throughput: Total vehicles completed
        phase_changes: Number of phase changes
        
    Returns:
        Dictionary of computed metrics
    """
    return {
        'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
        'max_waiting_time': np.max(waiting_times) if waiting_times else 0,
        'total_waiting_time': np.sum(waiting_times) if waiting_times else 0,
        'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
        'max_queue_length': np.max(queue_lengths) if queue_lengths else 0,
        'throughput': throughput,
        'phase_changes': phase_changes,
        'waiting_time_per_vehicle': (
            np.sum(waiting_times) / throughput if throughput > 0 else 0
        )
    }


def compare_policies(
    results: Dict[str, Dict],
    baseline_name: str = "Fixed-Time"
) -> Dict:
    """
    Compare multiple policies and compute improvements.
    
    Args:
        results: Dictionary mapping policy names to their metrics
        baseline_name: Name of baseline policy for comparison
        
    Returns:
        Comparison results with improvement percentages
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")
    
    baseline = results[baseline_name]
    comparison = {}
    
    for name, metrics in results.items():
        if name == baseline_name:
            continue
        
        # Compute improvements (negative is better for waiting time)
        waiting_improvement = (
            (baseline['avg_waiting_time'] - metrics['avg_waiting_time']) /
            baseline['avg_waiting_time'] * 100
            if baseline['avg_waiting_time'] > 0 else 0
        )
        
        queue_improvement = (
            (baseline['avg_queue_length'] - metrics['avg_queue_length']) /
            baseline['avg_queue_length'] * 100
            if baseline['avg_queue_length'] > 0 else 0
        )
        
        throughput_improvement = (
            (metrics['throughput'] - baseline['throughput']) /
            baseline['throughput'] * 100
            if baseline['throughput'] > 0 else 0
        )
        
        comparison[name] = {
            'waiting_time_improvement': waiting_improvement,
            'queue_length_improvement': queue_improvement,
            'throughput_improvement': throughput_improvement,
            'metrics': metrics
        }
    
    comparison['baseline'] = {
        'name': baseline_name,
        'metrics': baseline
    }
    
    return comparison


class EvaluationResult:
    """Stores evaluation results for a single policy."""
    
    def __init__(self, policy_name: str):
        self.policy_name = policy_name
        self.episodes = []
        self.avg_waiting_times = []
        self.avg_queue_lengths = []
        self.max_queue_lengths = []
        self.throughputs = []
        self.total_rewards = []
    
    def add_episode(
        self,
        reward: float,
        waiting_time: float,
        queue_length: float,
        max_queue: float,
        throughput: int
    ):
        self.episodes.append(len(self.episodes) + 1)
        self.total_rewards.append(reward)
        self.avg_waiting_times.append(waiting_time)
        self.avg_queue_lengths.append(queue_length)
        self.max_queue_lengths.append(max_queue)
        self.throughputs.append(throughput)
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all episodes."""
        return {
            'policy': self.policy_name,
            'num_episodes': len(self.episodes),
            'avg_reward': np.mean(self.total_rewards),
            'std_reward': np.std(self.total_rewards),
            'avg_waiting_time': np.mean(self.avg_waiting_times),
            'std_waiting_time': np.std(self.avg_waiting_times),
            'avg_queue_length': np.mean(self.avg_queue_lengths),
            'max_queue_length': np.max(self.max_queue_lengths),
            'avg_throughput': np.mean(self.throughputs),
            'std_throughput': np.std(self.throughputs)
        }
