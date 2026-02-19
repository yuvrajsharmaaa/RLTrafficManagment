"""
Evaluation Script for Traffic Signal Controllers.

Evaluates trained DQN agent against baseline policies:
- Random policy
- Fixed-time policy

Reports metrics:
- Average waiting time per vehicle
- Average/max queue length
- Total throughput

Usage:
    python evaluate.py --model checkpoints/best_model.pt
    python evaluate.py --model checkpoints/best_model.pt --episodes 20 --gui
    python evaluate.py --quick-test  # Verify SUMO installation
"""

import argparse
import os
import sys
import yaml
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Set SUMO_HOME if not already set
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DQNAgent
from src.environment import TrafficEnvironment
from src.baselines import RandomPolicy, FixedTimePolicy
from src.utils.metrics import MetricsTracker, EvaluationResult, compare_policies
from src.utils.visualization import (
    plot_evaluation_results, 
    plot_comparison,
    print_summary_table
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_policy(env, policy, num_episodes: int, policy_name: str) -> EvaluationResult:
    """
    Evaluate a single policy for multiple episodes.
    
    Args:
        env: Traffic environment
        policy: Policy to evaluate (agent or baseline)
        num_episodes: Number of evaluation episodes
        policy_name: Name for logging
        
    Returns:
        EvaluationResult with episode metrics
    """
    result = EvaluationResult(policy_name)
    
    print(f"\nEvaluating {policy_name}...")
    
    for ep in range(1, num_episodes + 1):
        # Reset environment and policy
        state, _ = env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        
        episode_reward = 0
        waiting_times = []
        queue_lengths = []
        done = False
        
        while not done:
            # Select action (disable exploration for trained agents)
            if hasattr(policy, 'select_action'):
                if hasattr(policy, 'epsilon'):  # DQN agent
                    action = policy.select_action(state, training=False)
                else:  # Baseline policy
                    action = policy.select_action(state)
            else:
                action = env.action_space.sample()
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            waiting_times.append(info.get('waiting_time', 0))
            queue_lengths.append(info.get('queue_length', 0))
            
            state = next_state
        
        # Get episode summary
        ep_metrics = env.get_episode_metrics()
        
        result.add_episode(
            reward=episode_reward,
            waiting_time=ep_metrics.get('avg_waiting_time', np.mean(waiting_times)),
            queue_length=ep_metrics.get('avg_queue_length', np.mean(queue_lengths)),
            max_queue=ep_metrics.get('max_queue_length', np.max(queue_lengths)),
            throughput=ep_metrics.get('throughput', 0)
        )
        
        print(f"  Episode {ep}/{num_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"AvgWait={result.avg_waiting_times[-1]:.1f}s, "
              f"Throughput={result.throughputs[-1]}")
    
    return result


def evaluate(args):
    """Main evaluation function."""
    
    # Load configuration
    config_path = args.config if args.config else PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Override config with args
    if args.episodes:
        config['evaluation']['num_episodes'] = args.episodes
    if args.gui:
        config['environment']['use_gui'] = True
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / config['evaluation']['metrics_dir'] / timestamp
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("DQN Traffic Signal Optimization - Evaluation")
    print("="*60)
    
    # Create environment
    print("\nInitializing environment...")
    env_config = config['environment']
    env_config['net_file'] = str(PROJECT_ROOT / env_config['net_file'])
    env_config['route_file'] = str(PROJECT_ROOT / env_config['route_file'])
    env_config['waiting_time_weight'] = config['reward']['waiting_time_weight']
    env_config['phase_change_penalty'] = config['reward']['phase_change_penalty']
    
    env = TrafficEnvironment(**env_config)
    
    num_episodes = config['evaluation']['num_episodes']
    print(f"Evaluation episodes per policy: {num_episodes}")
    
    # Initialize policies to evaluate
    policies = {}
    
    # 1. Random Policy
    print("\nInitializing Random Policy...")
    policies['Random'] = RandomPolicy(
        action_dim=env.action_dim,
        seed=config['baselines']['random'].get('seed', 42)
    )
    
    # 2. Fixed-Time Policy
    print("Initializing Fixed-Time Policy...")
    fixed_config = config['baselines']['fixed_time']
    policies['Fixed-Time'] = FixedTimePolicy(
        action_dim=env.action_dim,
        ns_green_duration=fixed_config['ns_green_duration'],
        ew_green_duration=fixed_config['ew_green_duration'],
        delta_time=config['environment']['delta_time']
    )
    
    # 3. Trained DQN Agent
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        
        if model_path.exists():
            print(f"Loading DQN agent from: {model_path}")
            dqn_config = config['dqn']
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_layers=dqn_config['hidden_layers'],
                activation=dqn_config['activation']
            )
            agent.load(str(model_path))
            agent.epsilon = 0  # No exploration during evaluation
            policies['DQN'] = agent
        else:
            print(f"Warning: Model file not found: {model_path}")
            print("Skipping DQN evaluation.")
    else:
        print("No model specified. Evaluating baselines only.")
        print("Use --model <path> to evaluate a trained agent.")
    
    # Evaluate all policies
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    
    all_results = {}
    
    for name, policy in policies.items():
        result = evaluate_policy(env, policy, num_episodes, name)
        all_results[name] = result.get_summary()
    
    # Print summary
    print_summary_table(all_results)
    
    # Compare against Fixed-Time baseline
    if 'Fixed-Time' in all_results and len(all_results) > 1:
        comparison = compare_policies(all_results, baseline_name='Fixed-Time')
        
        print("\nIMPROVEMENT OVER FIXED-TIME BASELINE:")
        print("-"*40)
        for policy_name, comp_data in comparison.items():
            if policy_name != 'baseline':
                wti = comp_data['waiting_time_improvement']
                qi = comp_data['queue_length_improvement']
                ti = comp_data['throughput_improvement']
                print(f"{policy_name}:")
                print(f"  Waiting Time: {wti:+.1f}%")
                print(f"  Queue Length: {qi:+.1f}%")
                print(f"  Throughput:   {ti:+.1f}%")
    
    # Check if DQN beats baselines by 20-50%
    if 'DQN' in all_results and 'Fixed-Time' in all_results:
        dqn_wait = all_results['DQN']['avg_waiting_time']
        fixed_wait = all_results['Fixed-Time']['avg_waiting_time']
        improvement = ((fixed_wait - dqn_wait) / fixed_wait) * 100
        
        print("\n" + "="*60)
        if improvement >= 20:
            print(f"SUCCESS! DQN beats Fixed-Time by {improvement:.1f}%")
            if improvement >= 50:
                print("Excellent performance! Exceeds 50% improvement target.")
            print("Target of 20-50% improvement achieved!")
        else:
            print(f"DQN improvement: {improvement:.1f}%")
            print("Note: Target is 20-50% improvement over Fixed-Time baseline.")
            print("Consider training for more episodes or tuning hyperparameters.")
        print("="*60)
    
    # Save results
    results_file = results_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    if len(all_results) > 1:
        plot_path = str(results_dir / "evaluation_comparison.png")
        plot_evaluation_results(all_results, save_path=plot_path, show=args.show_plots)
        print(f"Comparison plot saved to: {plot_path}")
        
        if 'Fixed-Time' in all_results:
            comparison_path = str(results_dir / "improvement_chart.png")
            comparison = compare_policies(all_results, baseline_name='Fixed-Time')
            plot_comparison(comparison, save_path=comparison_path, show=args.show_plots)
            print(f"Improvement chart saved to: {comparison_path}")
    
    # Close environment
    env.close()
    
    return all_results


def run_quick_test(args):
    """Run a quick test to verify setup works."""
    print("\n" + "="*60)
    print("RUNNING QUICK TEST")
    print("="*60)
    
    config_path = args.config if args.config else PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    
    env_config = config['environment']
    env_config['net_file'] = str(PROJECT_ROOT / env_config['net_file'])
    env_config['route_file'] = str(PROJECT_ROOT / env_config['route_file'])
    env_config['num_seconds'] = 300  # Short simulation
    env_config['waiting_time_weight'] = config['reward']['waiting_time_weight']
    env_config['phase_change_penalty'] = config['reward']['phase_change_penalty']
    
    print("Creating environment...")
    env = TrafficEnvironment(**env_config)
    
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    
    print("\nRunning 10 random steps...")
    state, _ = env.reset()
    
    for i in range(10):
        action = np.random.randint(env.action_dim)
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, "
              f"wait={info['waiting_time']:.1f}")
        if done:
            break
        state = next_state
    
    env.close()
    print("\nQuick test completed successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate traffic signal controllers"
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained DQN model (.pt file)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of evaluation episodes per policy'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Enable SUMO GUI visualization'
    )
    parser.add_argument(
        '--show-plots', action='store_true',
        help='Display plots (in addition to saving)'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Run a quick test to verify setup'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
