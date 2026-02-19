"""
Test/Evaluate Trained Agent vs Baselines

Compares:
- Trained DQN agent
- Random policy baseline
- Fixed-time policy baseline

Usage:
    python test_agent.py --model checkpoints/dqn_sb3_xxx/dqn_traffic_final.zip
    python test_agent.py --model checkpoints/dqn_sb3_xxx/dqn_traffic_final.zip --gui
    python test_agent.py --baselines-only  # Only run baselines
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Set SUMO_HOME if not already set
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment

# Project paths
NET_FILE = "networks/single_intersection.net.xml"
ROUTE_FILE = "networks/single_intersection.rou.xml"


class FixedTimePolicy:
    """Fixed-time traffic signal controller: cycles through phases."""
    
    def __init__(self, num_phases, phase_duration=30, delta_time=5):
        self.num_phases = num_phases
        self.phase_duration = phase_duration
        self.delta_time = delta_time
        self.current_phase = 0
        self.time_in_phase = 0
    
    def predict(self, obs, deterministic=True):
        """Return action based on fixed timing."""
        if self.time_in_phase >= self.phase_duration:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.time_in_phase = 0
        
        self.time_in_phase += self.delta_time
        return self.current_phase, None
    
    def reset(self):
        self.current_phase = 0
        self.time_in_phase = 0


class RandomPolicy:
    """Random action selection."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


def run_evaluation(policy, env, policy_name, num_episodes=5, use_gui=False):
    """Run evaluation episodes and collect metrics."""
    print(f"\nEvaluating: {policy_name}")
    print("-" * 40)
    
    all_rewards = []
    all_waiting_times = []
    all_queue_lengths = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        # Reset fixed-time policy if applicable
        if hasattr(policy, 'reset'):
            policy.reset()
        
        done = False
        episode_reward = 0
        episode_waiting = []
        episode_queue = []
        steps = 0
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Track waiting time and queue from info if available
            if 'agents_total_accumulated_waiting_time' in info:
                episode_waiting.append(list(info['agents_total_accumulated_waiting_time'].values())[0])
            if 'agents_total_stopped' in info:
                episode_queue.append(list(info['agents_total_stopped'].values())[0])
        
        all_rewards.append(episode_reward)
        if episode_waiting:
            all_waiting_times.append(np.mean(episode_waiting))
        if episode_queue:
            all_queue_lengths.append(np.mean(episode_queue))
        
        print(f"  Episode {ep+1}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    results = {
        'policy': policy_name,
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_waiting_time': np.mean(all_waiting_times) if all_waiting_times else 0,
        'avg_queue_length': np.mean(all_queue_lengths) if all_queue_lengths else 0,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic signal policies")
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI')
    parser.add_argument('--baselines-only', action='store_true', help='Only run baselines')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Traffic Signal Policy Evaluation")
    print("=" * 60)
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    # Create environment for evaluation
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=f"outputs/eval_{timestamp}.csv",
        use_gui=args.gui,
        single_agent=True,
        num_seconds=3600,
        delta_time=5,
        yellow_time=3,
        min_green=10,
        max_green=50,
    )
    
    print(f"Action space: {env.action_space}")
    num_phases = env.action_space.n
    
    # Evaluate Random Policy
    random_policy = RandomPolicy(env.action_space)
    random_results = run_evaluation(random_policy, env, "Random", args.episodes, args.gui)
    results.append(random_results)
    
    # Evaluate Fixed-Time Policy
    fixed_policy = FixedTimePolicy(num_phases, phase_duration=30, delta_time=5)
    fixed_results = run_evaluation(fixed_policy, env, "Fixed-Time (30s)", args.episodes, args.gui)
    results.append(fixed_results)
    
    # Evaluate Trained Model
    if args.model and not args.baselines_only:
        if os.path.exists(args.model):
            print(f"\nLoading model: {args.model}")
            model = DQN.load(args.model)
            dqn_results = run_evaluation(model, env, "DQN (Trained)", args.episodes, args.gui)
            results.append(dqn_results)
        else:
            print(f"\nWARNING: Model not found: {args.model}")
    
    env.close()
    
    # Print Results Table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Policy':<20} {'Avg Reward':<15} {'Std Reward':<12} {'Avg Wait':<12} {'Avg Queue':<12}")
    print("-" * 60)
    
    baseline_reward = None
    for r in results:
        if r['policy'] == 'Fixed-Time (30s)':
            baseline_reward = r['avg_reward']
        
        print(f"{r['policy']:<20} {r['avg_reward']:<15.1f} {r['std_reward']:<12.1f} "
              f"{r['avg_waiting_time']:<12.1f} {r['avg_queue_length']:<12.1f}")
    
    print("-" * 60)
    
    # Calculate improvements
    if baseline_reward and len(results) > 2:
        dqn_reward = results[-1]['avg_reward']  # Last is DQN
        improvement = ((dqn_reward - baseline_reward) / abs(baseline_reward)) * 100
        print(f"\nDQN vs Fixed-Time improvement: {improvement:+.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = f"results/evaluation_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
