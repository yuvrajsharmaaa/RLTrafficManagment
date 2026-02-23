"""
Training Script for DQN Traffic Signal Controller.

Trains a DQN agent to optimize traffic signal timing at
a single 4-way intersection using SUMO simulation.

Usage:
    python train.py                    # Train with default config
    python train.py --episodes 1000    # Train for 1000 episodes
    python train.py --gui              # Train with SUMO GUI visualization
"""

import argparse
import os
import sys
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

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
else:
    print("WARNING: SUMO_HOME not found. Please install SUMO first.")
    print("Download from: https://sumo.dlr.de/docs/Downloads.php")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DQNAgent
from src.environment import TrafficEnvironment
from src.utils.metrics import MetricsTracker
from src.utils.visualization import plot_training_curves


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(args):
    """Main training loop."""
    
    # Load configuration
    config_path = args.config if args.config else PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Override config with command-line arguments
    if args.episodes:
        config['training']['num_episodes'] = args.episodes
    if args.gui:
        config['environment']['use_gui'] = True
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Set random seeds
    seed = config['training'].get('seed', 42)
    np.random.seed(seed)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dqn_traffic_{timestamp}"
    
    checkpoint_dir = PROJECT_ROOT / config['training']['checkpoint_dir'] / experiment_name
    log_dir = PROJECT_ROOT / config['training']['log_dir'] / experiment_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*60)
    print("DQN Traffic Signal Optimization - Training")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log dir: {log_dir}")
    print("="*60)
    
    # Create environment
    print("\nInitializing environment...")
    env_config = config['environment']
    env_config['net_file'] = str(PROJECT_ROOT / env_config['net_file'])
    env_config['route_file'] = str(PROJECT_ROOT / env_config['route_file'])
    if 'additional_file' in env_config:
        env_config['additional_file'] = str(PROJECT_ROOT / env_config['additional_file'])
    env_config['seed'] = seed
    
    # Remove keys not accepted by TrafficEnvironment
    env_config.pop('sumo_cfg', None)
    
    # Add reward config
    env_config['waiting_time_weight'] = config['reward']['waiting_time_weight']
    env_config['phase_change_penalty'] = config['reward']['phase_change_penalty']
    
    env = TrafficEnvironment(**env_config)
    
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Create DQN agent
    print("\nInitializing DQN agent...")
    dqn_config = config['dqn']
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        epsilon_start=dqn_config['epsilon_start'],
        epsilon_end=dqn_config['epsilon_end'],
        epsilon_decay=dqn_config['epsilon_decay'],
        batch_size=dqn_config['batch_size'],
        buffer_size=dqn_config['buffer_size'],
        target_update_freq=dqn_config['target_update_freq'],
        hidden_layers=dqn_config['hidden_layers'],
        activation=dqn_config['activation'],
        use_double=True,  # Enable Double DQN
        seed=seed
    )
    
    # Initialize metrics tracker
    metrics = MetricsTracker(log_dir=str(log_dir), experiment_name=experiment_name)
    
    # Training settings
    num_episodes = config['training']['num_episodes']
    max_steps = config['training']['max_steps_per_episode']
    save_freq = config['training']['save_freq']
    eval_freq = config['training']['eval_freq']
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print("-"*60)
    
    best_reward = float('-inf')
    
    try:
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, info = env.reset()
            metrics.start_episode()
            
            episode_reward = 0
            step = 0
            
            while step < max_steps:
                # Select action
                action = agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train_step()
                
                # Log metrics
                metrics.log_step(
                    reward=reward,
                    waiting_time=info.get('waiting_time', 0),
                    queue_length=info.get('queue_length', 0),
                    loss=loss
                )
                
                episode_reward += reward
                state = next_state
                step += 1
                
                if done:
                    break
            
            # Get episode metrics
            ep_metrics = env.get_episode_metrics()
            summary = metrics.end_episode(
                throughput=ep_metrics.get('throughput', 0),
                phase_changes=ep_metrics.get('phase_changes', 0)
            )
            
            # Update agent episode counter
            agent.episode_count = episode
            
            # Print progress
            if episode % 10 == 0 or episode == 1:
                stats = metrics.get_recent_stats(window=50)
                print(f"Episode {episode:4d} | "
                      f"Reward: {summary['reward']:8.1f} | "
                      f"Avg Wait: {summary['avg_waiting_time']:6.1f}s | "
                      f"Queue: {summary['avg_queue_length']:5.1f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Avg(50): {stats.get('avg_reward', 0):8.1f}")
            
            # Save best model
            if summary['reward'] > best_reward:
                best_reward = summary['reward']
                agent.save(str(checkpoint_dir / "best_model.pt"))
            
            # Periodic checkpoint
            if episode % save_freq == 0:
                agent.save(str(checkpoint_dir / f"checkpoint_ep{episode}.pt"))
                metrics.save()
        
        print("-"*60)
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        # Save final model and metrics
        agent.save(str(checkpoint_dir / "final_model.pt"))
        metrics_path = metrics.save()
        
        print(f"\nFinal model saved to: {checkpoint_dir / 'final_model.pt'}")
        print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")
        print(f"Metrics saved to: {metrics_path}")
        
        # Plot training curves
        if len(metrics.episode_rewards) > 0:
            plot_path = str(log_dir / "training_curves.png")
            plot_training_curves(metrics, save_path=plot_path, show=False)
            print(f"Training curves saved to: {plot_path}")
        
        # Print summary
        final_stats = metrics.get_recent_stats(window=100)
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(metrics.episode_rewards)}")
        print(f"Best Reward: {best_reward:.1f}")
        print(f"Final Avg Reward (100 ep): {final_stats.get('avg_reward', 0):.1f}")
        print(f"Best Avg Waiting Time: {metrics.best_waiting_time:.1f}s")
        print(f"Final Avg Waiting Time: {final_stats.get('avg_waiting_time', 0):.1f}s")
        print("="*60)
        
        # Close environment
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN agent for traffic signal optimization"
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Enable SUMO GUI visualization'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
