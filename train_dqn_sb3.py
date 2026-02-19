"""
Train DQN Agent for Traffic Signal Optimization using Stable-Baselines3

This script trains a DQN agent to optimize traffic signal timing
at a single 4-way intersection using SUMO simulation.

Usage:
    python train_dqn_sb3.py                  # Train with defaults
    python train_dqn_sb3.py --timesteps 100000
    python train_dqn_sb3.py --gui            # With visualization (slower)
"""

import os
import sys
import argparse
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

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from sumo_rl import SumoEnvironment

# Project paths
NET_FILE = "networks/single_intersection.net.xml"
ROUTE_FILE = "networks/single_intersection.rou.xml"


def create_env(use_gui=False, out_csv="outputs/train_output.csv"):
    """Create and wrap the SUMO environment."""
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=out_csv,
        use_gui=use_gui,
        single_agent=True,
        num_seconds=3600,  # 1 hour simulation
        delta_time=5,      # 5 seconds per action
        yellow_time=3,
        min_green=10,
        max_green=50,
    )
    return Monitor(env)


def train(args):
    """Train DQN agent."""
    print("="*60)
    print("DQN Traffic Signal Optimization - Stable-Baselines3")
    print("="*60)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/dqn_sb3_{timestamp}"
    checkpoint_dir = f"./checkpoints/dqn_sb3_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Tensorboard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Total timesteps: {args.timesteps}")
    print("="*60)
    
    # Create environment
    print("\nCreating environment...")
    env = create_env(use_gui=args.gui, out_csv=f"outputs/train_{timestamp}.csv")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create DQN agent
    print("\nInitializing DQN agent...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        # Hyperparameters
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,  # Hard update
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        # Exploration
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        # Network
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="dqn_traffic",
        save_replay_buffer=True,
    )
    
    # Train
    print(f"\nStarting training for {args.timesteps} timesteps...")
    print("Monitor with: tensorboard --logdir ./logs/")
    print("-"*60)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            log_interval=10,
            tb_log_name="DQN",
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    model_path = f"{checkpoint_dir}/dqn_traffic_final"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}.zip")
    
    env.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"To evaluate: python test_agent.py --model {model_path}.zip")
    print(f"To view logs: tensorboard --logdir {log_dir}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN agent for traffic signal optimization"
    )
    parser.add_argument(
        '--timesteps', type=int, default=50000,
        help='Total training timesteps (default: 50000)'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Enable SUMO GUI (slower but visual)'
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
