"""
Test Environment Script
Verifies SUMO-RL environment is working correctly.
"""

import os
import sys

# Set SUMO_HOME if not already set (adjust path if installed elsewhere)
if 'SUMO_HOME' not in os.environ:
    # Common Windows installation paths - adjust if needed
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break
    else:
        print("ERROR: SUMO not found! Please install SUMO:")
        print("  1. Download from: https://sumo.dlr.de/docs/Downloads.php")
        print("  2. Install to default location")
        print("  3. Or set SUMO_HOME environment variable manually")
        sys.exit(1)

print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")

# Add SUMO tools to path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import gymnasium as gym
from sumo_rl import SumoEnvironment

# Use the network files from our project
NET_FILE = "networks/single_intersection.net.xml"
ROUTE_FILE = "networks/single_intersection.rou.xml"

def test_environment():
    """Test the SUMO-RL environment with random actions."""
    print("\n" + "="*60)
    print("Testing SUMO-RL Environment")
    print("="*60)
    
    # Create environment
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name="outputs/test_output.csv",
        use_gui=True,  # Set False for faster headless runs
        single_agent=True,
        num_seconds=1000,  # Shorter test run
        delta_time=5,
        yellow_time=3,
        min_green=10,
        max_green=50,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run episode with random actions
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    done = False
    total_reward = 0
    steps = 0
    
    print("\nRunning simulation with random actions...")
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 50 == 0:
            print(f"  Step {steps}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Output saved to: outputs/test_output.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    test_environment()
