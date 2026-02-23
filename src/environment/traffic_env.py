"""
Traffic Environment Wrapper for SUMO-RL.

Provides a customizable wrapper around sumo-rl environment with:
- Custom state representation
- Custom reward function with phase change penalty
- Metrics tracking
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Tuple, Any, List
import os
import sys

# Check for SUMO_HOME - auto-detect common paths
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
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    raise EnvironmentError(
        "SUMO_HOME not found! Please install SUMO:\n"
        "  Windows: Download from https://sumo.dlr.de/docs/Downloads.php\n"
        "  Linux: sudo apt-get install sumo sumo-tools\n"
        "Then set SUMO_HOME environment variable."
    )

import sumo_rl


class TrafficEnvironment:
    """
    Custom Traffic Environment Wrapper.
    
    Wraps sumo-rl environment with custom state representation and reward function.
    
    State Vector:
        - Lane densities (vehicles/capacity per lane)
        - Lane queues (stopped vehicles/capacity per lane)
        - Current phase one-hot encoding
        - Time since phase start (normalized)
    
    Reward:
        - Negative total waiting time
        - Phase change penalty
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        max_green: int = 50,
        reward_fn: str = "custom",
        waiting_time_weight: float = -1.0,
        phase_change_penalty: float = 0.1,
        seed: Optional[int] = None,
        additional_file: Optional[str] = None
    ):
        """
        Initialize the traffic environment.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            use_gui: Whether to show SUMO GUI
            num_seconds: Simulation duration in seconds
            delta_time: Seconds per action step
            yellow_time: Yellow phase duration
            min_green: Minimum green phase duration
            max_green: Maximum green phase duration
            reward_fn: Reward function type ('custom' or sumo-rl built-in)
            waiting_time_weight: Weight for waiting time in reward
            phase_change_penalty: Penalty for changing phases
            seed: Random seed
        """
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_fn = reward_fn
        self.waiting_time_weight = waiting_time_weight
        self.phase_change_penalty = phase_change_penalty
        self.seed = seed
        self.additional_file = additional_file
        
        # Build additional SUMO command string
        additional_cmd = None
        if additional_file:
            additional_cmd = f"--additional-files {additional_file}"
        
        # Create SUMO-RL environment
        self.env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=True,
            additional_sumo_cmd=additional_cmd
        )
        
        # Get traffic signal ID (single intersection)
        self.ts_id = list(self.env.traffic_signals.keys())[0]
        self.traffic_signal = self.env.traffic_signals[self.ts_id]
        
        # State and action dimensions
        self.num_green_phases = self.traffic_signal.num_green_phases
        self.num_phases = self.num_green_phases  # only green phases are valid actions
        self.num_lanes = len(self.traffic_signal.lanes)
        
        # State dim: densities + queues + phase one-hot + time on phase
        self.state_dim = 2 * self.num_lanes + self.num_phases + 1
        self.action_dim = self.num_phases
        
        # Tracking
        self.last_phase = None
        self.time_on_phase = 0
        self.total_waiting_time = 0
        self.prev_waiting_time = 0
        
        # Episode metrics
        self.episode_metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': 0,
            'phase_changes': 0
        }
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            Initial state and info dict
        """
        if seed is not None:
            self.seed = seed
        
        # Reset SUMO-RL environment
        obs, info = self.env.reset()
        
        # Re-acquire traffic signal reference after SUMO restart
        self.traffic_signal = self.env.traffic_signals[self.ts_id]
        
        # Reset tracking
        self.last_phase = self.traffic_signal.green_phase
        self.time_on_phase = 0
        self.prev_waiting_time = 0
        
        # Reset metrics
        self.episode_metrics = {
            'waiting_times': [],
            'queue_lengths': [],
            'throughput': 0,
            'phase_changes': 0
        }
        
        # Get custom state
        state = self._get_state()
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return transition.
        
        Args:
            action: Phase index to switch to
            
        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Execute action in SUMO-RL
        obs, sumo_reward, terminated, truncated, info = self.env.step(action)
        
        # Get custom reward
        reward = self._compute_reward(action)
        
        # Get custom state
        state = self._get_state()
        
        # Update tracking
        if action != self.last_phase:
            self.time_on_phase = 0
            self.episode_metrics['phase_changes'] += 1
        else:
            self.time_on_phase += self.delta_time
        self.last_phase = action
        
        # Track metrics
        self._track_metrics()
        
        # Add custom info
        info['custom_reward'] = reward
        info['waiting_time'] = self._get_total_waiting_time()
        info['queue_length'] = self._get_total_queue()
        
        return state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        Construct custom state vector.
        
        Returns:
            State vector with:
            - Lane densities
            - Lane queues (normalized)
            - Phase one-hot
            - Time on phase (normalized)
        """
        ts = self.traffic_signal
        
        # Lane densities and queues
        densities = []
        queues = []
        
        for lane in ts.lanes:
            # Density: vehicles / lane_capacity
            density = ts.sumo.lane.getLastStepVehicleNumber(lane) / (
                ts.sumo.lane.getLength(lane) / 7.5  # Avg vehicle length + gap
            )
            densities.append(min(density, 1.0))
            
            # Queue: halting vehicles / lane_capacity
            queue = ts.sumo.lane.getLastStepHaltingNumber(lane) / (
                ts.sumo.lane.getLength(lane) / 7.5
            )
            queues.append(min(queue, 1.0))
        
        # Current phase one-hot
        phase_one_hot = np.zeros(self.num_phases)
        phase_one_hot[ts.green_phase] = 1.0
        
        # Time on phase (normalized by max_green)
        time_normalized = min(self.time_on_phase / self.max_green, 1.0)
        
        # Combine
        state = np.concatenate([
            np.array(densities, dtype=np.float32),
            np.array(queues, dtype=np.float32),
            phase_one_hot.astype(np.float32),
            np.array([time_normalized], dtype=np.float32)
        ])
        
        return state
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute custom reward.
        
        Reward = waiting_time_weight * total_waiting_time 
                 - phase_change_penalty * (1 if phase changed else 0)
        """
        # Waiting time component
        current_waiting = self._get_total_waiting_time()
        waiting_reward = self.waiting_time_weight * (current_waiting - self.prev_waiting_time)
        self.prev_waiting_time = current_waiting
        
        # Phase change penalty
        phase_penalty = 0
        if self.last_phase is not None and action != self.last_phase:
            phase_penalty = self.phase_change_penalty
        
        reward = waiting_reward - phase_penalty
        
        return reward
    
    def _get_total_waiting_time(self) -> float:
        """Get total waiting time across all vehicles."""
        ts = self.traffic_signal
        waiting_time = 0
        for lane in ts.lanes:
            waiting_time += ts.sumo.lane.getWaitingTime(lane)
        return waiting_time
    
    def _get_total_queue(self) -> int:
        """Get total queue length (halting vehicles) across all lanes."""
        ts = self.traffic_signal
        queue = 0
        for lane in ts.lanes:
            queue += ts.sumo.lane.getLastStepHaltingNumber(lane)
        return queue
    
    def _track_metrics(self) -> None:
        """Track episode metrics."""
        self.episode_metrics['waiting_times'].append(self._get_total_waiting_time())
        self.episode_metrics['queue_lengths'].append(self._get_total_queue())
        # Use traci to count departed vehicles (compatible with sumo-rl >= 1.4)
        try:
            import traci
            self.episode_metrics['throughput'] = traci.simulation.getArrivedNumber()
        except Exception:
            self.episode_metrics['throughput'] = len(self.env.vehicles)
    
    def get_episode_metrics(self) -> Dict:
        """Get metrics for completed episode."""
        metrics = self.episode_metrics.copy()
        
        if metrics['waiting_times']:
            metrics['avg_waiting_time'] = np.mean(metrics['waiting_times'])
            metrics['max_waiting_time'] = np.max(metrics['waiting_times'])
        else:
            metrics['avg_waiting_time'] = 0
            metrics['max_waiting_time'] = 0
        
        if metrics['queue_lengths']:
            metrics['avg_queue_length'] = np.mean(metrics['queue_lengths'])
            metrics['max_queue_length'] = np.max(metrics['queue_lengths'])
        else:
            metrics['avg_queue_length'] = 0
            metrics['max_queue_length'] = 0
        
        return metrics
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Return observation space."""
        return gym.spaces.Box(
            low=0, high=1, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
    
    @property
    def action_space(self):
        """Return action space."""
        return gym.spaces.Discrete(self.action_dim)


def create_sumo_env(config: Dict) -> TrafficEnvironment:
    """
    Create traffic environment from config dict.
    
    Args:
        config: Configuration dictionary with environment settings
        
    Returns:
        TrafficEnvironment instance
    """
    return TrafficEnvironment(
        net_file=config.get('net_file'),
        route_file=config.get('route_file'),
        use_gui=config.get('use_gui', False),
        num_seconds=config.get('num_seconds', 3600),
        delta_time=config.get('delta_time', 5),
        yellow_time=config.get('yellow_time', 3),
        min_green=config.get('min_green', 10),
        max_green=config.get('max_green', 50),
        reward_fn=config.get('reward_fn', 'custom'),
        waiting_time_weight=config.get('waiting_time_weight', -1.0),
        phase_change_penalty=config.get('phase_change_penalty', 0.1),
        seed=config.get('seed')
    )


class SimpleSumoEnv:
    """
    Simplified SUMO environment that works without sumo-rl dependency.
    
    Uses direct TraCI API for simulation control.
    Useful for testing or when sumo-rl is not available.
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 3,
        seed: Optional[int] = None
    ):
        import traci
        self.traci = traci
        
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.seed = seed
        
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.conn = None
        self.current_step = 0
        self.ts_id = None
        self.lanes = []
        self.phases = []
        self.current_phase = 0
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Start new episode."""
        if self.conn is not None:
            self.traci.close()
        
        # Build SUMO command
        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1"
        ]
        
        if self.seed is not None:
            sumo_cmd.extend(["--seed", str(self.seed)])
        
        # Start SUMO
        self.traci.start(sumo_cmd)
        self.conn = self.traci
        self.current_step = 0
        
        # Get traffic light info
        self.ts_id = self.traci.trafficlight.getIDList()[0]
        self.lanes = list(self.traci.trafficlight.getControlledLanes(self.ts_id))
        
        # Get initial state
        state = self._get_state()
        return state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action for delta_time seconds."""
        # Set phase
        self.current_phase = action
        self.traci.trafficlight.setPhase(self.ts_id, action)
        
        # Simulate delta_time seconds
        for _ in range(self.delta_time):
            self.traci.simulationStep()
            self.current_step += 1
        
        # Get state and reward
        state = self._get_state()
        reward = -self._get_total_waiting_time()
        
        # Check termination
        terminated = self.current_step >= self.num_seconds
        truncated = False
        
        info = {
            'waiting_time': self._get_total_waiting_time(),
            'queue_length': self._get_total_queue()
        }
        
        return state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Get state vector."""
        densities = []
        queues = []
        
        for lane in self.lanes:
            # Density
            length = self.traci.lane.getLength(lane)
            num_vehicles = self.traci.lane.getLastStepVehicleNumber(lane)
            density = num_vehicles / (length / 7.5)
            densities.append(min(density, 1.0))
            
            # Queue
            halting = self.traci.lane.getLastStepHaltingNumber(lane)
            queue = halting / (length / 7.5)
            queues.append(min(queue, 1.0))
        
        state = np.array(densities + queues, dtype=np.float32)
        return state
    
    def _get_total_waiting_time(self) -> float:
        """Get total waiting time."""
        return sum(
            self.traci.lane.getWaitingTime(lane) 
            for lane in self.lanes
        )
    
    def _get_total_queue(self) -> int:
        """Get total queue length."""
        return sum(
            self.traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes
        )
    
    def close(self):
        """Close SUMO connection."""
        if self.conn is not None:
            self.traci.close()
            self.conn = None
