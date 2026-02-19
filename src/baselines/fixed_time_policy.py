"""
Fixed-Time Policy Baseline for Traffic Signal Control.

Implements a traditional fixed-timing traffic signal controller
that cycles through phases with predefined durations.
"""

import numpy as np
from typing import List, Optional, Dict


class FixedTimePolicy:
    """
    Fixed-Time Traffic Signal Controller.
    
    Cycles through phases with predefined durations.
    This is the traditional approach used in most traffic signals.
    
    Default cycle: 30s NS-green → 30s EW-green → repeat
    """
    
    def __init__(
        self,
        action_dim: int,
        phase_durations: Optional[List[int]] = None,
        delta_time: int = 5,
        ns_green_duration: int = 30,
        ew_green_duration: int = 30,
        include_left_turns: bool = True,
        left_turn_duration: int = 15
    ):
        """
        Initialize fixed-time policy.
        
        Args:
            action_dim: Number of phases (usually 4)
            phase_durations: List of duration for each phase (seconds)
            delta_time: Action step duration (seconds)
            ns_green_duration: Duration for NS through phase
            ew_green_duration: Duration for EW through phase
            include_left_turns: Whether to include left turn phases
            left_turn_duration: Duration for left turn phases
        """
        self.action_dim = action_dim
        self.delta_time = delta_time
        
        # Set up phase durations
        if phase_durations is not None:
            self.phase_durations = phase_durations
        else:
            # Default cycle for 4-phase intersection
            if action_dim >= 4 and include_left_turns:
                self.phase_durations = [
                    ns_green_duration,      # Phase 0: NS through
                    ew_green_duration,      # Phase 1: EW through
                    left_turn_duration,     # Phase 2: NS left
                    left_turn_duration      # Phase 3: EW left
                ]
            else:
                # Simple 2-phase cycle
                self.phase_durations = [
                    ns_green_duration,      # Phase 0: NS
                    ew_green_duration       # Phase 1: EW
                ]
        
        # Current state
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_time = 0
        
        # Calculate phase schedule
        self._build_schedule()
        
        self.name = "Fixed-Time Policy"
    
    def _build_schedule(self) -> None:
        """Build phase transition schedule."""
        self.schedule = []
        cumulative_time = 0
        
        for phase, duration in enumerate(self.phase_durations):
            self.schedule.append({
                'phase': phase,
                'start_time': cumulative_time,
                'end_time': cumulative_time + duration
            })
            cumulative_time += duration
        
        self.cycle_duration = cumulative_time
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on fixed timing schedule.
        
        Args:
            state: Current state (used only to extract time info)
            
        Returns:
            Phase index based on current time in cycle
        """
        # Determine where we are in the cycle
        cycle_time = self.total_time % self.cycle_duration
        
        # Find appropriate phase
        for phase_info in self.schedule:
            if phase_info['start_time'] <= cycle_time < phase_info['end_time']:
                action = phase_info['phase']
                break
        else:
            # Default to first phase
            action = 0
        
        # Update time tracking
        self.total_time += self.delta_time
        
        return action
    
    def reset(self) -> None:
        """Reset policy to beginning of cycle."""
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_time = 0
    
    def get_cycle_info(self) -> Dict:
        """Get information about current cycle."""
        cycle_time = self.total_time % self.cycle_duration
        current_phase = 0
        time_remaining = 0
        
        for phase_info in self.schedule:
            if phase_info['start_time'] <= cycle_time < phase_info['end_time']:
                current_phase = phase_info['phase']
                time_remaining = phase_info['end_time'] - cycle_time
                break
        
        return {
            'current_phase': current_phase,
            'cycle_time': cycle_time,
            'cycle_duration': self.cycle_duration,
            'time_remaining_in_phase': time_remaining,
            'phase_durations': self.phase_durations
        }
    
    def set_phase_durations(self, durations: List[int]) -> None:
        """Update phase durations dynamically."""
        self.phase_durations = durations
        self._build_schedule()
    
    def __repr__(self) -> str:
        return f"FixedTimePolicy(phases={self.phase_durations}, cycle={self.cycle_duration}s)"


class ActuatedPolicy:
    """
    Simple Actuated Traffic Signal Policy.
    
    Extends minimum green time based on vehicle detection.
    If vehicles are still arriving, extend green up to max_green.
    Otherwise, switch to next phase.
    
    This is more advanced than fixed-time but still rule-based.
    """
    
    def __init__(
        self,
        action_dim: int,
        min_green: int = 10,
        max_green: int = 50,
        extension_time: int = 3,
        delta_time: int = 5,
        queue_threshold: float = 0.1
    ):
        """
        Initialize actuated policy.
        
        Args:
            action_dim: Number of phases
            min_green: Minimum green time per phase
            max_green: Maximum green time per phase
            extension_time: Time to extend if vehicles detected
            delta_time: Action step duration
            queue_threshold: Queue ratio threshold for extension
        """
        self.action_dim = action_dim
        self.min_green = min_green
        self.max_green = max_green
        self.extension_time = extension_time
        self.delta_time = delta_time
        self.queue_threshold = queue_threshold
        
        self.current_phase = 0
        self.time_in_phase = 0
        
        self.name = "Actuated Policy"
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on current queue state.
        
        Extends current phase if queues exist and under max_green.
        Otherwise advances to next phase.
        
        Args:
            state: Current state [densities, queues, phase_onehot, time]
        """
        # Parse state to get queue information
        # Assuming state format: [densities, queues, phase_onehot, time]
        num_features = (len(state) - self.action_dim - 1) // 2
        queues = state[num_features:2*num_features]
        
        # Calculate average queue for current phase lanes
        # Simplified: use all queues
        avg_queue = np.mean(queues)
        
        # Decision logic
        if self.time_in_phase < self.min_green:
            # Stay in current phase (minimum green)
            action = self.current_phase
        elif self.time_in_phase >= self.max_green:
            # Switch to next phase (maximum green reached)
            action = (self.current_phase + 1) % self.action_dim
            self.current_phase = action
            self.time_in_phase = 0
        elif avg_queue > self.queue_threshold:
            # Extend current phase (vehicles still waiting)
            action = self.current_phase
        else:
            # No more vehicles, switch to next phase
            action = (self.current_phase + 1) % self.action_dim
            self.current_phase = action
            self.time_in_phase = 0
        
        self.time_in_phase += self.delta_time
        
        return action
    
    def reset(self) -> None:
        """Reset policy state."""
        self.current_phase = 0
        self.time_in_phase = 0
    
    def __repr__(self) -> str:
        return f"ActuatedPolicy(min={self.min_green}, max={self.max_green})"
