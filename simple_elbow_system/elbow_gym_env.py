"""
Phase 1.5: Gymnasium Environment for Simple Elbow System
Wraps our working elbow physics in a proper RL environment

Goal: Make our elbow system compatible with Stable-Baselines3
Task: Track target angles using muscle activations
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import our working elbow system
from dataclasses import dataclass

@dataclass
class ElbowParams:
    """Simple parameters for our elbow system"""
    # Joint properties
    joint_inertia: float = 0.5    # kg*m^2 (increased for stability)
    joint_damping: float = 2.0    # N*m*s/rad (more damping)
    
    # Muscle properties (balanced torques now)
    bicep_max_force: float = 200.0    # N (reduced)
    tricep_max_force: float = 250.0   # N (reduced) 
    bicep_moment_arm: float = 0.05    # m
    tricep_moment_arm: float = 0.04   # m (FIXED - balanced torques)
    
    # Simulation
    dt: float = 0.01  # 100 Hz simulation

class SimpleElbowSystem:
    """Our working elbow physics - unchanged"""
    
    def __init__(self, params=None):
        self.params = params or ElbowParams()
        
        # State variables
        self.angle = 0.0      # Current joint angle (rad)  
        self.velocity = 0.0   # Current joint velocity (rad/s)
        
        # Muscle activations (0-1)
        self.bicep_activation = 0.0
        self.tricep_activation = 0.0
        
        # For tracking
        self.time = 0.0
        
    def compute_muscle_forces(self):
        """Simple muscle force calculation - just activation * max_force"""
        bicep_force = self.bicep_activation * self.params.bicep_max_force
        tricep_force = self.tricep_activation * self.params.tricep_max_force
        return bicep_force, tricep_force
    
    def compute_joint_torque(self):
        """Convert muscle forces to joint torque"""
        bicep_force, tricep_force = self.compute_muscle_forces()
        
        # Bicep flexes (positive torque), tricep extends (negative torque)
        bicep_torque = bicep_force * self.params.bicep_moment_arm
        tricep_torque = -tricep_force * self.params.tricep_moment_arm
        
        total_torque = bicep_torque + tricep_torque
        return total_torque
    
    def step(self, bicep_activation, tricep_activation):
        """One simulation step - returns current angle"""
        # Clamp muscle activations to valid range
        self.bicep_activation = np.clip(bicep_activation, 0.0, 1.0)
        self.tricep_activation = np.clip(tricep_activation, 0.0, 1.0)
        
        # Compute forces and torques
        joint_torque = self.compute_joint_torque()
        
        # Simple joint dynamics: torque = inertia * acceleration + damping * velocity
        acceleration = (joint_torque - self.params.joint_damping * self.velocity) / self.params.joint_inertia
        
        # Integrate to get new velocity and angle
        self.velocity += acceleration * self.params.dt
        self.angle += self.velocity * self.params.dt
        
        # Update time
        self.time += self.params.dt
        
        return self.angle
    
    def get_state(self):
        """Get current state for RL"""
        return np.array([self.angle, self.velocity])
    
    def reset(self):
        """Reset to initial state"""
        self.angle = 0.0
        self.velocity = 0.0
        self.bicep_activation = 0.0
        self.tricep_activation = 0.0
        self.time = 0.0


class ElbowTrackingEnv(gym.Env):
    """
    Gymnasium environment for elbow angle tracking
    
    Observation: [current_angle, current_velocity, target_angle, angle_error]
    Action: [bicep_activation, tricep_activation] (both 0-1)
    Reward: -error^2 - effort_penalty
    """
    
    def __init__(self, max_episode_steps=500):
        super().__init__()
        
        # Initialize our elbow physics
        self.elbow = SimpleElbowSystem()
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Define action space: [bicep_activation, tricep_activation]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Define observation space: [angle, velocity, target, error]
        # Reasonable ranges: angle ±3 rad, velocity ±10 rad/s, error ±3 rad
        self.observation_space = spaces.Box(
            low=np.array([-3.0, -10.0, -3.0, -3.0]),
            high=np.array([3.0, 10.0, 3.0, 3.0]),
            dtype=np.float32
        )
        
        # Target tracking
        self.target_angle = 0.0
        self.target_frequency = 0.5  # Hz - target changes this fast
        
    def _generate_target(self):
        """Generate target angle - simple sine wave for now"""
        target = 0.5 * np.sin(2 * np.pi * self.target_frequency * self.elbow.time)
        return target
        
    def _get_observation(self):
        """Get current observation"""
        angle = self.elbow.angle
        velocity = self.elbow.velocity
        target = self.target_angle
        error = target - angle
        
        obs = np.array([angle, velocity, target, error], dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs
        
    def _calculate_reward(self, action):
        """Calculate reward: minimize tracking error and muscle effort"""
        error = abs(self.target_angle - self.elbow.angle)
        
        # Primary reward: negative squared error (encourages accurate tracking)
        tracking_reward = -error**2
        
        # Effort penalty: discourage excessive muscle activation
        effort_penalty = -0.01 * np.sum(action**2)
        
        # Small bonus for being close (within 0.1 rad ~ 6 degrees)
        proximity_bonus = 0.1 if error < 0.1 else 0.0
        
        total_reward = tracking_reward + effort_penalty + proximity_bonus
        return total_reward
        
    def reset(self, seed=None, options=None):
        """Reset environment to start new episode"""
        super().reset(seed=seed)
        
        # Reset physics
        self.elbow.reset()
        
        # Reset episode tracking
        self.current_step = 0
        
        # Set initial target
        self.target_angle = self._generate_target()
        
        # Get initial observation
        obs = self._get_observation()
        info = {"target_angle": self.target_angle}
        
        return obs, info
        
    def step(self, action):
        """Execute one environment step"""
        # Ensure action is correct shape and type
        action = np.array(action, dtype=np.float32)
        bicep_activation, tricep_activation = action[0], action[1]
        
        # Step the physics
        self.elbow.step(bicep_activation, tricep_activation)
        
        # Update target (makes it dynamic)
        self.target_angle = self._generate_target()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        self.current_step += 1
        terminated = False  # Task doesn't have a "success" termination condition
        truncated = self.current_step >= self.max_episode_steps
        
        # Info for debugging
        info = {
            "target_angle": self.target_angle,
            "current_angle": self.elbow.angle,
            "tracking_error": abs(self.target_angle - self.elbow.angle),
            "step": self.current_step
        }
        
        return obs, reward, terminated, truncated, info
        
    def render(self, mode="human"):
        """Simple text rendering for debugging"""
        if mode == "human":
            print(f"Step {self.current_step}: "
                  f"Angle={self.elbow.angle:.3f}, "
                  f"Target={self.target_angle:.3f}, "
                  f"Error={abs(self.target_angle - self.elbow.angle):.3f}")


def test_environment():
    """Test the environment works correctly"""
    print("=== Testing Gymnasium Environment ===")
    
    # Create environment
    env = ElbowTrackingEnv(max_episode_steps=200)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test a few random steps
    total_reward = 0
    tracking_errors = []
    
    for step in range(10):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        tracking_errors.append(info['tracking_error'])
        
        if step < 3:  # Show first few steps
            print(f"Step {step}: Action={action}, Reward={reward:.3f}, "
                  f"Error={info['tracking_error']:.3f}")
    
    print(f"Average tracking error: {np.mean(tracking_errors):.3f} rad")
    print(f"Total reward (10 steps): {total_reward:.2f}")
    
    return env

def test_with_simple_controller():
    """Test environment with our working proportional controller"""
    print("\n=== Testing with Simple Controller ===")
    
    env = ElbowTrackingEnv(max_episode_steps=500)
    obs, info = env.reset()
    
    # Simple proportional controller (like our working one)
    kp = 4.0
    total_reward = 0
    tracking_errors = []
    
    for step in range(100):
        # Extract error from observation
        error = obs[3]  # angle_error is 4th element
        
        # Simple controller logic
        if error > 0:
            # Need to flex (positive angle)
            action = [min(kp * error, 1.0), 0.0]
        else:
            # Need to extend (negative angle)
            action = [0.0, min(kp * abs(error), 1.0)]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        tracking_errors.append(info['tracking_error'])
        
        if terminated or truncated:
            break
    
    mean_error = np.mean(tracking_errors)
    print(f"Steps completed: {len(tracking_errors)}")
    print(f"Mean tracking error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f} deg)")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Average reward per step: {total_reward/len(tracking_errors):.3f}")
    
    # Success check
    error_percentage = (mean_error / 0.5) * 100  # 0.5 is our target amplitude
    if error_percentage < 20:
        print("✅ Environment working well with simple controller!")
    else:
        print("⚠️ High tracking error - environment may need tuning")
    
    return env, tracking_errors

if __name__ == "__main__":
    # Test basic functionality
    env = test_environment()
    
    # Test with our working controller
    env, errors = test_with_simple_controller()
    
    print(f"\n=== Environment Ready for RL! ===")
    print("Next steps:")
    print("1. Install Stable-Baselines3: pip install stable-baselines3")
    print("2. Train SAC agent on this environment")
    print("3. Compare RL performance vs simple controller")