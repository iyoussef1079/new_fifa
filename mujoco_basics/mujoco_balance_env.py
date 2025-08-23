"""
Phase 3 Week 1: MuJoCo Simple Balance Environment
Proper physics foundation using MuJoCo + muscle actuators
Building toward realistic human biomechanics
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os

class MuJoCoBalanceEnv(gym.Env):
    """
    Simple balance environment using MuJoCo physics
    Single inverted pendulum with muscle actuators
    Foundation for realistic human biomechanics
    """
    
    def __init__(self, xml_file="simple_balance.xml"):
        super().__init__()
        
        # Load MuJoCo model
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"MuJoCo XML file not found: {xml_file}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_episode_steps = 500  # 5 seconds at 100Hz
        self.current_step = 0
        
        # Target and success criteria
        self.target_angle = 0.0  # Upright
        self.success_tolerance = np.radians(2)  # ±2 degrees
        self.fall_threshold = np.radians(20)   # ±20 degrees
        
        # Observation space: [ankle_angle, ankle_velocity, target_angle, angle_error]
        obs_high = np.array([
            np.pi,     # ankle angle
            10.0,      # ankle velocity 
            np.pi,     # target angle
            np.pi      # angle error
        ])
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        
        # Action space: [plantarflexor_activation, dorsiflexor_activation] ∈ [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Get joint and actuator indices
        self.ankle_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ankle")
        self.plantarflexor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "plantarflexor")
        self.dorsiflexor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "dorsiflexor")
        
        print(f"Initialized MuJoCo Balance Environment")
        print(f"- Ankle joint ID: {self.ankle_joint_id}")
        print(f"- Plantarflexor ID: {self.plantarflexor_id}")
        print(f"- Dorsiflexor ID: {self.dorsiflexor_id}")
        
    def reset(self, seed=None, **kwargs):
        """Reset environment to initial perturbed state"""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial perturbation (recovery task like Phase 2)
        initial_perturbations = [-0.17, 0.17]  # ±10 degrees
        perturbation = self.np_random.choice(initial_perturbations)
        
        # Set initial joint position
        self.data.qpos[self.ankle_joint_id] = perturbation
        self.data.qvel[self.ankle_joint_id] = 0.0
        
        # Forward dynamics to update state
        mujoco.mj_forward(self.model, self.data)
        
        # Reset episode tracking
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one simulation step"""
        # Clip and apply muscle activations
        action = np.clip(action, 0.0, 1.0)
        self.data.ctrl[self.plantarflexor_id] = action[0]
        self.data.ctrl[self.dorsiflexor_id] = action[1]
        
        # Step MuJoCo simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation and compute reward
        obs = self._get_observation()
        reward = self._compute_reward(action)
        
        # Check termination conditions
        self.current_step += 1
        terminated = self._is_fallen() or self._is_recovered()
        truncated = self.current_step >= self.max_episode_steps
        
        # Info dict
        info = {
            'ankle_angle_deg': np.degrees(self.data.qpos[self.ankle_joint_id]),
            'is_recovered': self._is_recovered(),
            'is_fallen': self._is_fallen(),
            'muscle_activations': action.copy()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation vector"""
        ankle_angle = self.data.qpos[self.ankle_joint_id]
        ankle_velocity = self.data.qvel[self.ankle_joint_id]
        angle_error = ankle_angle - self.target_angle
        
        obs = np.array([
            ankle_angle,
            ankle_velocity, 
            self.target_angle,
            angle_error
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, action):
        """Compute reward (similar to Phase 2 reaching)"""
        ankle_angle = self.data.qpos[self.ankle_joint_id]
        angle_error = abs(ankle_angle - self.target_angle)
        
        # Primary reward: minimize distance from target (like Phase 2)
        distance_reward = -10.0 * angle_error**2
        
        # Success bonus
        success_bonus = 10.0 if self._is_recovered() else 0.0
        
        # Effort penalty (encourage efficiency)
        effort_penalty = -0.1 * np.sum(action**2)
        
        # Fall penalty
        fall_penalty = -50.0 if self._is_fallen() else 0.0
        
        total_reward = distance_reward + success_bonus + effort_penalty + fall_penalty
        return total_reward
    
    def _is_recovered(self):
        """Check if successfully recovered balance"""
        ankle_angle = self.data.qpos[self.ankle_joint_id]
        return abs(ankle_angle - self.target_angle) < self.success_tolerance
    
    def _is_fallen(self):
        """Check if fallen over"""
        ankle_angle = self.data.qpos[self.ankle_joint_id]
        return abs(ankle_angle) > self.fall_threshold
    
    def render(self, mode="human"):
        """Render the environment (MuJoCo viewer)"""
        if not hasattr(self, 'viewer') or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.viewer.sync()
        
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Test the environment
if __name__ == "__main__":
    print("Testing MuJoCo Balance Environment...")
    
    # Create environment
    env = MuJoCoBalanceEnv("simple_balance.xml")
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test 1: Random policy
    print("\nTest 1: Random policy (should mostly fail)")
    obs, info = env.reset()
    total_reward = 0
    episode_length = 0
    
    for step in range(100):
        action = env.action_space.sample()  # Random muscle activations
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if step % 20 == 0:
            print(f"Step {step}: Angle = {info['ankle_angle_deg']:.1f}°, "
                  f"Reward = {reward:.2f}, Recovered = {info['is_recovered']}")
        
        if terminated or truncated:
            break
    
    print(f"Episode ended: Length = {episode_length}, Total reward = {total_reward:.1f}")
    print(f"Final state: Recovered = {info['is_recovered']}, Fallen = {info['is_fallen']}")
    
    # Test 2: Simple proportional control (should work better)
    print("\nTest 2: Simple proportional controller")
    obs, info = env.reset()
    total_reward = 0
    episode_length = 0
    
    for step in range(200):
        # Simple P controller: activate muscle opposite to lean direction
        ankle_angle = obs[0]  # First element is ankle angle
        
        if ankle_angle > 0:  # Leaning forward
            action = np.array([0.0, 0.5])  # Activate dorsiflexor
        else:  # Leaning backward  
            action = np.array([0.5, 0.0])  # Activate plantarflexor
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if step % 40 == 0:
            print(f"Step {step}: Angle = {info['ankle_angle_deg']:.1f}°, "
                  f"Reward = {reward:.2f}, Recovered = {info['is_recovered']}")
        
        if terminated or truncated:
            break
    
    print(f"Episode ended: Length = {episode_length}, Total reward = {total_reward:.1f}")
    print(f"Final state: Recovered = {info['is_recovered']}, Fallen = {info['is_fallen']}")
    
    env.close()
    print("\nMuJoCo Environment validation complete! ✓")
    print("Next: SAC training script (train_mujoco_balance.py)")