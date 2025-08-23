"""
Phase 2.2: 2D Reaching Environment
Extends your working ElbowTrackingEnv to 2D point-to-point reaching
Uses your validated TwoJointArm system
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from two_joint_arm import TwoJointArm, TwoJointArmParams

class TwoJointReachingEnv(gym.Env):
    """
    2D Reaching Environment - Direct extension of your ElbowTrackingEnv
    
    Task: Move hand from current position to random target positions
    Observation: [shoulder_angle, shoulder_vel, elbow_angle, elbow_vel, target_x, target_y, error_x, error_y]
    Action: [shoulder_flexor, shoulder_extensor, bicep, tricep] (0-1)
    Reward: Same structure as your working elbow system
    """
    
    def __init__(self, max_episode_steps=500, render_mode=None):
        super().__init__()
        
        # Use your validated arm system
        self.arm = TwoJointArm()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Action space: 4 muscle activations (same as your elbow pattern)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(4,), dtype=np.float32
        )
        
        # Observation space: joint states + target + error (extends your 4D elbow obs)
        # [shoulder_angle, shoulder_vel, elbow_angle, elbow_vel, target_x, target_y, error_x, error_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(8,), dtype=np.float32
        )
        
        # Target and tracking variables
        self.target_position = np.array([0.3, 0.2])  # Default target
        self.current_step = 0
        
        # Workspace limits (reachable area for targets)
        self.workspace_center = np.array([0.3, 0.0])  # In front of shoulder
        self.workspace_radius = 0.4  # Reachable distance
        
        # Episode history for analysis
        self.episode_positions = []
        self.episode_errors = []
        self.episode_efforts = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset arm to neutral position (like your elbow system)
        self.arm = TwoJointArm()
        self.arm.shoulder_angle = np.pi/6    # Start 30° up (reasonable starting pose)
        self.arm.elbow_angle = np.pi/4       # Start 45° bent (natural arm position)
        
        # Generate new random target within workspace
        self.target_position = self._generate_target()
        self.current_step = 0
        
        # Clear episode history
        self.episode_positions = []
        self.episode_errors = []
        self.episode_efforts = []
        
        return self._get_observation(), {}
    
    def _generate_target(self):
        """Generate random target within reachable workspace"""
        # Random angle and distance (ensures targets are reachable)
        angle = np.random.uniform(-np.pi/2, np.pi/2)  # -90° to +90°
        distance = np.random.uniform(0.2, self.workspace_radius)  # 0.2m to 0.4m from shoulder
        
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        
        return np.array([target_x, target_y])
    
    def _get_observation(self):
        """Build observation vector (extends your elbow obs pattern)"""
        # Get current arm state
        arm_state = self.arm.get_state()  # [shoulder_angle, shoulder_vel, elbow_angle, elbow_vel]
        hand_pos = self.arm.get_end_effector_position()
        
        # Compute error (same concept as your elbow error)
        error = self.target_position - hand_pos
        
        # Build observation: arm_state + target + error (8D total)
        obs = np.concatenate([
            arm_state,                    # 4D: joint angles and velocities
            self.target_position,         # 2D: target x,y
            error                         # 2D: error x,y
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """Environment step (extends your elbow step pattern)"""
        # Simulate arm dynamics
        state, hand_pos = self.arm.step(action)
        
        # Compute reward (same structure as your elbow reward)
        reward = self._compute_reward(action, hand_pos)
        
        # Episode management
        self.current_step += 1
        terminated = self._check_success(hand_pos)
        truncated = self.current_step >= self.max_episode_steps
        
        # Store episode data for analysis
        error = np.linalg.norm(self.target_position - hand_pos)
        effort = np.sum(action**2)
        self.episode_positions.append(hand_pos.copy())
        self.episode_errors.append(error)
        self.episode_efforts.append(effort)
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _compute_reward(self, action, hand_pos):
        """
        Reward function - Direct extension of your working elbow reward
        Your formula: -error² - 0.01×effort² + proximity_bonus
        """
        # Distance error (replaces your angle error)
        error_distance = np.linalg.norm(self.target_position - hand_pos)
        
        # Effort penalty (same as your elbow system)
        effort_penalty = 0.01 * np.sum(action**2)
        
        # Main reward components
        distance_reward = -error_distance**2  # Quadratic penalty (like your angle error²)
        
        # Success bonus (like your proximity bonus)
        success_bonus = 0.0
        if error_distance < 0.05:  # Within 5cm of target
            success_bonus = 10.0
        elif error_distance < 0.1:  # Within 10cm
            success_bonus = 2.0
        
        # Stability penalty (prevent wild movements)
        stability_penalty = 0.0
        joint_velocities = self.arm.get_state()[1::2]  # [shoulder_vel, elbow_vel]
        if np.any(np.abs(joint_velocities) > 10.0):  # rad/s limit
            stability_penalty = -5.0
        
        total_reward = distance_reward - effort_penalty + success_bonus + stability_penalty
        return total_reward
    
    def _check_success(self, hand_pos):
        """Check if target reached (task completion)"""
        error_distance = np.linalg.norm(self.target_position - hand_pos)
        return error_distance < 0.03  # Success within 3cm (tight tolerance)
    
    def render(self, mode='human'):
        """Simple 2D visualization (extends your elbow plotting)"""
        if mode != 'human':
            return
            
        plt.clf()
        
        # Draw arm segments
        shoulder_pos = np.array([0.0, 0.0])
        elbow_pos = shoulder_pos + self.arm.params.upper_arm_length * np.array([
            np.cos(self.arm.shoulder_angle), 
            np.sin(self.arm.shoulder_angle)
        ])
        hand_pos = self.arm.get_end_effector_position()
        
        # Arm visualization
        plt.plot([shoulder_pos[0], elbow_pos[0]], [shoulder_pos[1], elbow_pos[1]], 
                'b-', linewidth=4, label='Upper arm')
        plt.plot([elbow_pos[0], hand_pos[0]], [elbow_pos[1], hand_pos[1]], 
                'g-', linewidth=3, label='Forearm')
        
        # Joint markers
        plt.plot(shoulder_pos[0], shoulder_pos[1], 'ko', markersize=8, label='Shoulder')
        plt.plot(elbow_pos[0], elbow_pos[1], 'ro', markersize=6, label='Elbow')
        plt.plot(hand_pos[0], hand_pos[1], 'go', markersize=8, label='Hand')
        
        # Target and trajectory
        plt.plot(self.target_position[0], self.target_position[1], 'r*', 
                markersize=12, label='Target')
        
        if len(self.episode_positions) > 1:
            positions = np.array(self.episode_positions)
            plt.plot(positions[:, 0], positions[:, 1], 'g--', alpha=0.5, label='Trajectory')
        
        # Workspace boundary
        circle = plt.Circle(self.workspace_center, self.workspace_radius, 
                           fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        
        plt.xlim(-0.6, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'2D Reaching Task - Step {self.current_step}')
        plt.tight_layout()
        plt.pause(0.01)


def test_simple_2d_controller():
    """
    Test your Simple Controller approach extended to 2D
    Same logic as your working elbow controller
    """
    print("Testing Simple 2D Controller...")
    
    env = TwoJointReachingEnv()
    obs, _ = env.reset()
    
    def simple_2d_controller(obs, kp=8.0):
        """
        SIMPLIFIED: Direct end-effector error control (like your working elbow system)
        Much simpler than inverse kinematics - just push toward target
        """
        # Extract error components (same as your elbow error)
        error_x, error_y = obs[6:8]
        error_distance = np.sqrt(error_x**2 + error_y**2)
        
        # Simple heuristic: map 2D error to muscle activations
        activations = np.zeros(4)
        
        if error_distance > 0.01:  # Only act if significant error
            # Normalize error direction
            error_x_norm = error_x / error_distance
            error_y_norm = error_y / error_distance
            
            # Shoulder muscles (vertical movement)
            if error_y_norm > 0:  # Need to move up
                activations[0] = min(kp * abs(error_y_norm) * error_distance, 1.0)  # Shoulder flexor
            else:  # Need to move down
                activations[1] = min(kp * abs(error_y_norm) * error_distance, 1.0)  # Shoulder extensor
            
            # Elbow muscles (horizontal + fine tuning)
            if error_x_norm > 0:  # Need to reach forward
                activations[2] = min(kp * abs(error_x_norm) * error_distance, 1.0)  # Bicep (extends reach)
            else:  # Need to pull back
                activations[3] = min(kp * abs(error_x_norm) * error_distance, 1.0)  # Tricep (retracts)
        
        return activations
    
    # Test reaching performance
    episode_errors = []
    episode_successes = []
    
    for episode in range(5):  # Test 5 reaching attempts
        obs, _ = env.reset()
        total_error = 0.0
        steps = 0
        
        print(f"\nEpisode {episode + 1}: Target at ({obs[4]:.3f}, {obs[5]:.3f})")
        
        for step in range(500):
            action = simple_2d_controller(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            current_error = np.sqrt(obs[6]**2 + obs[7]**2)  # Distance error
            total_error += current_error
            steps += 1
            
            if step % 100 == 0:
                hand_x = obs[4] - obs[6]  # target_x - error_x
                hand_y = obs[5] - obs[7]  # target_y - error_y
                print(f"  Step {step}: Error={current_error:.4f}m, Hand=({hand_x:.3f}, {hand_y:.3f})")
            
            if terminated:
                print(f"  ✅ SUCCESS in {step+1} steps!")
                episode_successes.append(True)
                break
                
            if truncated:
                print(f"  ⏰ Time limit reached")
                episode_successes.append(False)
                break
        
        avg_error = total_error / steps
        episode_errors.append(avg_error)
        print(f"  Average error: {avg_error:.4f}m")
    
    # Results summary
    success_rate = sum(episode_successes) / len(episode_successes) * 100
    avg_error = np.mean(episode_errors)
    
    print(f"\n=== Simple 2D Controller Results ===")
    print(f"Success rate: {success_rate:.0f}% ({sum(episode_successes)}/{len(episode_successes)})")
    print(f"Average error: {avg_error:.4f}m")
    
    if success_rate >= 60 and avg_error < 0.1:
        print("✅ Simple 2D Controller VALIDATED: Ready for SAC training")
        return True
    else:
        print("⚠️  Simple 2D Controller needs tuning before SAC training")
        return False


if __name__ == "__main__":
    print("=== Phase 2.2: 2D Reaching Environment ===")
    print("Extending your working ElbowTrackingEnv to 2D reaching tasks")
    
    # Validate environment
    print("\n1. Testing environment creation...")
    env = TwoJointReachingEnv()
    obs, _ = env.reset()
    print(f"   Observation shape: {obs.shape} (should be 8D)")
    print(f"   Action space: {env.action_space.shape} (should be 4D)")
    print(f"   Initial target: ({obs[4]:.3f}, {obs[5]:.3f})")
    
    # Test basic step
    print("\n2. Testing environment step...")
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(random_action)
    print(f"   Step completed. Reward: {reward:.3f}")
    
    # Test controller
    print("\n3. Testing Simple 2D Controller...")
    controller_ok = test_simple_2d_controller()
    
    if controller_ok:
        print("\n🎉 PHASE 2.2 SUCCESS: Environment validated, controller working")
        print("Next: Phase 2.3 - Train SAC agent on 2D reaching")
    else:
        print("\n⚠️  Fix controller performance before SAC training")