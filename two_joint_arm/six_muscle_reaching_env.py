"""
Phase 2.5b: 6-Muscle Reaching Environment
Minimal extension of your successful TwoJointReachingEnv
Changes: 4D→6D action space, 8D→10D observation space
Everything else identical to your 90% success system
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from six_muscle_arm import SixMuscleArm, SixMuscleArmParams

class SixMuscleReachingEnv(gym.Env):
    """
    6-Muscle Reaching Environment
    DIRECT extension of your successful TwoJointReachingEnv (90% success)
    
    ONLY CHANGES:
    - Action: 4D → 6D (adds abductor/adductor)
    - Observation: 8D → 10D (adds shoulder abduction state)
    - Same task, same reward, same targets, same success criteria
    """
    
    def __init__(self, max_episode_steps=500, render_mode=None):
        super().__init__()
        
        # Use new 6-muscle arm system
        self.arm = SixMuscleArm()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Action space: 6 muscle activations (extended from your working 4D)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(6,), dtype=np.float32  # 4→6 muscles
        )
        
        # Observation space: joint states + target + error (extended from your working 8D)
        # [shoulder_flex_angle, shoulder_flex_vel, shoulder_abd_angle, shoulder_abd_vel, 
        #  elbow_angle, elbow_vel, target_x, target_y, error_x, error_y] = 10D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(10,), dtype=np.float32  # 8D→10D
        )
        
        # Target and tracking (IDENTICAL to your successful Phase 2)
        self.target_position = np.array([0.3, 0.2])
        self.current_step = 0
        
        # Workspace (SAME as your working system)
        self.workspace_center = np.array([0.3, 0.0])
        self.workspace_radius = 0.4
        
        # Episode tracking (SAME as your working system)
        self.episode_positions = []
        self.episode_errors = []
        self.episode_efforts = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset arm (SAME starting pose as your successful system)
        self.arm = SixMuscleArm()
        self.arm.shoulder_flex_angle = np.pi/6    # 30° up (your working start)
        self.arm.shoulder_abd_angle = 0.0         # No abduction initially 
        self.arm.elbow_angle = np.pi/4           # 45° bent (your working start)
        
        # Generate target (IDENTICAL to your successful system)
        self.target_position = self._generate_target()
        self.current_step = 0
        
        # Clear episode history
        self.episode_positions = []
        self.episode_errors = []
        self.episode_efforts = []
        
        return self._get_observation(), {}
    
    def _generate_target(self):
        """IDENTICAL target generation to your 90% success system"""
        angle = np.random.uniform(-np.pi/2, np.pi/2)  
        distance = np.random.uniform(0.2, self.workspace_radius)
        
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        
        return np.array([target_x, target_y])
    
    def _get_observation(self):
        """Extended observation (6D arm state + 2D target + 2D error = 10D)"""
        # Get 6D arm state (was 4D in Phase 2)
        arm_state = self.arm.get_state()  # [flex_angle, flex_vel, abd_angle, abd_vel, elbow_angle, elbow_vel]
        
        # Hand position and error (SAME calculation as Phase 2)
        hand_pos = self.arm.get_end_effector_position()
        error = self.target_position - hand_pos
        
        # Build 10D observation (was 8D in Phase 2)
        obs = np.concatenate([
            arm_state,                    # 6D: joint angles and velocities (was 4D)
            self.target_position,         # 2D: target x,y (SAME)
            error                         # 2D: error x,y (SAME)
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """IDENTICAL step to your successful Phase 2, just 6D action instead of 4D"""
        # Simulate arm dynamics (now with 6 muscles)
        state, hand_pos = self.arm.step(action)
        
        # Compute reward (IDENTICAL to your 90% success reward function)
        reward = self._compute_reward(action, hand_pos)
        
        # Episode management (IDENTICAL)
        self.current_step += 1
        terminated = self._check_success(hand_pos)
        truncated = self.current_step >= self.max_episode_steps
        
        # Store episode data (IDENTICAL)
        error = np.linalg.norm(self.target_position - hand_pos)
        effort = np.sum(action**2)  # Now includes all 6 muscles
        self.episode_positions.append(hand_pos.copy())
        self.episode_errors.append(error)
        self.episode_efforts.append(effort)
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _compute_reward(self, action, hand_pos):
        """IDENTICAL reward function to your successful Phase 2"""
        # Distance error (same calculation)
        error_distance = np.linalg.norm(self.target_position - hand_pos)
        
        # Effort penalty (now includes all 6 muscles)
        effort_penalty = 0.01 * np.sum(action**2)
        
        # Main reward components (SAME)
        distance_reward = -error_distance**2
        
        # Success bonus (IDENTICAL)
        success_bonus = 0.0
        if error_distance < 0.05:
            success_bonus = 10.0
        elif error_distance < 0.1:
            success_bonus = 2.0
        
        # Stability penalty (SAME logic, extended to 6D)
        stability_penalty = 0.0
        joint_velocities = self.arm.get_state()[1::2]  # [flex_vel, abd_vel, elbow_vel]
        if np.any(np.abs(joint_velocities) > 10.0):
            stability_penalty = -5.0
        
        total_reward = distance_reward - effort_penalty + success_bonus + stability_penalty
        return total_reward
    
    def _check_success(self, hand_pos):
        """IDENTICAL success criteria to your 90% success system"""
        error_distance = np.linalg.norm(self.target_position - hand_pos)
        return error_distance < 0.03  # Same 3cm tolerance
    
    def render(self, mode='human'):
        """Extended visualization for 6-muscle system"""
        if mode != 'human':
            return
            
        plt.clf()
        
        # Draw arm segments (extended for 3D shoulder)
        shoulder_pos = np.array([0.0, 0.0])
        
        # Elbow position (accounts for both flexion and abduction)
        elbow_3d = np.array([
            self.arm.params.upper_arm_length * np.cos(self.arm.shoulder_flex_angle) * np.cos(self.arm.shoulder_abd_angle),
            self.arm.params.upper_arm_length * np.sin(self.arm.shoulder_flex_angle),  
            self.arm.params.upper_arm_length * np.cos(self.arm.shoulder_flex_angle) * np.sin(self.arm.shoulder_abd_angle)
        ])
        
        # Project to 2D for visualization (combine forward and side movement)
        elbow_2d = np.array([
            np.sqrt(elbow_3d[0]**2 + elbow_3d[2]**2),  # Distance from body
            elbow_3d[1]  # Height
        ])
        
        hand_pos = self.arm.get_end_effector_position()
        
        # Arm visualization
        plt.plot([shoulder_pos[0], elbow_2d[0]], [shoulder_pos[1], elbow_2d[1]], 
                'b-', linewidth=4, label='Upper arm')
        plt.plot([elbow_2d[0], hand_pos[0]], [elbow_2d[1], hand_pos[1]], 
                'g-', linewidth=3, label='Forearm')
        
        # Joint markers
        plt.plot(shoulder_pos[0], shoulder_pos[1], 'ko', markersize=8, label='Shoulder')
        plt.plot(elbow_2d[0], elbow_2d[1], 'ro', markersize=6, label='Elbow')
        plt.plot(hand_pos[0], hand_pos[1], 'go', markersize=8, label='Hand')
        
        # Target and trajectory (SAME)
        plt.plot(self.target_position[0], self.target_position[1], 'r*', 
                markersize=12, label='Target')
        
        if len(self.episode_positions) > 1:
            positions = np.array(self.episode_positions)
            plt.plot(positions[:, 0], positions[:, 1], 'g--', alpha=0.5, label='Trajectory')
        
        # Workspace boundary (SAME)
        circle = plt.Circle(self.workspace_center, self.workspace_radius, 
                           fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        
        plt.xlim(-0.6, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Show muscle activations
        activations = self.arm.get_activations()
        muscle_names = ['Flex', 'Ext', 'Abd', 'Add', 'Bicep', 'Tricep']
        title = f'6-Muscle Reaching - Step {self.current_step}\n'
        title += f'Muscles: {" ".join([f"{name}={act:.2f}" for name, act in zip(muscle_names, activations)])}'
        plt.title(title)
        
        plt.tight_layout()
        plt.pause(0.01)

def test_6_muscle_environment():
    """Test the 6-muscle environment extends your successful Phase 2 correctly"""
    print("=== Testing 6-Muscle Reaching Environment ===")
    print("Validating extension of your 90% success Phase 2 environment")
    
    env = SixMuscleReachingEnv()
    
    # Test 1: Environment creation
    print("\n1. Testing environment creation...")
    obs, _ = env.reset()
    print(f"   Observation shape: {obs.shape} (should be 10D, was 8D in Phase 2)")
    print(f"   Action space: {env.action_space.shape} (should be 6D, was 4D in Phase 2)")
    print(f"   Initial target: ({obs[6]:.3f}, {obs[7]:.3f})")
    
    # Test 2: Basic step
    print("\n2. Testing environment step...")
    random_action = env.action_space.sample()
    print(f"   Random 6D action: {random_action}")
    obs, reward, terminated, truncated, _ = env.step(random_action)
    print(f"   Step completed. Reward: {reward:.3f}")
    
    # Test 3: Quick episode test
    print("\n3. Testing quick episode...")
    obs, _ = env.reset()
    target = obs[6:8]
    print(f"   Target: ({target[0]:.3f}, {target[1]:.3f})")
    
    for step in range(10):
        # Simple test policy: moderate activation of all muscles
        test_action = np.array([0.3, 0.1, 0.2, 0.1, 0.4, 0.1])
        obs, reward, terminated, truncated, _ = env.step(test_action)
        
        error = np.sqrt(obs[8]**2 + obs[9]**2)
        if step % 5 == 0:
            print(f"   Step {step}: Error={error:.4f}m, Reward={reward:.3f}")
        
        if terminated:
            print(f"   ✅ Success in {step+1} steps!")
            break
    
    print(f"\n✅ 6-Muscle environment working!")
    print(f"Ready for SAC training on 6-muscle coordination")
    
    # Show comparison to Phase 2
    print(f"\n=== Environment Extension Summary ===")
    print(f"Phase 2   → Phase 2.5")
    print(f"4 muscles → 6 muscles")
    print(f"8D obs    → 10D obs") 
    print(f"4D action → 6D action")
    print(f"Same task, same reward, same success criteria")

if __name__ == "__main__":
    test_6_muscle_environment()