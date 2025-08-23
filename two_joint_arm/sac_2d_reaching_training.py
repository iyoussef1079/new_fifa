"""
Phase 2.3: SAC Training on 2D Reaching
Direct extension of your working Phase 1 SAC training
Test: Can RL learn 2D coordination that Simple Controller can't achieve?
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from two_joint_reaching_env import TwoJointReachingEnv

class TrainingMonitor(BaseCallback):
    """
    Monitor training progress - extends your Phase 1 callback pattern
    """
    def __init__(self, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        
        # Training metrics (same as Phase 1)
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_errors = []
        self.episode_efforts = []
        
        # Evaluation metrics
        self.eval_success_rates = []
        self.eval_errors = []
        self.eval_steps = []
        
    def _init_callback(self):
        # Separate evaluation environment (learned from Phase 1 Gymnasium issue)
        self.eval_env = TwoJointReachingEnv(max_episode_steps=500)
        
    def _on_step(self):
        # Collect episode data when episode ends
        if self.locals['dones'][0]:  # Episode finished
            info = self.locals['infos'][0]
            
            # Get episode stats from environment
            if hasattr(self.training_env.envs[0], 'episode_errors'):
                episode_error = np.mean(self.training_env.envs[0].episode_errors)
                episode_effort = np.mean(self.training_env.envs[0].episode_efforts) 
                
                self.episode_rewards.append(self.locals['rewards'][0])
                self.episode_errors.append(episode_error)
                self.episode_efforts.append(episode_effort)
        
        # Periodic evaluation (like Phase 1)
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
            
        return True
    
    def _evaluate_agent(self):
        """Evaluate current agent performance"""
        print(f"\n=== Evaluation at Step {self.n_calls} ===")
        
        successes = 0
        total_errors = []
        total_steps = 0
        
        for episode in range(10):  # 10 evaluation episodes
            obs, _ = self.eval_env.reset()
            episode_error = 0
            episode_steps = 0
            
            for step in range(500):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                # Track error
                current_error = np.sqrt(obs[6]**2 + obs[7]**2)
                episode_error += current_error
                episode_steps += 1
                
                if terminated:  # Success!
                    successes += 1
                    break
                    
                if truncated:  # Time limit
                    break
            
            avg_error = episode_error / episode_steps if episode_steps > 0 else 0
            total_errors.append(avg_error)
            total_steps += episode_steps
        
        # Evaluation results
        success_rate = successes / 10.0 * 100
        avg_error = np.mean(total_errors)
        avg_steps = total_steps / 10.0
        
        self.eval_success_rates.append(success_rate)
        self.eval_errors.append(avg_error)
        self.eval_steps.append(self.n_calls)
        
        print(f"Success Rate: {success_rate:.0f}% ({successes}/10)")
        print(f"Average Error: {avg_error:.4f}m")
        print(f"Average Steps: {avg_steps:.1f}")
        
        # Compare to Simple Controller (60% success, 0.30m error)
        if success_rate > 60:
            improvement = (success_rate - 60) / 60 * 100
            print(f"🚀 {improvement:.0f}% better success than Simple Controller!")
        
        if avg_error < 0.30:
            improvement = (0.30 - avg_error) / 0.30 * 100
            print(f"🎯 {improvement:.0f}% better accuracy than Simple Controller!")
    
    def plot_training_progress(self):
        """Plot training progress (extends Phase 1 plotting)"""
        if len(self.eval_steps) < 2:
            print("Not enough evaluation data to plot")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Success rate over time
        plt.subplot(1, 3, 1)
        plt.plot(self.eval_steps, self.eval_success_rates, 'b-', linewidth=2)
        plt.axhline(y=60, color='r', linestyle='--', label='Simple Controller (60%)')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Error over time  
        plt.subplot(1, 3, 2)
        plt.plot(self.eval_steps, self.eval_errors, 'g-', linewidth=2)
        plt.axhline(y=0.30, color='r', linestyle='--', label='Simple Controller (0.30m)')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Error (m)')
        plt.title('Error Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Recent episode rewards
        plt.subplot(1, 3, 3)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
            plt.plot(recent_rewards, 'purple', alpha=0.6)
            plt.xlabel('Recent Episodes')
            plt.ylabel('Episode Reward')
            plt.title('Recent Training Rewards')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sac_2d_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

def train_sac_2d_reaching():
    """
    Train SAC on 2D reaching - extends your Phase 1 training approach
    """
    print("=== Phase 2.3: SAC Training on 2D Reaching ===")
    print("Testing: Can RL learn coordination that Simple Controller can't?")
    
    # Create vectorized environment (same pattern as Phase 1)
    env = make_vec_env(
        lambda: TwoJointReachingEnv(max_episode_steps=500), 
        n_envs=1
    )
    
    # SAC hyperparameters (from your working Phase 1, extended for 2D complexity)
    model = SAC(
        "MlpPolicy", 
        env,
        learning_rate=1e-3,          # Your working value
        buffer_size=100000,          # Increased for 2D complexity  
        batch_size=256,              # Your working value
        gamma=0.95,                  # Your working value
        tau=0.01,                    # Your working value
        policy_kwargs=dict(
            net_arch=[256, 256, 256]  # Deeper network for 2D coordination
        ),
        verbose=1
    )
    
    # Training monitor (same as Phase 1)
    callback = TrainingMonitor(eval_freq=5000)
    
    print(f"\nTraining SAC for 100,000 steps (2x your Phase 1 for 2D complexity)")
    print(f"Environment: 8D observation, 4D action (4 muscles)")
    print(f"Target: Beat Simple Controller (60% success, 0.30m error)")
    
    # Train the agent
    model.learn(
        total_timesteps=100000,  # 2x Phase 1 for 2D complexity
        callback=callback,
        progress_bar=True
    )
    
    # Save trained model
    model.save("sac_2d_reaching_model")
    print("\n✅ Model saved as 'sac_2d_reaching_model'")
    
    # Final evaluation
    print("\n=== Final Performance Test ===")
    callback._evaluate_agent()
    
    # Plot results
    callback.plot_training_progress()
    
    return model, callback

def test_trained_agent(model_path="sac_2d_reaching_model"):
    """
    Test the trained SAC agent
    """
    print(f"\n=== Testing Trained SAC Agent ===")
    
    # Load trained model
    try:
        model = SAC.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    except:
        print(f"❌ Could not load model from {model_path}")
        return
    
    env = TwoJointReachingEnv()
    
    print("\nRunning 5 test episodes...")
    
    successes = 0
    total_errors = []
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_error = 0
        episode_steps = 0
        
        print(f"\nEpisode {episode+1}: Target at ({obs[4]:.3f}, {obs[5]:.3f})")
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_error = np.sqrt(obs[6]**2 + obs[7]**2)
            episode_error += current_error
            episode_steps += 1
            
            if step % 100 == 0:
                hand_x = obs[4] - obs[6]
                hand_y = obs[5] - obs[7]
                print(f"  Step {step}: Error={current_error:.4f}m, Hand=({hand_x:.3f}, {hand_y:.3f})")
            
            if terminated:
                print(f"  ✅ SUCCESS in {step+1} steps!")
                successes += 1
                break
                
            if truncated:
                print(f"  ⏰ Time limit reached")
                break
        
        avg_error = episode_error / episode_steps
        total_errors.append(avg_error)
        print(f"  Average error: {avg_error:.4f}m")
    
    # Results
    success_rate = successes / 5 * 100
    avg_error = np.mean(total_errors)
    
    print(f"\n=== SAC Results vs Simple Controller ===")
    print(f"SAC Success Rate: {success_rate:.0f}% vs Simple: 60%")
    print(f"SAC Average Error: {avg_error:.4f}m vs Simple: 0.3055m")
    
    if success_rate > 60:
        improvement = (success_rate - 60) / 60 * 100
        print(f"🚀 SAC is {improvement:.0f}% better at success rate!")
    
    if avg_error < 0.3055:
        improvement = (0.3055 - avg_error) / 0.3055 * 100
        print(f"🎯 SAC is {improvement:.0f}% more accurate!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test existing model
        test_trained_agent()
    else:
        # Train new model
        model, callback = train_sac_2d_reaching()
        
        print(f"\n🎉 Training Complete!")
        print(f"To test the trained agent, run:")
        print(f"python sac_2d_reaching_training.py test")