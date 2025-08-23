"""
Phase 2.5c: SAC Training on 6-Muscle Coordination
Direct extension of your successful Phase 2 SAC training (90% success)
Test: Can SAC learn even more sophisticated coordination with full shoulder mobility?
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from six_muscle_reaching_env import SixMuscleReachingEnv

class SixMuscleTrainingMonitor(BaseCallback):
    """
    Training monitor - IDENTICAL to your successful Phase 2 callback
    Only difference: tracks 6-muscle patterns instead of 4-muscle
    """
    def __init__(self, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        
        # Training metrics (SAME as your successful Phase 2)
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_errors = []
        self.episode_efforts = []
        
        # Evaluation metrics
        self.eval_success_rates = []
        self.eval_errors = []
        self.eval_steps = []
        
    def _init_callback(self):
        # Separate evaluation environment (your proven approach)
        self.eval_env = SixMuscleReachingEnv(max_episode_steps=500)
        
    def _on_step(self):
        # Collect episode data (SAME logic as Phase 2)
        if self.locals['dones'][0]:  # Episode finished
            if hasattr(self.training_env.envs[0], 'episode_errors'):
                episode_error = np.mean(self.training_env.envs[0].episode_errors)
                episode_effort = np.mean(self.training_env.envs[0].episode_efforts) 
                
                self.episode_rewards.append(self.locals['rewards'][0])
                self.episode_errors.append(episode_error)
                self.episode_efforts.append(episode_effort)
        
        # Periodic evaluation (SAME as Phase 2)
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
            
        return True
    
    def _evaluate_agent(self):
        """Evaluate 6-muscle coordination performance"""
        print(f"\n=== 6-Muscle Evaluation at Step {self.n_calls} ===")
        
        successes = 0
        total_errors = []
        total_steps = 0
        muscle_usage_patterns = []
        
        for episode in range(10):  # 10 evaluation episodes
            obs, _ = self.eval_env.reset()
            episode_error = 0
            episode_steps = 0
            episode_muscle_usage = []
            
            for step in range(500):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                # Track error
                current_error = np.sqrt(obs[8]**2 + obs[9]**2)  # error_x, error_y now at indices 8,9
                episode_error += current_error
                episode_steps += 1
                episode_muscle_usage.append(action.copy())
                
                if terminated:  # Success!
                    successes += 1
                    break
                    
                if truncated:  # Time limit
                    break
            
            avg_error = episode_error / episode_steps if episode_steps > 0 else 0
            total_errors.append(avg_error)
            total_steps += episode_steps
            
            if episode_muscle_usage:
                muscle_usage_patterns.append(np.array(episode_muscle_usage))
        
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
        
        # Compare to Phase 2 results (4-muscle: 90% success, 0.18m error)
        if success_rate > 90:
            improvement = (success_rate - 90) / 90 * 100
            print(f"🚀 {improvement:.0f}% better success than 4-muscle system!")
        
        if avg_error < 0.18:
            improvement = (0.18 - avg_error) / 0.18 * 100
            print(f"🎯 {improvement:.0f}% better accuracy than 4-muscle system!")
        
        # 6-Muscle coordination analysis
        if muscle_usage_patterns:
            self._analyze_6_muscle_patterns(muscle_usage_patterns)
    
    def _analyze_6_muscle_patterns(self, muscle_usage_patterns):
        """Analyze what 6-muscle coordination patterns SAC learned"""
        print(f"\n--- 6-Muscle Coordination Analysis ---")
        
        # Combine all muscle usage data
        all_usage = np.concatenate(muscle_usage_patterns, axis=0)  # Shape: (steps, 6)
        
        # Average muscle usage
        avg_usage = np.mean(all_usage, axis=0)
        muscle_names = ['Shoulder Flex', 'Shoulder Ext', 'Shoulder Abd', 'Shoulder Add', 'Bicep', 'Tricep']
        
        print("Average muscle usage:")
        for i, (name, usage) in enumerate(zip(muscle_names, avg_usage)):
            print(f"  {name:13s}: {usage:.3f}")
        
        # Co-activation analysis (extend Phase 1 analysis to 6 muscles)
        shoulder_flex_ext_coact = np.mean(np.minimum(all_usage[:, 0], all_usage[:, 1]))  # Flex vs Ext
        shoulder_abd_add_coact = np.mean(np.minimum(all_usage[:, 2], all_usage[:, 3]))   # Abd vs Add  
        elbow_coact = np.mean(np.minimum(all_usage[:, 4], all_usage[:, 5]))              # Bicep vs Tricep
        
        print("Co-activation patterns:")
        print(f"  Shoulder Flex/Ext: {shoulder_flex_ext_coact:.3f}")
        print(f"  Shoulder Abd/Add:  {shoulder_abd_add_coact:.3f}")
        print(f"  Elbow Bicep/Tri:   {elbow_coact:.3f}")
        
        # Shoulder coordination complexity
        shoulder_muscles = all_usage[:, :4]  # First 4 muscles are shoulder muscles
        shoulder_complexity = np.mean(np.std(shoulder_muscles, axis=1))  # How varied are shoulder activations
        
        print(f"Shoulder coordination complexity: {shoulder_complexity:.3f}")
        print("  (Higher = more complex shoulder coordination patterns)")
    
    def plot_6_muscle_progress(self):
        """Plot 6-muscle training progress"""
        if len(self.eval_steps) < 2:
            print("Not enough evaluation data to plot")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Success rate comparison
        plt.subplot(1, 3, 1)
        plt.plot(self.eval_steps, self.eval_success_rates, 'b-', linewidth=2, label='6-Muscle SAC')
        plt.axhline(y=90, color='g', linestyle='--', label='4-Muscle SAC (90%)')
        plt.axhline(y=60, color='r', linestyle='--', label='Simple Controller (60%)')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate (%)')
        plt.title('6-Muscle Success Rate Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Error comparison
        plt.subplot(1, 3, 2)
        plt.plot(self.eval_steps, self.eval_errors, 'g-', linewidth=2, label='6-Muscle SAC')
        plt.axhline(y=0.18, color='g', linestyle='--', label='4-Muscle SAC (0.18m)')
        plt.axhline(y=0.30, color='r', linestyle='--', label='Simple Controller (0.30m)')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Error (m)')
        plt.title('6-Muscle Error Progress')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Recent episode rewards
        plt.subplot(1, 3, 3)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
            plt.plot(recent_rewards, 'purple', alpha=0.6)
            plt.xlabel('Recent Episodes')
            plt.ylabel('Episode Reward')
            plt.title('6-Muscle Training Rewards')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sac_6_muscle_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

def train_sac_6_muscle():
    """
    Train SAC on 6-muscle coordination
    Extended from your successful Phase 2 training
    """
    print("=== Phase 2.5c: SAC Training on 6-Muscle Coordination ===")
    print("Testing: Can SAC master full shoulder mobility better than 4-muscle system?")
    print("Baseline to beat: 4-muscle SAC (90% success, 0.18m error)")
    
    # Create vectorized environment (SAME pattern as your successful training)
    env = make_vec_env(
        lambda: SixMuscleReachingEnv(max_episode_steps=500), 
        n_envs=1
    )
    
    # SAC hyperparameters - IDENTICAL to your successful Phase 2 training
    # Only change: deeper network for 6-muscle complexity
    model = SAC(
        "MlpPolicy", 
        env,
        learning_rate=1e-3,              # Your proven value
        buffer_size=100000,              # Your proven value  
        batch_size=256,                  # Your proven value
        gamma=0.95,                      # Your proven value
        tau=0.01,                        # Your proven value
        policy_kwargs=dict(
            net_arch=[512, 512, 256]      # Deeper network for 6-muscle coordination
        ),
        verbose=1
    )
    
    # Training monitor
    callback = SixMuscleTrainingMonitor(eval_freq=5000)
    
    print(f"\nTraining SAC for 100,000 steps (same as your successful Phase 2)")
    print(f"Environment: 10D observation, 6D action (6 muscles)")
    print(f"Target: Beat 4-muscle system (90% success, 0.18m error)")
    
    # Train the agent
    model.learn(
        total_timesteps=100000,  # Same as your successful Phase 2
        callback=callback,
        progress_bar=True
    )
    
    # Save trained model
    model.save("sac_6_muscle_reaching_model")
    print("\n✅ 6-Muscle model saved as 'sac_6_muscle_reaching_model'")
    
    # Final evaluation
    print("\n=== Final 6-Muscle Performance Test ===")
    callback._evaluate_agent()
    
    # Plot results
    callback.plot_6_muscle_progress()
    
    print(f"\n=== Training Complete! ===")
    print(f"4-Muscle Baseline: 90% success, 0.18m error")
    if callback.eval_success_rates:
        final_success = callback.eval_success_rates[-1]
        final_error = callback.eval_errors[-1] 
        print(f"6-Muscle Result:   {final_success:.0f}% success, {final_error:.3f}m error")
        
        if final_success > 90:
            print(f"🚀 SUCCESS: 6-muscle system is {((final_success-90)/90*100):.0f}% better!")
        
        if final_error < 0.18:
            print(f"🎯 SUCCESS: 6-muscle system is {((0.18-final_error)/0.18*100):.0f}% more accurate!")
    
    return model, callback

def test_6_muscle_agent(model_path="sac_6_muscle_reaching_model"):
    """Test the trained 6-muscle SAC agent"""
    print(f"\n=== Testing Trained 6-Muscle SAC Agent ===")
    
    try:
        model = SAC.load(model_path)
        print(f"✅ Loaded 6-muscle model from {model_path}")
    except:
        print(f"❌ Could not load model from {model_path}")
        return
    
    env = SixMuscleReachingEnv()
    
    print("\nRunning 5 test episodes...")
    
    successes = 0
    total_errors = []
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_error = 0
        episode_steps = 0
        
        print(f"\nEpisode {episode+1}: Target at ({obs[6]:.3f}, {obs[7]:.3f})")
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_error = np.sqrt(obs[8]**2 + obs[9]**2)  # 6-muscle obs indices
            episode_error += current_error
            episode_steps += 1
            
            if step % 100 == 0:
                hand_x = obs[6] - obs[8]  # target_x - error_x
                hand_y = obs[7] - obs[9]  # target_y - error_y
                print(f"  Step {step}: Error={current_error:.4f}m, Hand=({hand_x:.3f}, {hand_y:.3f})")
                print(f"            6-Muscle activations: {action}")
            
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
    
    # Results comparison
    success_rate = successes / 5 * 100
    avg_error = np.mean(total_errors)
    
    print(f"\n=== Performance Comparison ===")
    print(f"4-Muscle SAC:   90% success, 0.180m error, ~119 steps")
    print(f"6-Muscle SAC:   {success_rate:.0f}% success, {avg_error:.3f}m error")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test existing 6-muscle model
        test_6_muscle_agent()
    else:
        # Train new 6-muscle model
        model, callback = train_sac_6_muscle()
        
        print(f"\n🎉 6-Muscle Training Complete!")
        print(f"To test the trained agent, run:")
        print(f"python sac_6_muscle_training.py test")