"""
Phase 2: Train SAC Agent on Elbow Tracking Task
Single script to train RL agent and compare vs simple controller

Goal: Prove RL can learn better muscle coordination than classical control
Expected: SAC should beat the 5.96° baseline error
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import time

# Import our working environment (make sure elbow_gym_env.py is in same folder)
from elbow_gym_env import ElbowTrackingEnv


class TrainingMonitor(BaseCallback):
    """Simple callback to monitor training progress"""
    
    def __init__(self, eval_freq=1000):
        super().__init__()
        self.eval_freq = eval_freq
        self.eval_rewards = []
        self.eval_errors = []
        self.eval_env = None
        
    def _init_callback(self) -> None:
        # Create separate evaluation environment
        self.eval_env = ElbowTrackingEnv(max_episode_steps=500)
        
    def _on_step(self) -> bool:
        # Every eval_freq steps, test current policy
        if self.n_calls % self.eval_freq == 0:
            # Quick evaluation
            total_reward = 0
            total_error = 0
            n_eval_episodes = 3  # Reduced for speed
            
            for _ in range(n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                episode_errors = []
                
                for _ in range(100):  # 1 second at 100Hz
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_errors.append(info['tracking_error'])
                    
                    if terminated or truncated:
                        break
                
                total_reward += episode_reward
                total_error += np.mean(episode_errors)
            
            avg_reward = total_reward / n_eval_episodes
            avg_error = total_error / n_eval_episodes
            
            self.eval_rewards.append(avg_reward)
            self.eval_errors.append(avg_error)
            
            print(f"Step {self.n_calls}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Error = {np.degrees(avg_error):.2f}°")
        
        return True


def baseline_controller_performance(env, n_episodes=5):
    """Test our simple controller baseline"""
    print("=== Baseline Controller Performance ===")
    
    total_errors = []
    total_rewards = []
    kp = 4.0  # Same gain as before
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_errors = []
        episode_reward = 0
        
        for step in range(500):  # 5 second episode
            # Simple proportional controller
            error = obs[3]  # angle_error
            
            if error > 0:
                action = [min(kp * error, 1.0), 0.0]
            else:
                action = [0.0, min(kp * abs(error), 1.0)]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_errors.append(info['tracking_error'])
            
            if terminated or truncated:
                break
        
        total_errors.extend(episode_errors)
        total_rewards.append(episode_reward)
    
    mean_error = np.mean(total_errors)
    mean_reward = np.mean(total_rewards)
    
    print(f"Baseline Results ({n_episodes} episodes):")
    print(f"  Mean tracking error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f}°)")
    print(f"  Mean episode reward: {mean_reward:.1f}")
    print(f"  Max error: {np.degrees(np.max(total_errors)):.2f}°")
    print(f"  Min error: {np.degrees(np.min(total_errors)):.2f}°")
    
    return mean_error, mean_reward


def train_sac_agent(total_timesteps=10000):
    """Train SAC agent on elbow tracking task"""
    print("=== Training SAC Agent ===")
    
    # Create environment for training
    # Using vectorized env for better sample efficiency
    env = make_vec_env(ElbowTrackingEnv, n_envs=4, env_kwargs={'max_episode_steps': 500})
    
    # Create SAC model with optimized hyperparameters for continuous control
    model = SAC(
        "MlpPolicy", 
        env,
        learning_rate=1e-3,       # Good starting learning rate
        buffer_size=50000,        # Reasonable buffer for our simple task
        batch_size=256,           # Larger batches for stability
        gamma=0.95,               # Slightly lower for shorter episodes
        tau=0.01,                 # Soft update coefficient
        policy_kwargs=dict(net_arch=[256, 256]),  # Deep enough for muscle control
        verbose=1,                # Show training progress
        seed=42                   # Reproducible results
    )
    
    # Create callback for monitoring
    monitor = TrainingMonitor(eval_freq=2000)
    
    # Train the model
    print(f"Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=monitor,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Save the model
    model.save("sac_elbow_model")
    print("Model saved as 'sac_elbow_model.zip'")
    
    return model, monitor


def test_trained_agent(model, n_episodes=5):
    """Test the trained SAC agent"""
    print("=== Testing Trained SAC Agent ===")
    
    # Create test environment
    test_env = ElbowTrackingEnv(max_episode_steps=500)
    
    total_errors = []
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = test_env.reset()
        episode_errors = []
        episode_reward = 0
        
        for step in range(500):
            # Use trained agent to select actions
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            episode_reward += reward
            episode_errors.append(info['tracking_error'])
            
            if terminated or truncated:
                break
        
        total_errors.extend(episode_errors)
        total_rewards.append(episode_reward)
    
    mean_error = np.mean(total_errors)
    mean_reward = np.mean(total_rewards)
    
    print(f"SAC Agent Results ({n_episodes} episodes):")
    print(f"  Mean tracking error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f}°)")
    print(f"  Mean episode reward: {mean_reward:.1f}")
    print(f"  Max error: {np.degrees(np.max(total_errors)):.2f}°")
    print(f"  Min error: {np.degrees(np.min(total_errors)):.2f}°")
    
    return mean_error, mean_reward


def compare_results(baseline_error, baseline_reward, sac_error, sac_reward):
    """Compare baseline controller vs SAC agent"""
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    
    print(f"Baseline Controller:")
    print(f"  Tracking Error: {np.degrees(baseline_error):.2f}°")
    print(f"  Episode Reward: {baseline_reward:.1f}")
    
    print(f"\nSAC Agent:")
    print(f"  Tracking Error: {np.degrees(sac_error):.2f}°")
    print(f"  Episode Reward: {sac_reward:.1f}")
    
    # Calculate improvements
    error_improvement = ((baseline_error - sac_error) / baseline_error) * 100
    reward_improvement = ((sac_reward - baseline_reward) / abs(baseline_reward)) * 100
    
    print(f"\nImprovements:")
    if error_improvement > 0:
        print(f"  ✅ Tracking Error: {error_improvement:.1f}% better")
    else:
        print(f"  ❌ Tracking Error: {abs(error_improvement):.1f}% worse")
    
    if reward_improvement > 0:
        print(f"  ✅ Reward: {reward_improvement:.1f}% better")
    else:
        print(f"  ❌ Reward: {abs(reward_improvement):.1f}% worse")
    
    # Overall assessment
    if error_improvement > 10 and reward_improvement > 10:
        print("\n🎉 SUCCESS: RL significantly outperforms classical control!")
    elif error_improvement > 0 and reward_improvement > 0:
        print("\n✅ SUCCESS: RL learns better muscle coordination")
    elif error_improvement > 0:
        print("\n📈 PARTIAL SUCCESS: Better tracking but needs reward tuning")
    else:
        print("\n⚠️ NEEDS WORK: Try longer training or hyperparameter tuning")


def visualize_training_progress(monitor):
    """Plot training progress"""
    if len(monitor.eval_errors) == 0:
        print("No evaluation data to plot")
        return
        
    steps = np.array(range(len(monitor.eval_errors))) * monitor.eval_freq
    errors_deg = np.degrees(monitor.eval_errors)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, errors_deg, 'b-', linewidth=2)
    plt.axhline(y=5.96, color='r', linestyle='--', label='Baseline (5.96°)')
    plt.xlabel('Training Steps')
    plt.ylabel('Tracking Error (degrees)')
    plt.title('Learning Progress: Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, monitor.eval_rewards, 'g-', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.title('Learning Progress: Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🚀 Starting RL Training on Elbow Control Task")
    print("="*60)
    
    # Step 1: Test baseline controller
    baseline_env = ElbowTrackingEnv()
    baseline_error, baseline_reward = baseline_controller_performance(baseline_env)
    
    print("\n" + "="*60)
    
    # Step 2: Train SAC agent
    trained_model, training_monitor = train_sac_agent(total_timesteps=50000)
    
    print("\n" + "="*60)
    
    # Step 3: Test trained agent
    sac_error, sac_reward = test_trained_agent(trained_model)
    
    # Step 4: Compare results
    compare_results(baseline_error, baseline_reward, sac_error, sac_reward)
    
    # Step 5: Show training progress
    try:
        visualize_training_progress(training_monitor)
    except:
        print("\n📊 Training plots skipped (matplotlib not available)")
    
    print("\n✅ Phase 2 Complete - RL Training Done!")
    print("\nNext steps:")
    print("1. If RL wins: Try harder tasks (faster targets, disturbances)")
    print("2. If RL struggles: Tune hyperparameters or increase training time")
    print("3. Ready for Phase 3: Expand to 2D arm system!")