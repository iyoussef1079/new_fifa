"""
Phase 3 Week 1: SAC Training for MuJoCo Balance
Same proven SAC approach from Phase 2, now with proper MuJoCo physics
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os

from mujoco_balance_env import MuJoCoBalanceEnv

class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress
    """
    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_rewards = []
        self.eval_success_rates = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            rewards, success_rate = self.evaluate_policy()
            self.eval_rewards.append(rewards)
            self.eval_success_rates.append(success_rate)
            
            print(f"\nEvaluation at step {self.n_calls}:")
            print(f"  Mean reward: {rewards:.2f}")
            print(f"  Success rate: {success_rate:.1%}")
            
        return True
        
    def evaluate_policy(self, n_eval_episodes=10):
        """Evaluate the current policy"""
        env = MuJoCoBalanceEnv("simple_balance.xml")
        
        episode_rewards = []
        successes = 0
        
        for episode in range(n_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(500):  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if info['is_recovered']:
                        successes += 1
                    break
            
            episode_rewards.append(episode_reward)
        
        env.close()
        mean_reward = np.mean(episode_rewards)
        success_rate = successes / n_eval_episodes
        
        return mean_reward, success_rate

def train_sac_balance(total_timesteps=50000):
    """
    Train SAC agent on MuJoCo balance task
    Using same hyperparameters that worked in Phase 2
    """
    
    print("Starting SAC training on MuJoCo balance task...")
    
    # Create environment (single environment for now)
    env = MuJoCoBalanceEnv("simple_balance.xml")
    
    # SAC configuration (same as Phase 2 success)
    sac_config = {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'gamma': 0.95,
        'tau': 0.01,
        'policy_kwargs': {
            'net_arch': [256, 256, 256]
        },
        'verbose': 1
    }
    
    # Create SAC model
    model = SAC(env=env, **sac_config)
    
    # Set up logging
    log_path = "./logs/sac_mujoco_balance/"
    os.makedirs(log_path, exist_ok=True)
    model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
    
    # Create callback for evaluation
    callback = TrainingCallback(eval_freq=5000)
    
    print(f"Training for {total_timesteps} timesteps...")
    print("Training progress will be logged every 5000 steps")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model_path = "sac_mujoco_balance_model"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Plot training progress
        plot_training_progress(callback)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model_path = "sac_mujoco_balance_model_partial"
        model.save(model_path)
        print(f"Partial model saved to {model_path}")
    
    env.close()
    return model, callback

def plot_training_progress(callback):
    """Plot training progress"""
    if len(callback.eval_rewards) == 0:
        print("No evaluation data to plot")
        return
        
    steps = np.arange(1, len(callback.eval_rewards) + 1) * 5000
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(steps, callback.eval_rewards, 'b-', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Evaluation Reward')
    ax1.set_title('Training Progress: Rewards')
    ax1.grid(True, alpha=0.3)
    
    # Plot success rates
    ax2.plot(steps, [sr * 100 for sr in callback.eval_success_rates], 'g-', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Training Progress: Success Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('mujoco_training_progress.png', dpi=150)
    plt.show()

def test_trained_model(model_path="sac_mujoco_balance_model"):
    """Test the trained model"""
    print(f"\nTesting trained model: {model_path}")
    
    # Load model
    try:
        model = SAC.load(model_path)
        print("Model loaded successfully!")
    except:
        print(f"Could not load model from {model_path}")
        return
    
    # Test environment
    env = MuJoCoBalanceEnv("simple_balance.xml")
    
    # Run test episodes
    n_test_episodes = 5
    total_successes = 0
    
    for episode in range(n_test_episodes):
        print(f"\n--- Test Episode {episode + 1} ---")
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if step % 50 == 0:
                print(f"Step {step}: Angle = {info['ankle_angle_deg']:.1f}°, "
                      f"Muscles = [{action[0]:.2f}, {action[1]:.2f}]")
            
            if terminated or truncated:
                success = info['is_recovered']
                if success:
                    total_successes += 1
                print(f"Episode ended at step {step}")
                print(f"Final angle: {info['ankle_angle_deg']:.1f}°")
                print(f"Success: {success}")
                print(f"Total reward: {episode_reward:.1f}")
                break
    
    success_rate = total_successes / n_test_episodes
    print(f"\n=== FINAL RESULTS ===")
    print(f"Success rate: {success_rate:.1%} ({total_successes}/{n_test_episodes})")
    
    env.close()

if __name__ == "__main__":
    print("MuJoCo SAC Balance Training")
    print("=" * 40)
    
    # Check if model already exists
    model_path = "sac_mujoco_balance_model.zip"
    
    if os.path.exists(model_path):
        print(f"Found existing model: {model_path}")
        choice = input("Train new model (t) or test existing (e)? [t/e]: ").lower()
        
        if choice == 'e':
            test_trained_model(model_path.replace('.zip', ''))
        else:
            print("Training new model...")
            model, callback = train_sac_balance(total_timesteps=50000)
            print("\nTraining complete! Testing the trained model...")
            test_trained_model()
    else:
        print("No existing model found. Starting training...")
        model, callback = train_sac_balance(total_timesteps=50000)
        print("\nTraining complete! Testing the trained model...")
        test_trained_model()
    
    print("\nMuJoCo SAC training complete! ✓")
    print("Expected: SAC should achieve >80% success rate (like Phase 2)")