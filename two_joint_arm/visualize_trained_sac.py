"""
Visualize Trained SAC Agent
Watch the learned 2D reaching coordination patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from two_joint_reaching_env import TwoJointReachingEnv
import time

def visualize_sac_reaching(model_path="sac_2d_reaching_model", episodes=5):
    """
    Load trained SAC and watch it perform reaching tasks
    """
    print("=== Visualizing Trained SAC Agent ===")
    
    # Load trained model
    try:
        model = SAC.load(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    except:
        print(f"❌ Could not load {model_path}")
        print("Make sure training completed and model was saved")
        return
    
    # Create environment with rendering
    env = TwoJointReachingEnv(max_episode_steps=500, render_mode='human')
    
    # Set up matplotlib for interactive plotting
    plt.ion()
    plt.figure(figsize=(10, 8))
    
    print(f"\nWatching SAC perform {episodes} reaching tasks...")
    print("Close the plot window to end visualization")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        target_x, target_y = obs[4:6]
        
        print(f"\n--- Episode {episode+1} ---")
        print(f"Target: ({target_x:.3f}, {target_y:.3f})")
        
        episode_data = {
            'positions': [],
            'activations': [],
            'errors': [],
            'step': 0
        }
        
        for step in range(500):
            # Get SAC action
            action, _ = model.predict(obs, deterministic=True)
            
            # Store data for analysis
            hand_pos = env.arm.get_end_effector_position()
            episode_data['positions'].append(hand_pos.copy())
            episode_data['activations'].append(action.copy())
            episode_data['errors'].append(np.sqrt(obs[6]**2 + obs[7]**2))
            episode_data['step'] = step
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Render every few steps for smooth animation
            if step % 5 == 0:  # Render every 5 steps
                env.render()
                plt.pause(0.05)  # Small pause for animation
                
                # Print progress
                error = np.sqrt(obs[6]**2 + obs[7]**2)
                if step % 25 == 0:
                    print(f"  Step {step:3d}: Error={error:.4f}m, Action=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}]")
            
            if terminated:
                print(f"  ✅ SUCCESS in {step+1} steps!")
                env.render()  # Final position
                plt.pause(1.0)  # Pause to see success
                break
                
            if truncated:
                print(f"  ⏰ Time limit reached")
                break
        
        # Analyze episode patterns
        analyze_episode_patterns(episode_data, episode+1)
        
        # Wait before next episode
        print("Press Enter for next episode (or Ctrl+C to exit)...")
        try:
            input()
        except KeyboardInterrupt:
            break
    
    plt.ioff()
    plt.show()
    print("\nVisualization complete!")

def analyze_episode_patterns(episode_data, episode_num):
    """
    Analyze the coordination patterns SAC learned
    """
    positions = np.array(episode_data['positions'])
    activations = np.array(episode_data['activations'])
    errors = np.array(episode_data['errors'])
    
    if len(positions) < 2:
        return
    
    print(f"\n  === Episode {episode_num} Analysis ===")
    
    # Movement efficiency
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    straight_distance = np.linalg.norm(positions[-1] - positions[0])
    efficiency = straight_distance / total_distance if total_distance > 0 else 0
    
    print(f"  Movement efficiency: {efficiency:.3f} (1.0 = perfectly straight)")
    
    # Muscle usage patterns
    avg_activations = np.mean(activations, axis=0)
    max_activations = np.max(activations, axis=0)
    
    muscle_names = ['Shoulder Flexor', 'Shoulder Extensor', 'Bicep', 'Tricep']
    print(f"  Muscle usage (avg/max):")
    for i, name in enumerate(muscle_names):
        print(f"    {name:16s}: {avg_activations[i]:.3f} / {max_activations[i]:.3f}")
    
    # Co-activation analysis (like Phase 1)
    shoulder_coactivation = np.mean(np.minimum(activations[:, 0], activations[:, 1]))
    elbow_coactivation = np.mean(np.minimum(activations[:, 2], activations[:, 3]))
    
    print(f"  Co-activation patterns:")
    print(f"    Shoulder: {shoulder_coactivation:.3f}")
    print(f"    Elbow:    {elbow_coactivation:.3f}")
    
    # Final accuracy
    final_error = errors[-1] if len(errors) > 0 else 999
    print(f"  Final error: {final_error:.4f}m")

def compare_simple_vs_sac():
    """
    Side-by-side comparison of Simple Controller vs SAC
    """
    print("\n=== Simple Controller vs SAC Comparison ===")
    
    # Load SAC model
    try:
        sac_model = SAC.load("sac_2d_reaching_model")
    except:
        print("❌ No SAC model found. Run training first.")
        return
    
    env = TwoJointReachingEnv()
    
    def simple_2d_controller(obs, kp=8.0):
        """Simple controller from Phase 2.2"""
        error_x, error_y = obs[6:8]
        error_distance = np.sqrt(error_x**2 + error_y**2)
        activations = np.zeros(4)
        
        if error_distance > 0.01:
            error_x_norm = error_x / error_distance
            error_y_norm = error_y / error_distance
            
            if error_y_norm > 0:
                activations[0] = min(kp * abs(error_y_norm) * error_distance, 1.0)
            else:
                activations[1] = min(kp * abs(error_y_norm) * error_distance, 1.0)
                
            if error_x_norm > 0:
                activations[2] = min(kp * abs(error_x_norm) * error_distance, 1.0)
            else:
                activations[3] = min(kp * abs(error_x_norm) * error_distance, 1.0)
        
        return activations
    
    controllers = [
        ("Simple Controller", simple_2d_controller),
        ("SAC Agent", lambda obs: sac_model.predict(obs, deterministic=True)[0])
    ]
    
    for name, controller in controllers:
        print(f"\nTesting {name}...")
        
        successes = 0
        total_errors = []
        total_steps = 0
        
        for episode in range(5):
            obs, _ = env.reset()
            episode_error = 0
            episode_steps = 0
            
            for step in range(500):
                if name == "Simple Controller":
                    action = controller(obs)
                else:
                    action = controller(obs)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                
                current_error = np.sqrt(obs[6]**2 + obs[7]**2)
                episode_error += current_error
                episode_steps += 1
                
                if terminated:
                    successes += 1
                    break
                    
                if truncated:
                    break
            
            avg_error = episode_error / episode_steps
            total_errors.append(avg_error)
            total_steps += episode_steps
        
        success_rate = successes / 5 * 100
        avg_error = np.mean(total_errors)
        avg_steps = total_steps / 5
        
        print(f"  Success Rate: {success_rate:.0f}%")
        print(f"  Average Error: {avg_error:.4f}m")
        print(f"  Average Steps: {avg_steps:.1f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Compare Simple vs SAC performance
        compare_simple_vs_sac()
    else:
        # Visualize SAC reaching
        visualize_sac_reaching()
        print("\nTo compare Simple vs SAC performance, run:")
        print("python visualize_trained_sac.py compare")