"""
Experiment 4: Random Walk Adaptability Test
Test SAC vs Simple Controller on unpredictable target changes

This is where SAC should finally show its learning advantage!

Random Walk Types:
- Slow Random Walk: Changes every 0.5 seconds (predictable timing)
- Fast Random Walk: Changes every 0.1 seconds (rapid adaptation needed)
- Variable Walk: Random timing + random magnitude (chaos test)
- Noisy Sine: Sine wave + random noise (pattern + disturbance)

Key Questions:
- Can SAC's neural network adapt better to unpredictable targets?
- Which controller handles sudden direction changes better?  
- Does learning give advantages over reactive control?
- How quickly can each controller adapt to new patterns?
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time

# Import our working systems
from elbow_gym_env import ElbowTrackingEnv


class RandomWalkEnv(ElbowTrackingEnv):
    """Environment with unpredictable random walk targets"""
    
    def __init__(self, walk_type="slow", max_episode_steps=1000):
        super().__init__(max_episode_steps=max_episode_steps)
        self.walk_type = walk_type
        
        # Random walk parameters
        self.current_target = 0.0
        self.steps_since_change = 0
        self.change_interval = 50  # Default: change every 0.5 seconds
        
        # Set parameters based on walk type
        if walk_type == "slow":
            self.change_interval = 50    # 0.5 seconds at 100Hz
            self.step_size = 0.3         # Max change per step (radians ~17°)
        elif walk_type == "fast":
            self.change_interval = 10    # 0.1 seconds at 100Hz  
            self.step_size = 0.2         # Smaller steps for faster changes
        elif walk_type == "variable":
            self.change_interval = 30    # Will be randomized
            self.step_size = 0.4         # Larger possible steps
        elif walk_type == "noisy_sine":
            self.sine_frequency = 0.5    # Base sine wave
            self.noise_level = 0.1       # Noise amplitude
        
        # Limits for random walk
        self.min_target = -1.5  # ~-85°
        self.max_target = 1.5   # ~+85°
        
    def _generate_target(self):
        """Generate random walk target based on type"""
        
        if self.walk_type == "noisy_sine":
            # Sine wave with random noise
            clean_sine = 0.5 * np.sin(2 * np.pi * self.sine_frequency * self.elbow.time)
            noise = self.noise_level * np.random.normal()
            return clean_sine + noise
        
        # Random walk logic
        self.steps_since_change += 1
        
        # Check if time to change target
        change_now = False
        if self.walk_type == "variable":
            # Variable timing: random interval between 20-80 steps (0.2-0.8 sec)
            if self.steps_since_change >= np.random.randint(20, 81):
                change_now = True
        else:
            # Fixed timing
            if self.steps_since_change >= self.change_interval:
                change_now = True
        
        if change_now:
            # Generate new random target
            direction = np.random.choice([-1, 1])
            magnitude = np.random.uniform(0.5, 1.0) * self.step_size
            
            new_target = self.current_target + direction * magnitude
            
            # Clamp to limits
            new_target = np.clip(new_target, self.min_target, self.max_target)
            
            # If we hit a limit, reverse direction
            if new_target == self.min_target or new_target == self.max_target:
                new_target = self.current_target - direction * magnitude * 0.5
                new_target = np.clip(new_target, self.min_target, self.max_target)
            
            self.current_target = new_target
            self.steps_since_change = 0
        
        return self.current_target


def test_adaptability(controller_type, walk_type, model=None, episode_duration=15.0):
    """
    Test controller adaptability on random walk targets
    Returns: adaptation metrics
    """
    print(f"    Testing {walk_type} random walk...")
    
    # Create environment with specific walk type
    episode_steps = int(episode_duration * 100)  # 100 Hz simulation
    env = RandomWalkEnv(walk_type=walk_type, max_episode_steps=episode_steps)
    
    # Run test
    obs, info = env.reset()
    
    times = []
    angles = []
    targets = []
    tracking_errors = []
    target_changes = []
    adaptation_times = []
    rewards = []
    
    prev_target = obs[2]  # Initial target
    
    for step in range(episode_steps):
        if controller_type == "Simple":
            # Simple proportional controller
            error = obs[3]
            kp = 4.0
            if error > 0:
                action = [min(kp * error, 1.0), 0.0]
            else:
                action = [0.0, min(kp * abs(error), 1.0)]
                
        elif controller_type == "SAC":
            # Trained SAC agent
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record data
        times.append(env.elbow.time)
        angles.append(env.elbow.angle)
        targets.append(obs[2])  # Current target from observation
        tracking_errors.append(info['tracking_error'])
        rewards.append(reward)
        
        # Detect target changes for adaptation analysis
        current_target = obs[2]
        if abs(current_target - prev_target) > 0.05:  # Significant change
            target_changes.append(step)
            prev_target = current_target
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    times = np.array(times)
    angles = np.array(angles)
    targets = np.array(targets)
    errors = np.array(tracking_errors)
    
    # Calculate adaptation metrics
    adaptation_performance = analyze_adaptation_speed(
        times, angles, targets, target_changes, tracking_errors
    )
    
    # Overall performance metrics  
    mean_error = np.mean(errors)
    error_std = np.std(errors)
    mean_reward = np.mean(rewards)
    
    # Target change frequency
    n_changes = len(target_changes)
    change_frequency = n_changes / (episode_duration) if episode_duration > 0 else 0
    
    return {
        'mean_error': mean_error,
        'error_std': error_std,
        'mean_reward': mean_reward,
        'adaptation_speed': adaptation_performance['mean_adaptation_time'],
        'adaptation_accuracy': adaptation_performance['mean_adaptation_accuracy'],
        'change_frequency': change_frequency,
        'n_changes': n_changes,
        'times': times,
        'angles': angles,
        'targets': targets,
        'target_changes': target_changes
    }


def analyze_adaptation_speed(times, angles, targets, target_changes, tracking_errors):
    """Analyze how quickly controller adapts to target changes"""
    adaptation_times = []
    adaptation_accuracies = []
    
    for change_idx in target_changes:
        if change_idx + 50 < len(tracking_errors):  # Need at least 0.5 sec after change
            
            # Look at error for 2 seconds after change
            post_change_errors = tracking_errors[change_idx:change_idx+200]
            
            # Find time to reach 90% improvement (adaptation time)
            initial_error = post_change_errors[0]
            target_error = np.min(post_change_errors[20:])  # Best achieved after 0.2 sec
            threshold_error = initial_error - 0.9 * (initial_error - target_error)
            
            adaptation_time = None
            for i, error in enumerate(post_change_errors):
                if error <= threshold_error:
                    adaptation_time = i * 0.01  # Convert to seconds
                    break
            
            if adaptation_time is not None:
                adaptation_times.append(adaptation_time)
                
                # Calculate adaptation accuracy (error reduction achieved)
                final_error = np.mean(post_change_errors[-20:])  # Last 0.2 seconds
                adaptation_accuracy = max(0, 1 - final_error / initial_error)
                adaptation_accuracies.append(adaptation_accuracy)
    
    return {
        'mean_adaptation_time': np.mean(adaptation_times) if adaptation_times else float('inf'),
        'mean_adaptation_accuracy': np.mean(adaptation_accuracies) if adaptation_accuracies else 0.0,
        'adaptation_times': adaptation_times,
        'adaptation_accuracies': adaptation_accuracies
    }


def run_adaptability_sweep(controller_type, model=None):
    """Test controller across different random walk types"""
    print(f"\n=== Adaptability Sweep: {controller_type} Controller ===")
    
    walk_types = ["slow", "fast", "variable", "noisy_sine"]
    walk_names = ["Slow Walk", "Fast Walk", "Variable Walk", "Noisy Sine"]
    
    results = {}
    
    for walk_type, name in zip(walk_types, walk_names):
        print(f"  {name}:")
        
        result = test_adaptability(
            controller_type, 
            walk_type=walk_type,
            model=model,
            episode_duration=15.0  # 15 seconds per test
        )
        
        results[walk_type] = result
        
        # Print key metrics
        print(f"      Mean Error:     {np.degrees(result['mean_error']):.2f}°")
        print(f"      Error Std:      {np.degrees(result['error_std']):.2f}°")
        print(f"      Adaptation:     {result['adaptation_speed']:.3f}s")
        print(f"      Changes/sec:    {result['change_frequency']:.1f}")
        print(f"      Mean Reward:    {result['mean_reward']:.1f}")
    
    return results


def plot_adaptability_analysis(simple_results, sac_results):
    """Plot adaptability comparison"""
    walk_types = list(simple_results.keys())
    walk_names = ["Slow Walk", "Fast Walk", "Variable Walk", "Noisy Sine"]
    
    # Extract metrics for plotting
    simple_errors = [np.degrees(simple_results[w]['mean_error']) for w in walk_types]
    sac_errors = [np.degrees(sac_results[w]['mean_error']) for w in walk_types]
    
    simple_adapt_times = [simple_results[w]['adaptation_speed'] for w in walk_types]
    sac_adapt_times = [sac_results[w]['adaptation_speed'] for w in walk_types]
    
    simple_error_stds = [np.degrees(simple_results[w]['error_std']) for w in walk_types]
    sac_error_stds = [np.degrees(sac_results[w]['error_std']) for w in walk_types]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Adaptability Analysis: Simple Controller vs SAC Agent', fontsize=16)
    
    x_pos = np.arange(len(walk_names))
    width = 0.35
    
    # Plot 1: Mean Tracking Error
    axes[0,0].bar(x_pos - width/2, simple_errors, width, label='Simple Controller', alpha=0.8, color='blue')
    axes[0,0].bar(x_pos + width/2, sac_errors, width, label='SAC Agent', alpha=0.8, color='red')
    axes[0,0].set_xlabel('Random Walk Type')
    axes[0,0].set_ylabel('Mean Tracking Error (degrees)')
    axes[0,0].set_title('Tracking Error Across Walk Types')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(walk_names, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Plot 2: Adaptation Speed
    # Filter out infinite values for plotting
    simple_adapt_plot = [t if t != float('inf') else 5.0 for t in simple_adapt_times]
    sac_adapt_plot = [t if t != float('inf') else 5.0 for t in sac_adapt_times]
    
    axes[0,1].bar(x_pos - width/2, simple_adapt_plot, width, label='Simple Controller', alpha=0.8, color='blue')
    axes[0,1].bar(x_pos + width/2, sac_adapt_plot, width, label='SAC Agent', alpha=0.8, color='red')
    axes[0,1].set_xlabel('Random Walk Type')
    axes[0,1].set_ylabel('Adaptation Time (seconds)')
    axes[0,1].set_title('Adaptation Speed')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(walk_names, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Error Variability (Consistency)
    axes[1,0].bar(x_pos - width/2, simple_error_stds, width, label='Simple Controller', alpha=0.8, color='blue')
    axes[1,0].bar(x_pos + width/2, sac_error_stds, width, label='SAC Agent', alpha=0.8, color='red')
    axes[1,0].set_xlabel('Random Walk Type')
    axes[1,0].set_ylabel('Error Std Dev (degrees)')
    axes[1,0].set_title('Error Consistency (Lower = More Consistent)')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(walk_names, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Sample trajectory for most challenging walk type
    most_challenging = "variable"  # Variable timing + magnitude
    if most_challenging in simple_results:
        # Plot first 8 seconds
        end_idx = min(800, len(simple_results[most_challenging]['times']))
        
        times = simple_results[most_challenging]['times'][:end_idx]
        simple_angles = np.degrees(simple_results[most_challenging]['angles'][:end_idx])
        sac_angles = np.degrees(sac_results[most_challenging]['angles'][:end_idx])
        targets = np.degrees(simple_results[most_challenging]['targets'][:end_idx])
        
        axes[1,1].plot(times, targets, 'k-', linewidth=3, label='Random Target', alpha=0.9)
        axes[1,1].plot(times, simple_angles, 'b-', linewidth=2, label='Simple', alpha=0.8)
        axes[1,1].plot(times, sac_angles, 'r-', linewidth=2, label='SAC', alpha=0.8)
        
        # Mark target changes
        changes = simple_results[most_challenging]['target_changes']
        for change in changes:
            if change < end_idx:
                axes[1,1].axvline(x=times[change], color='gray', linestyle=':', alpha=0.5)
        
        axes[1,1].set_xlabel('Time (sec)')
        axes[1,1].set_ylabel('Angle (degrees)')
        axes[1,1].set_title('Variable Random Walk Sample')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_adaptability_characteristics(simple_results, sac_results):
    """Analyze adaptability and learning characteristics"""
    print(f"\n{'='*60}")
    print("ADAPTABILITY ANALYSIS")
    print("="*60)
    
    walk_types = ["slow", "fast", "variable", "noisy_sine"]
    walk_names = ["Slow Walk", "Fast Walk", "Variable Walk", "Noisy Sine"]
    
    # Compare each walk type
    sac_wins = 0
    simple_wins = 0
    
    for walk_type, name in zip(walk_types, walk_names):
        simple_error = np.degrees(simple_results[walk_type]['mean_error'])
        sac_error = np.degrees(sac_results[walk_type]['mean_error'])
        
        simple_adapt = simple_results[walk_type]['adaptation_speed']
        sac_adapt = sac_results[walk_type]['adaptation_speed']
        
        # Determine winner for this walk type
        if sac_error < simple_error:
            winner = "SAC"
            improvement = ((simple_error - sac_error) / simple_error) * 100
            sac_wins += 1
        else:
            winner = "Simple"
            improvement = ((sac_error - simple_error) / sac_error) * 100
            simple_wins += 1
        
        print(f"{name:12}: {winner:6} wins by {improvement:4.1f}% (errors: {simple_error:5.2f}° vs {sac_error:5.2f}°)")
        
        # Adaptation speed comparison
        if simple_adapt != float('inf') and sac_adapt != float('inf'):
            faster = "SAC" if sac_adapt < simple_adapt else "Simple"
            speed_diff = abs(simple_adapt - sac_adapt)
            print(f"             {faster:6} adapts {speed_diff:.3f}s faster")
    
    print(f"\n🏆 Overall Adaptability Winner:")
    if sac_wins > simple_wins:
        print(f"  SAC Agent wins {sac_wins}/{len(walk_types)} scenarios")
        print("  ✅ Neural networks show advantage in unpredictable environments!")
    elif simple_wins > sac_wins:
        print(f"  Simple Controller wins {simple_wins}/{len(walk_types)} scenarios") 
        print("  ⚡ Classical control dominates even unpredictable tasks!")
    else:
        print("  🤝 Tie! Both controllers have different strengths")
    
    # Consistency analysis
    print(f"\n📊 Consistency Analysis:")
    for walk_type, name in zip(walk_types, walk_names):
        simple_std = np.degrees(simple_results[walk_type]['error_std'])
        sac_std = np.degrees(sac_results[walk_type]['error_std'])
        
        more_consistent = "SAC" if sac_std < simple_std else "Simple"
        consistency_diff = abs(simple_std - sac_std) / max(simple_std, sac_std) * 100
        
        print(f"  {name:12}: {more_consistent} is {consistency_diff:.1f}% more consistent")


def main():
    print("🧠 Experiment 4: Random Walk Adaptability Test")
    print("="*60)
    print("Testing adaptability to unpredictable target changes")
    print("This is where SAC should finally show its learning advantage!")
    
    # Load trained SAC model
    try:
        sac_model = SAC.load("sac_elbow_model")
        print("✅ Loaded trained SAC model")
    except:
        print("❌ Could not load SAC model. Please train it first with elbow_rl_training.py")
        return
    
    # Test both controllers across random walk types
    print(f"\n{'='*60}")
    simple_results = run_adaptability_sweep("Simple")
    sac_results = run_adaptability_sweep("SAC", sac_model)
    
    # Analyze adaptability characteristics
    analyze_adaptability_characteristics(simple_results, sac_results)
    
    # Create comprehensive plots
    try:
        plot_adaptability_analysis(simple_results, sac_results)
        print("\n📊 Adaptability analysis plots displayed!")
    except Exception as e:
        print(f"\n○ Plotting failed: {e}")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print("="*60)
    print("🧠 Learning vs Reactive: Does neural adaptation beat classical control?")
    print("⚡ Adaptation Speed: How quickly can each controller handle changes?") 
    print("📊 Consistency: Which maintains stable performance during chaos?")
    print("🎯 Pattern Recognition: Can SAC learn to predict random patterns?")
    print("🎮 Game AI Implications: Which approach for dynamic sports environments?")
    
    print(f"\n✅ Experiment 4 Complete!")
    print("This completes our 🎯 Tracking Task Experiments!")
    print("Ready for 🔬 Robustness Experiments (External Disturbances)!")
    
    return simple_results, sac_results


if __name__ == "__main__":
    simple_results, sac_results = main()