"""
Experiment 2: Frequency Response Analysis
Test SAC vs Simple Controller across different movement speeds

Tests:
- Slow (0.1 Hz): Precise, careful movements
- Medium (0.5 Hz): Our baseline performance  
- Fast (1.0 Hz): Quick, athletic movements
- Very Fast (2.0 Hz): Rapid, explosive movements

Key Questions:
- Which controller handles slow vs fast movements better?
- Does SAC maintain its smoothness advantage at all speeds?
- What's the bandwidth limit for each controller?
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time

# Import our working systems
from elbow_gym_env import ElbowTrackingEnv


class FrequencyTestEnv(ElbowTrackingEnv):
    """Modified environment for frequency response testing"""
    
    def __init__(self, target_frequency=0.5, amplitude=0.5, max_episode_steps=1000):
        super().__init__(max_episode_steps=max_episode_steps)
        self.target_frequency = target_frequency  # Hz
        self.amplitude = amplitude  # radians (about 29 degrees)
        
    def _generate_target(self):
        """Generate sine wave target at specified frequency"""
        target = self.amplitude * np.sin(2 * np.pi * self.target_frequency * self.elbow.time)
        return target


def test_frequency_response(controller_type, frequency, model=None, episode_duration=10.0):
    """
    Test controller at specific frequency
    Returns: mean_error, max_error, phase_lag, amplitude_ratio
    """
    print(f"    Testing {frequency:.1f} Hz...")
    
    # Create environment with specific frequency
    episode_steps = int(episode_duration * 100)  # 100 Hz simulation
    env = FrequencyTestEnv(
        target_frequency=frequency, 
        amplitude=0.5,  # ~29 degrees
        max_episode_steps=episode_steps
    )
    
    # Run test
    obs, info = env.reset()
    
    times = []
    angles = []
    targets = []
    tracking_errors = []
    rewards = []
    
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
        targets.append(env.target_angle)
        tracking_errors.append(info['tracking_error'])
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    times = np.array(times)
    angles = np.array(angles)
    targets = np.array(targets)
    errors = np.array(tracking_errors)
    
    # Calculate metrics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Phase lag analysis (skip first 2 seconds for settling)
    if len(times) > 200:
        settle_idx = 200
        settled_times = times[settle_idx:]
        settled_angles = angles[settle_idx:]
        settled_targets = targets[settle_idx:]
        
        # Find phase lag using cross-correlation
        phase_lag = calculate_phase_lag(settled_times, settled_angles, settled_targets, frequency)
        
        # Calculate amplitude ratio (actual/target amplitude)
        target_amplitude = np.std(settled_targets)
        actual_amplitude = np.std(settled_angles)
        amplitude_ratio = actual_amplitude / target_amplitude if target_amplitude > 0 else 0
    else:
        phase_lag = 0
        amplitude_ratio = 1.0
    
    return {
        'mean_error': mean_error,
        'max_error': max_error, 
        'phase_lag': phase_lag,
        'amplitude_ratio': amplitude_ratio,
        'times': times,
        'angles': angles,
        'targets': targets,
        'mean_reward': np.mean(rewards)
    }


def calculate_phase_lag(times, angles, targets, frequency):
    """Calculate phase lag between target and actual using peak detection"""
    try:
        # Find peaks in target signal
        target_peaks = []
        angle_peaks = []
        
        # Simple peak detection
        for i in range(1, len(targets)-1):
            if targets[i] > targets[i-1] and targets[i] > targets[i+1] and targets[i] > 0.1:
                target_peaks.append(times[i])
                
        for i in range(1, len(angles)-1):
            if angles[i] > angles[i-1] and angles[i] > angles[i+1] and angles[i] > 0.1:
                angle_peaks.append(times[i])
        
        if len(target_peaks) >= 2 and len(angle_peaks) >= 2:
            # Calculate average time difference between corresponding peaks
            min_peaks = min(len(target_peaks), len(angle_peaks))
            lag_sum = 0
            valid_lags = 0
            
            for i in range(min_peaks):
                lag = angle_peaks[i] - target_peaks[i]
                if abs(lag) < 1.0/frequency:  # Reasonable lag range
                    lag_sum += lag
                    valid_lags += 1
            
            if valid_lags > 0:
                return lag_sum / valid_lags
    except:
        pass
    
    return 0.0  # Default to no lag if analysis fails


def run_frequency_sweep(controller_type, model=None):
    """Test controller across multiple frequencies"""
    print(f"\n=== Frequency Sweep: {controller_type} Controller ===")
    
    # Test frequencies: slow to very fast
    frequencies = [0.1, 0.3, 0.5, 1.0, 2.0]  # Hz
    frequency_names = ["Slow", "Moderate", "Baseline", "Fast", "Very Fast"]
    
    results = {}
    
    for freq, name in zip(frequencies, frequency_names):
        print(f"  {name} ({freq:.1f} Hz):")
        
        result = test_frequency_response(
            controller_type, 
            frequency=freq, 
            model=model,
            episode_duration=10.0  # 10 seconds per test
        )
        
        results[freq] = result
        
        # Print key metrics
        print(f"    Mean Error: {np.degrees(result['mean_error']):.2f}°")
        print(f"    Phase Lag:  {result['phase_lag']*1000:.0f} ms")
        print(f"    Amplitude:  {result['amplitude_ratio']*100:.0f}% of target")
    
    return results


def plot_frequency_comparison(simple_results, sac_results):
    """Plot frequency response comparison"""
    frequencies = list(simple_results.keys())
    
    # Extract metrics for plotting
    simple_errors = [np.degrees(simple_results[f]['mean_error']) for f in frequencies]
    sac_errors = [np.degrees(sac_results[f]['mean_error']) for f in frequencies]
    
    simple_lags = [simple_results[f]['phase_lag']*1000 for f in frequencies]  # ms
    sac_lags = [sac_results[f]['phase_lag']*1000 for f in frequencies]  # ms
    
    simple_amps = [simple_results[f]['amplitude_ratio']*100 for f in frequencies]  # %
    sac_amps = [sac_results[f]['amplitude_ratio']*100 for f in frequencies]  # %
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Frequency Response Analysis: Simple Controller vs SAC Agent', fontsize=16)
    
    # Plot 1: Tracking Error vs Frequency
    axes[0,0].plot(frequencies, simple_errors, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[0,0].plot(frequencies, sac_errors, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Mean Tracking Error (degrees)')
    axes[0,0].set_title('Tracking Error vs Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Plot 2: Phase Lag vs Frequency
    axes[0,1].plot(frequencies, simple_lags, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[0,1].plot(frequencies, sac_lags, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Phase Lag (ms)')
    axes[0,1].set_title('Phase Lag vs Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Amplitude Ratio vs Frequency
    axes[1,0].plot(frequencies, simple_amps, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[1,0].plot(frequencies, sac_amps, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[1,0].axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Tracking')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Amplitude Ratio (%)')
    axes[1,0].set_title('Amplitude Tracking vs Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Sample trajectory at highest frequency
    high_freq = max(frequencies)
    axes[1,1].plot(sac_results[high_freq]['times'][:500], 
                   np.degrees(sac_results[high_freq]['angles'][:500]), 
                   'r-', linewidth=2, label='SAC', alpha=0.8)
    axes[1,1].plot(simple_results[high_freq]['times'][:500], 
                   np.degrees(simple_results[high_freq]['angles'][:500]), 
                   'b-', linewidth=2, label='Simple', alpha=0.8)
    axes[1,1].plot(sac_results[high_freq]['times'][:500], 
                   np.degrees(sac_results[high_freq]['targets'][:500]), 
                   'k--', linewidth=1, label='Target', alpha=0.7)
    axes[1,1].set_xlabel('Time (sec)')
    axes[1,1].set_ylabel('Angle (degrees)')
    axes[1,1].set_title(f'Sample Trajectory at {high_freq:.1f} Hz')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_bandwidth_characteristics(simple_results, sac_results):
    """Analyze bandwidth and frequency response characteristics"""
    frequencies = list(simple_results.keys())
    
    print(f"\n{'='*60}")
    print("FREQUENCY RESPONSE ANALYSIS")
    print("="*60)
    
    # Find frequency where error doubles (approximate bandwidth)
    simple_baseline_error = np.degrees(simple_results[0.5]['mean_error'])  # 0.5 Hz baseline
    sac_baseline_error = np.degrees(sac_results[0.5]['mean_error'])
    
    simple_bandwidth = None
    sac_bandwidth = None
    
    for freq in frequencies:
        simple_error = np.degrees(simple_results[freq]['mean_error'])
        sac_error = np.degrees(sac_results[freq]['mean_error'])
        
        if simple_bandwidth is None and simple_error > 2 * simple_baseline_error:
            simple_bandwidth = freq
        if sac_bandwidth is None and sac_error > 2 * sac_baseline_error:
            sac_bandwidth = freq
    
    print(f"Approximate Bandwidth (error doubles):")
    print(f"  Simple Controller: {simple_bandwidth or '>2.0'} Hz")
    print(f"  SAC Agent:        {sac_bandwidth or '>2.0'} Hz")
    
    # Best and worst performance analysis
    simple_errors = [np.degrees(simple_results[f]['mean_error']) for f in frequencies]
    sac_errors = [np.degrees(sac_results[f]['mean_error']) for f in frequencies]
    
    simple_best_freq = frequencies[np.argmin(simple_errors)]
    sac_best_freq = frequencies[np.argmin(sac_errors)]
    
    print(f"\nBest Performance Frequency:")
    print(f"  Simple Controller: {simple_best_freq:.1f} Hz ({min(simple_errors):.2f}° error)")
    print(f"  SAC Agent:        {sac_best_freq:.1f} Hz ({min(sac_errors):.2f}° error)")
    
    # Movement type recommendations
    print(f"\n🏃 Movement Type Analysis:")
    for freq, name in zip([0.1, 0.5, 1.0, 2.0], ["Precise", "Normal", "Athletic", "Explosive"]):
        if freq in frequencies:
            simple_err = np.degrees(simple_results[freq]['mean_error'])
            sac_err = np.degrees(sac_results[freq]['mean_error'])
            
            winner = "SAC" if sac_err < simple_err else "Simple"
            improvement = abs(simple_err - sac_err) / max(simple_err, sac_err) * 100
            
            print(f"  {name} movements ({freq:.1f} Hz): {winner} wins by {improvement:.1f}%")


def main():
    print("⚡ Experiment 2: Frequency Response Analysis")
    print("="*60)
    print("Testing bandwidth limits across movement speeds")
    print("From slow precision tasks to explosive athletic movements")
    
    # Load trained SAC model
    try:
        sac_model = SAC.load("sac_elbow_model")
        print("✅ Loaded trained SAC model")
    except:
        print("❌ Could not load SAC model. Please train it first with elbow_rl_training.py")
        return
    
    # Test both controllers across frequencies
    print(f"\n{'='*60}")
    simple_results = run_frequency_sweep("Simple")
    sac_results = run_frequency_sweep("SAC", sac_model)
    
    # Analyze bandwidth characteristics
    analyze_bandwidth_characteristics(simple_results, sac_results)
    
    # Create comprehensive plots
    try:
        plot_frequency_comparison(simple_results, sac_results)
        print("\n📊 Frequency response plots displayed!")
    except Exception as e:
        print(f"\n○ Plotting failed: {e}")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print("="*60)
    print("🎯 Control Bandwidth: How fast can each controller track accurately?")
    print("⚡ Phase Response: Which controller has better timing at high speeds?") 
    print("🏃 Athletic Performance: Which is better for explosive vs precise movements?")
    print("🧠 Learning Generalization: Does SAC's training transfer across frequencies?")
    
    print(f"\n✅ Experiment 2 Complete!")
    print("Ready for Experiment 3: Multi-Amplitude Tracking!")
    
    return simple_results, sac_results


def run_frequency_sweep(controller_type, model=None):
    """Run the full frequency sweep test"""
    frequencies = [0.1, 0.3, 0.5, 1.0, 2.0]  # Hz
    results = {}
    
    for freq in frequencies:
        result = test_frequency_response(controller_type, freq, model)
        results[freq] = result
        
        # Print immediate results
        print(f"      Error: {np.degrees(result['mean_error']):.2f}°, "
              f"Lag: {result['phase_lag']*1000:.0f}ms, "
              f"Amplitude: {result['amplitude_ratio']*100:.0f}%")
    
    return results


if __name__ == "__main__":
    simple_results, sac_results = main()