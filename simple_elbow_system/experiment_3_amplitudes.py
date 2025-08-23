"""
Experiment 3: Multi-Amplitude Analysis
Test SAC vs Simple Controller across different movement scales

Tests:
- Micro (5°): Fine motor control - precise adjustments
- Small (15°): Small movements - careful positioning  
- Medium (30°): Our baseline performance
- Large (60°): Gross motor movements - athletic actions
- XLarge (90°): Extreme range - maximum capability

Key Questions:
- Which controller excels at fine vs gross motor control?
- Does control quality scale linearly with amplitude?
- What are the practical limits for each approach?
- How does this inform sports game movement design?
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time

# Import our working systems
from elbow_gym_env import ElbowTrackingEnv


class AmplitudeTestEnv(ElbowTrackingEnv):
    """Modified environment for amplitude testing"""
    
    def __init__(self, amplitude_degrees=30, frequency=0.5, max_episode_steps=1000):
        super().__init__(max_episode_steps=max_episode_steps)
        self.amplitude = np.radians(amplitude_degrees)  # Convert degrees to radians
        self.target_frequency = frequency  # Keep frequency constant at 0.5 Hz
        
    def _generate_target(self):
        """Generate sine wave target at specified amplitude"""
        target = self.amplitude * np.sin(2 * np.pi * self.target_frequency * self.elbow.time)
        return target


def test_amplitude_response(controller_type, amplitude_deg, model=None, episode_duration=8.0):
    """
    Test controller at specific amplitude
    Returns: tracking metrics, linearity metrics, control effort
    """
    print(f"    Testing {amplitude_deg:.0f}°...")
    
    # Create environment with specific amplitude
    episode_steps = int(episode_duration * 100)  # 100 Hz simulation
    env = AmplitudeTestEnv(
        amplitude_degrees=amplitude_deg,
        frequency=0.5,  # Keep frequency constant
        max_episode_steps=episode_steps
    )
    
    # Run test
    obs, info = env.reset()
    
    times = []
    angles = []
    targets = []
    tracking_errors = []
    actions_bicep = []
    actions_tricep = []
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
        actions_bicep.append(action[0])
        actions_tricep.append(action[1])
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    times = np.array(times)
    angles = np.array(angles)
    targets = np.array(targets)
    errors = np.array(tracking_errors)
    bicep_actions = np.array(actions_bicep)
    tricep_actions = np.array(actions_tricep)
    
    # Skip first 2 seconds for settling
    settle_idx = 200 if len(times) > 200 else 0
    settled_errors = errors[settle_idx:]
    settled_angles = angles[settle_idx:]
    settled_targets = targets[settle_idx:]
    settled_bicep = bicep_actions[settle_idx:]
    settled_tricep = tricep_actions[settle_idx:]
    
    # Calculate metrics
    mean_error = np.mean(settled_errors)
    max_error = np.max(settled_errors)
    error_std = np.std(settled_errors)
    
    # Control effort (muscle usage)
    mean_bicep_effort = np.mean(settled_bicep)
    mean_tricep_effort = np.mean(settled_tricep)
    total_effort = mean_bicep_effort + mean_tricep_effort
    
    # Linearity metrics
    target_amplitude = np.std(settled_targets)
    actual_amplitude = np.std(settled_angles)
    amplitude_ratio = actual_amplitude / target_amplitude if target_amplitude > 0 else 0
    
    # Peak tracking accuracy
    target_peaks = find_peaks(settled_targets)
    angle_peaks = find_peaks(settled_angles)
    peak_accuracy = calculate_peak_accuracy(target_peaks, angle_peaks, settled_targets, settled_angles)
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'error_std': error_std,
        'amplitude_ratio': amplitude_ratio,
        'total_effort': total_effort,
        'peak_accuracy': peak_accuracy,
        'mean_reward': np.mean(rewards),
        'times': times,
        'angles': angles,
        'targets': targets,
        'bicep_actions': bicep_actions,
        'tricep_actions': tricep_actions
    }


def find_peaks(signal, min_prominence=0.1):
    """Simple peak detection"""
    peaks = []
    for i in range(1, len(signal)-1):
        if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and 
            abs(signal[i]) > min_prominence):
            peaks.append(i)
    return peaks


def calculate_peak_accuracy(target_peaks, angle_peaks, targets, angles):
    """Calculate how accurately peaks are tracked"""
    if len(target_peaks) == 0 or len(angle_peaks) == 0:
        return 0.0
    
    # Match peaks and calculate accuracy
    accuracies = []
    for t_peak in target_peaks:
        # Find closest angle peak
        if len(angle_peaks) > 0:
            closest_a_peak = min(angle_peaks, key=lambda x: abs(x - t_peak))
            target_val = targets[t_peak]
            actual_val = angles[closest_a_peak]
            accuracy = 1.0 - abs(target_val - actual_val) / abs(target_val) if target_val != 0 else 1.0
            accuracies.append(max(0, accuracy))
    
    return np.mean(accuracies) if accuracies else 0.0


def run_amplitude_sweep(controller_type, model=None):
    """Test controller across multiple amplitudes"""
    print(f"\n=== Amplitude Sweep: {controller_type} Controller ===")
    
    # Test amplitudes: micro to extreme
    amplitudes = [5, 15, 30, 60, 90]  # degrees
    amplitude_names = ["Micro", "Small", "Medium", "Large", "XLarge"]
    
    results = {}
    
    for amp, name in zip(amplitudes, amplitude_names):
        print(f"  {name} ({amp}°):")
        
        result = test_amplitude_response(
            controller_type, 
            amplitude_deg=amp,
            model=model,
            episode_duration=8.0  # 8 seconds per test (4 full cycles at 0.5 Hz)
        )
        
        results[amp] = result
        
        # Print key metrics
        print(f"      Mean Error: {np.degrees(result['mean_error']):.2f}°")
        print(f"      Amplitude:  {result['amplitude_ratio']*100:.0f}% of target")
        print(f"      Effort:     {result['total_effort']:.3f}")
        print(f"      Peak Acc:   {result['peak_accuracy']*100:.0f}%")
    
    return results


def plot_amplitude_analysis(simple_results, sac_results):
    """Plot amplitude response comparison"""
    amplitudes = list(simple_results.keys())
    
    # Extract metrics for plotting
    simple_errors = [np.degrees(simple_results[a]['mean_error']) for a in amplitudes]
    sac_errors = [np.degrees(sac_results[a]['mean_error']) for a in amplitudes]
    
    simple_efforts = [simple_results[a]['total_effort'] for a in amplitudes]
    sac_efforts = [sac_results[a]['total_effort'] for a in amplitudes]
    
    simple_amps = [simple_results[a]['amplitude_ratio']*100 for a in amplitudes]
    sac_amps = [sac_results[a]['amplitude_ratio']*100 for a in amplitudes]
    
    simple_peaks = [simple_results[a]['peak_accuracy']*100 for a in amplitudes]
    sac_peaks = [sac_results[a]['peak_accuracy']*100 for a in amplitudes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Amplitude Analysis: Simple Controller vs SAC Agent', fontsize=16)
    
    # Plot 1: Tracking Error vs Amplitude
    axes[0,0].plot(amplitudes, simple_errors, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[0,0].plot(amplitudes, sac_errors, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[0,0].set_xlabel('Target Amplitude (degrees)')
    axes[0,0].set_ylabel('Mean Tracking Error (degrees)')
    axes[0,0].set_title('Tracking Error vs Movement Amplitude')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Plot 2: Control Effort vs Amplitude
    axes[0,1].plot(amplitudes, simple_efforts, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[0,1].plot(amplitudes, sac_efforts, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[0,1].set_xlabel('Target Amplitude (degrees)')
    axes[0,1].set_ylabel('Total Muscle Effort')
    axes[0,1].set_title('Control Effort vs Movement Amplitude')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Amplitude Tracking Accuracy
    axes[1,0].plot(amplitudes, simple_amps, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[1,0].plot(amplitudes, sac_amps, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[1,0].axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Tracking')
    axes[1,0].set_xlabel('Target Amplitude (degrees)')
    axes[1,0].set_ylabel('Amplitude Ratio (%)')
    axes[1,0].set_title('Amplitude Tracking Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Peak Tracking Accuracy
    axes[1,1].plot(amplitudes, simple_peaks, 'bo-', linewidth=2, markersize=8, label='Simple Controller')
    axes[1,1].plot(amplitudes, sac_peaks, 'ro-', linewidth=2, markersize=8, label='SAC Agent')
    axes[1,1].set_xlabel('Target Amplitude (degrees)')
    axes[1,1].set_ylabel('Peak Accuracy (%)')
    axes[1,1].set_title('Peak Tracking Accuracy')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_motor_control_characteristics(simple_results, sac_results):
    """Analyze fine vs gross motor control characteristics"""
    amplitudes = list(simple_results.keys())
    
    print(f"\n{'='*60}")
    print("MOTOR CONTROL ANALYSIS")
    print("="*60)
    
    # Fine motor control analysis (small amplitudes)
    fine_motor_amps = [a for a in amplitudes if a <= 15]
    gross_motor_amps = [a for a in amplitudes if a >= 60]
    
    if fine_motor_amps:
        print("🎯 Fine Motor Control (≤15°):")
        fine_simple_errors = [np.degrees(simple_results[a]['mean_error']) for a in fine_motor_amps]
        fine_sac_errors = [np.degrees(sac_results[a]['mean_error']) for a in fine_motor_amps]
        
        avg_simple_fine = np.mean(fine_simple_errors)
        avg_sac_fine = np.mean(fine_sac_errors)
        
        print(f"  Simple Controller: {avg_simple_fine:.2f}° average error")
        print(f"  SAC Agent:        {avg_sac_fine:.2f}° average error")
        
        if avg_simple_fine < avg_sac_fine:
            improvement = ((avg_sac_fine - avg_simple_fine) / avg_sac_fine) * 100
            print(f"  ✅ Simple wins fine motor by {improvement:.1f}%")
        else:
            improvement = ((avg_simple_fine - avg_sac_fine) / avg_simple_fine) * 100
            print(f"  ✅ SAC wins fine motor by {improvement:.1f}%")
    
    if gross_motor_amps:
        print(f"\n💪 Gross Motor Control (≥60°):")
        gross_simple_errors = [np.degrees(simple_results[a]['mean_error']) for a in gross_motor_amps]
        gross_sac_errors = [np.degrees(sac_results[a]['mean_error']) for a in gross_motor_amps]
        
        avg_simple_gross = np.mean(gross_simple_errors)
        avg_sac_gross = np.mean(gross_sac_errors)
        
        print(f"  Simple Controller: {avg_simple_gross:.2f}° average error")
        print(f"  SAC Agent:        {avg_sac_gross:.2f}° average error")
        
        if avg_simple_gross < avg_sac_gross:
            improvement = ((avg_sac_gross - avg_simple_gross) / avg_sac_gross) * 100
            print(f"  ✅ Simple wins gross motor by {improvement:.1f}%")
        else:
            improvement = ((avg_simple_gross - avg_sac_gross) / avg_simple_gross) * 100
            print(f"  ✅ SAC wins gross motor by {improvement:.1f}%")
    
    # Linearity analysis
    print(f"\n📏 Linearity Analysis:")
    simple_errors = [np.degrees(simple_results[a]['mean_error']) for a in amplitudes]
    sac_errors = [np.degrees(sac_results[a]['mean_error']) for a in amplitudes]
    
    # Calculate correlation between amplitude and error
    simple_correlation = np.corrcoef(amplitudes, simple_errors)[0,1]
    sac_correlation = np.corrcoef(amplitudes, sac_errors)[0,1]
    
    print(f"  Simple Controller: {simple_correlation:.3f} correlation (error vs amplitude)")
    print(f"  SAC Agent:        {sac_correlation:.3f} correlation (error vs amplitude)")
    
    if simple_correlation < sac_correlation:
        print(f"  ✅ Simple has more linear scaling")
    else:
        print(f"  ✅ SAC has more linear scaling")
    
    # Efficiency analysis
    print(f"\n⚡ Efficiency Analysis:")
    for amp in amplitudes:
        simple_efficiency = amp / np.degrees(simple_results[amp]['mean_error'])
        sac_efficiency = amp / np.degrees(sac_results[amp]['mean_error'])
        
        winner = "SAC" if sac_efficiency > simple_efficiency else "Simple"
        ratio = max(simple_efficiency, sac_efficiency) / min(simple_efficiency, sac_efficiency)
        
        print(f"  {amp}°: {winner} is {ratio:.1f}x more efficient")


def plot_sample_trajectories(simple_results, sac_results, sample_amplitudes=[5, 30, 90]):
    """Plot sample trajectories for different amplitudes"""
    n_samples = len(sample_amplitudes)
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 6))
    if n_samples == 1:
        axes = [axes]
    
    fig.suptitle('Sample Trajectories at Different Amplitudes', fontsize=16)
    
    for i, amp in enumerate(sample_amplitudes):
        if amp in simple_results and amp in sac_results:
            # Plot first 5 seconds
            end_idx = min(500, len(simple_results[amp]['times']))
            
            times = simple_results[amp]['times'][:end_idx]
            simple_angles = np.degrees(simple_results[amp]['angles'][:end_idx])
            sac_angles = np.degrees(sac_results[amp]['angles'][:end_idx])
            targets = np.degrees(simple_results[amp]['targets'][:end_idx])
            
            axes[i].plot(times, targets, 'k--', linewidth=1, label='Target', alpha=0.7)
            axes[i].plot(times, simple_angles, 'b-', linewidth=2, label='Simple', alpha=0.8)
            axes[i].plot(times, sac_angles, 'r-', linewidth=2, label='SAC', alpha=0.8)
            axes[i].set_xlabel('Time (sec)')
            axes[i].set_ylabel('Angle (degrees)')
            axes[i].set_title(f'{amp}° Amplitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Set consistent y-axis range
            axes[i].set_ylim(-amp*1.2, amp*1.2)
    
    plt.tight_layout()
    plt.show()


def main():
    print("💪 Experiment 3: Multi-Amplitude Analysis")
    print("="*60)
    print("Testing fine motor control vs gross motor movements")
    print("From micro-adjustments to full-range athletic actions")
    
    # Load trained SAC model
    try:
        sac_model = SAC.load("sac_elbow_model")
        print("✅ Loaded trained SAC model")
    except:
        print("❌ Could not load SAC model. Please train it first with elbow_rl_training.py")
        return
    
    # Test both controllers across amplitudes
    print(f"\n{'='*60}")
    simple_results = run_amplitude_sweep("Simple")
    sac_results = run_amplitude_sweep("SAC", sac_model)
    
    # Analyze motor control characteristics
    analyze_motor_control_characteristics(simple_results, sac_results)
    
    # Create comprehensive plots
    try:
        plot_amplitude_analysis(simple_results, sac_results)
        print("\n📊 Amplitude analysis plots displayed!")
        
        plot_sample_trajectories(simple_results, sac_results)
        print("📊 Sample trajectory plots displayed!")
    except Exception as e:
        print(f"\n○ Plotting failed: {e}")
    
    # Summary insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print("="*60)
    print("🎯 Fine Motor Control: Which excels at precise, small movements?")
    print("💪 Gross Motor Control: Which handles large, athletic movements better?") 
    print("📏 Linearity: How does performance scale with movement size?")
    print("⚡ Efficiency: Which uses less muscle effort for same result?")
    print("🎮 Sports Game Design: What movement ranges need which controller?")
    
    print(f"\n✅ Experiment 3 Complete!")
    print("Ready for Experiment 4: Random Walk Targets (Adaptability Test)!")
    
    return simple_results, sac_results


if __name__ == "__main__":
    simple_results, sac_results = main()