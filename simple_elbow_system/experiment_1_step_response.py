"""
Experiment 1: Step Response Analysis
Compare SAC vs Simple Controller on sudden target changes

Key Metrics:
- Rise time (time to reach 90% of target)
- Overshoot (maximum overshoot percentage) 
- Settling time (time to stay within 5% of target)
- Steady-state error (final tracking error)

This reveals control system characteristics that sine wave tracking can't show!
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time

# Import our working systems
from elbow_gym_env import ElbowTrackingEnv, SimpleElbowSystem, ElbowParams


class StepResponseEnv(ElbowTrackingEnv):
    """Modified environment for step response testing"""
    
    def __init__(self, step_targets=None, step_times=None, max_episode_steps=1000):
        super().__init__(max_episode_steps=max_episode_steps)
        
        # Default step sequence: 0° -> 30° -> -30° -> 15° -> 0°
        self.step_targets = step_targets or [0.0, 0.52, -0.52, 0.26, 0.0]  # radians
        self.step_times = step_times or [100, 200, 300, 400, 500]  # steps (1 sec each at 100Hz)
        
        self.current_target_index = 0
        self.step_start_time = 0
        
    def _generate_target(self):
        """Generate step response targets"""
        current_step = int(self.elbow.time / self.elbow.params.dt)
        
        # Check if we need to step to next target
        if (self.current_target_index < len(self.step_times) and 
            current_step >= self.step_times[self.current_target_index]):
            
            self.current_target_index += 1
            self.step_start_time = current_step
        
        # Return current target
        if self.current_target_index < len(self.step_targets):
            return self.step_targets[self.current_target_index]
        else:
            return self.step_targets[-1]  # Hold last target


def analyze_step_response(times, angles, targets, step_time_idx, target_value):
    """
    Analyze step response characteristics for a single step
    Returns: rise_time, overshoot, settling_time, steady_state_error
    """
    # Convert to numpy arrays for math operations
    times = np.array(times)
    angles = np.array(angles)
    targets = np.array(targets)
    
    # Extract data from step start onwards
    step_start_idx = step_time_idx
    if step_start_idx >= len(times):
        return None, None, None, None
    
    # Look at next 3 seconds after step (300 samples at 100Hz)
    end_idx = min(step_start_idx + 300, len(times))
    step_times = times[step_start_idx:end_idx] - times[step_start_idx]
    step_angles = angles[step_start_idx:end_idx]
    
    if len(step_angles) < 50:  # Need minimum data
        return None, None, None, None
    
    initial_angle = step_angles[0]
    target_change = target_value - initial_angle
    
    if abs(target_change) < 0.01:  # Ignore tiny steps
        return None, None, None, None
    
    # 1. Rise time (time to reach 90% of target change)
    ninety_percent = initial_angle + 0.9 * target_change
    rise_time = None
    for i, angle in enumerate(step_angles):
        if (target_change > 0 and angle >= ninety_percent) or \
           (target_change < 0 and angle <= ninety_percent):
            rise_time = step_times[i]
            break
    
    # 2. Overshoot (maximum overshoot as percentage)
    if target_change > 0:
        max_angle = np.max(step_angles)
        overshoot = max((max_angle - target_value) / abs(target_change) * 100, 0)
    else:
        min_angle = np.min(step_angles)
        overshoot = max((target_value - min_angle) / abs(target_change) * 100, 0)
    
    # 3. Settling time (time to stay within 5% of target)
    five_percent_band = 0.05 * abs(target_change)
    settling_time = None
    for i in range(len(step_angles)-50, -1, -1):  # Work backwards
        if abs(step_angles[i] - target_value) > five_percent_band:
            settling_time = step_times[min(i + 50, len(step_times)-1)]
            break
    if settling_time is None:
        settling_time = step_times[min(50, len(step_times)-1)]  # Settled quickly
    
    # 4. Steady-state error (average error in last 1 second)
    last_second_idx = max(0, len(step_angles) - 100)
    steady_state_error = np.mean(np.abs(step_angles[last_second_idx:] - target_value))
    
    return rise_time, overshoot, settling_time, steady_state_error


def test_controller_step_response(controller_type, model=None, n_runs=3):
    """Test a controller on step response tasks"""
    print(f"\n=== Testing {controller_type} Controller ===")
    
    all_results = {
        'rise_times': [], 'overshoots': [], 'settling_times': [], 'steady_errors': [],
        'times': [], 'angles': [], 'targets': []
    }
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        
        # Create test environment
        env = StepResponseEnv(max_episode_steps=600)  # 6 seconds
        obs, info = env.reset()
        
        times = [env.elbow.time]
        angles = [env.elbow.angle]
        targets = [env.target_angle]
        
        # Run episode
        for step in range(600):
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
            
            times.append(env.elbow.time)
            angles.append(env.elbow.angle)
            targets.append(env.target_angle)
            
            if terminated or truncated:
                break
        
        # Store data from this run
        all_results['times'].append(times)
        all_results['angles'].append(angles) 
        all_results['targets'].append(targets)
        
        # Analyze each step response in this run
        step_times_sec = [1.0, 2.0, 3.0, 4.0]  # When steps occur
        step_targets = [0.52, -0.52, 0.26, 0.0]  # Target values
        
        for i, (step_time_sec, target_val) in enumerate(zip(step_times_sec, step_targets)):
            step_idx = int(step_time_sec * 100)  # Convert to sample index
            
            rise_t, overshoot, settle_t, ss_error = analyze_step_response(
                times, angles, targets, step_idx, target_val
            )
            
            if rise_t is not None:  # Valid analysis
                all_results['rise_times'].append(rise_t)
                all_results['overshoots'].append(overshoot)
                all_results['settling_times'].append(settle_t)
                all_results['steady_errors'].append(ss_error)
    
    return all_results


def print_step_response_summary(controller_name, results):
    """Print summary statistics for step response analysis"""
    if len(results['rise_times']) == 0:
        print(f"{controller_name}: No valid step responses analyzed")
        return
        
    print(f"\n{controller_name} Step Response Summary:")
    print(f"  Rise Time (90%):    {np.mean(results['rise_times']):.3f} ± {np.std(results['rise_times']):.3f} sec")
    print(f"  Overshoot:          {np.mean(results['overshoots']):.1f} ± {np.std(results['overshoots']):.1f} %")
    print(f"  Settling Time (5%): {np.mean(results['settling_times']):.3f} ± {np.std(results['settling_times']):.3f} sec") 
    print(f"  Steady-State Error: {np.degrees(np.mean(results['steady_errors'])):.2f} ± {np.degrees(np.std(results['steady_errors'])):.2f} deg")


def plot_step_responses(simple_results, sac_results):
    """Plot representative step response comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Step Response Analysis: Simple Controller vs SAC Agent', fontsize=16)
    
    # Use first run from each controller for plotting
    simple_times = np.array(simple_results['times'][0])
    simple_angles = np.array(simple_results['angles'][0]) 
    simple_targets = np.array(simple_results['targets'][0])
    
    sac_times = np.array(sac_results['times'][0])
    sac_angles = np.array(sac_results['angles'][0])
    sac_targets = np.array(sac_results['targets'][0])
    
    # Plot 1: Full sequence
    axes[0,0].plot(simple_times, np.degrees(simple_angles), 'b-', linewidth=2, label='Simple Controller', alpha=0.8)
    axes[0,0].plot(sac_times, np.degrees(sac_angles), 'r-', linewidth=2, label='SAC Agent', alpha=0.8)
    axes[0,0].plot(simple_times, np.degrees(simple_targets), 'k--', linewidth=1, label='Target', alpha=0.7)
    axes[0,0].set_xlabel('Time (sec)')
    axes[0,0].set_ylabel('Angle (degrees)')
    axes[0,0].set_title('Complete Step Response Sequence')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2-4: Individual step responses
    step_ranges = [(90, 200), (190, 300), (290, 400)]  # Sample indices for each step
    step_titles = ['Step 1: 0° → 30°', 'Step 2: 30° → -30°', 'Step 3: -30° → 15°']
    
    for i, (start_idx, end_idx) in enumerate(step_ranges):
        ax = axes[0,1] if i == 0 else axes[1,0] if i == 1 else axes[1,1]
        
        # Extract step data
        s_t = simple_times[start_idx:end_idx] - simple_times[start_idx]
        s_a = np.degrees(simple_angles[start_idx:end_idx])
        s_target = np.degrees(simple_targets[start_idx:end_idx])
        
        sac_t = sac_times[start_idx:end_idx] - sac_times[start_idx]  
        sac_a = np.degrees(sac_angles[start_idx:end_idx])
        
        ax.plot(s_t, s_a, 'b-', linewidth=2, label='Simple', alpha=0.8)
        ax.plot(sac_t, sac_a, 'r-', linewidth=2, label='SAC', alpha=0.8)
        ax.plot(s_t, s_target, 'k--', linewidth=1, label='Target', alpha=0.7)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(step_titles[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add rise time markers
        target_val = s_target[-1]
        initial_val = s_a[0]
        ninety_percent = initial_val + 0.9 * (target_val - initial_val)
        ax.axhline(y=ninety_percent, color='gray', linestyle=':', alpha=0.5, label='90% line')
    
    plt.tight_layout()
    plt.show()


def main():
    print("🎯 Experiment 1: Step Response Analysis")
    print("="*60)
    print("Testing system responsiveness to sudden target changes")
    print("Classic control system analysis: rise time, overshoot, settling time")
    
    # Load trained SAC model
    try:
        sac_model = SAC.load("sac_elbow_model")
        print("✅ Loaded trained SAC model")
    except:
        print("❌ Could not load SAC model. Please train it first with elbow_rl_training.py")
        return
    
    # Test both controllers
    print("\n" + "="*60)
    simple_results = test_controller_step_response("Simple", n_runs=3)
    sac_results = test_controller_step_response("SAC", sac_model, n_runs=3)
    
    # Analyze results
    print("\n" + "="*60)
    print("STEP RESPONSE ANALYSIS RESULTS")
    print("="*60)
    
    print_step_response_summary("Simple Controller", simple_results)
    print_step_response_summary("SAC Agent", sac_results)
    
    # Compare key metrics
    if len(simple_results['rise_times']) > 0 and len(sac_results['rise_times']) > 0:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print("="*60)
        
        simple_rise = np.mean(simple_results['rise_times'])
        sac_rise = np.mean(sac_results['rise_times'])
        rise_improvement = ((simple_rise - sac_rise) / simple_rise) * 100
        
        simple_overshoot = np.mean(simple_results['overshoots'])
        sac_overshoot = np.mean(sac_results['overshoots'])
        
        simple_settle = np.mean(simple_results['settling_times'])  
        sac_settle = np.mean(sac_results['settling_times'])
        settle_improvement = ((simple_settle - sac_settle) / simple_settle) * 100
        
        print(f"Rise Time:    Simple {simple_rise:.3f}s vs SAC {sac_rise:.3f}s", end="")
        if rise_improvement > 0:
            print(f" (✅ {rise_improvement:.1f}% faster)")
        else:
            print(f" (❌ {abs(rise_improvement):.1f}% slower)")
        
        print(f"Overshoot:    Simple {simple_overshoot:.1f}% vs SAC {sac_overshoot:.1f}%", end="")
        if sac_overshoot < simple_overshoot:
            print(f" (✅ {simple_overshoot-sac_overshoot:.1f}% less overshoot)")
        else:
            print(f" (❌ {sac_overshoot-simple_overshoot:.1f}% more overshoot)")
        
        print(f"Settle Time:  Simple {simple_settle:.3f}s vs SAC {sac_settle:.3f}s", end="")
        if settle_improvement > 0:
            print(f" (✅ {settle_improvement:.1f}% faster)")
        else:
            print(f" (❌ {abs(settle_improvement):.1f}% slower)")
    
    # Plot results
    try:
        plot_step_responses(simple_results, sac_results)
        print("\n📊 Step response plots displayed!")
    except:
        print("\n○ Plotting skipped (matplotlib not available)")
    
    print(f"\n✅ Experiment 1 Complete!")
    print("\n🔍 Key Insights:")
    print("- Rise time shows how quickly each controller responds to changes")
    print("- Overshoot reveals control stability and smoothness")  
    print("- Settling time indicates how well each controller handles transitions")
    print("- SAC may excel at different aspects than simple proportional control")
    
    print(f"\n📁 Data saved for further analysis:")
    print("- simple_results: baseline controller performance")
    print("- sac_results: RL agent performance") 
    print("\nReady for Experiment 2: Different Frequencies!")


if __name__ == "__main__":
    main()