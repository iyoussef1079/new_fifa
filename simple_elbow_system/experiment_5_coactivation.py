"""
Experiment 5: Muscle Co-activation Pattern Analysis
THE MOST IMPORTANT EXPERIMENT - reveals what each controller actually learned!

This answers the fundamental question:
What muscle coordination strategies distinguish RL from classical control?

Key Analyses:
1. Co-activation Patterns: When do both muscles fire simultaneously?
2. Antagonistic Strategies: How do they handle opposing muscles?
3. Energy Efficiency: Which uses muscles more efficiently?
4. Biological Realism: Which patterns match human EMG data?
5. Coordination Timing: Muscle activation sequences and transitions

This reveals the SECRET SAUCE of each approach!
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time

# Import our working systems
from elbow_gym_env import ElbowTrackingEnv


def analyze_muscle_coordination(controller_type, model=None, test_scenarios=None):
    """
    Comprehensive analysis of muscle coordination patterns
    Tests multiple scenarios to reveal coordination strategies
    """
    print(f"\n=== Analyzing {controller_type} Muscle Coordination ===")
    
    if test_scenarios is None:
        test_scenarios = [
            {"name": "Sine Wave", "env_type": "sine", "freq": 0.5, "amp": 30},
            {"name": "Step Response", "env_type": "step", "targets": [0, 45, -45, 20]},
            {"name": "Large Movements", "env_type": "sine", "freq": 0.3, "amp": 60},
            {"name": "Fast Movements", "env_type": "sine", "freq": 1.0, "amp": 30},
        ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"  Testing {scenario['name']}...")
        
        if scenario['env_type'] == 'sine':
            result = test_sine_coordination(
                controller_type, model, 
                frequency=scenario['freq'], 
                amplitude=scenario['amp']
            )
        elif scenario['env_type'] == 'step':
            result = test_step_coordination(controller_type, model, scenario['targets'])
        
        all_results[scenario['name']] = result
    
    return all_results


def test_sine_coordination(controller_type, model, frequency=0.5, amplitude=30, duration=8.0):
    """Test muscle coordination on sine wave tracking"""
    
    # Create environment
    env = ElbowTrackingEnv(max_episode_steps=int(duration * 100))
    env.target_frequency = frequency
    
    # Override target generation for custom amplitude
    original_generate = env._generate_target
    def custom_target():
        return np.radians(amplitude) * np.sin(2 * np.pi * frequency * env.elbow.time)
    env._generate_target = custom_target
    
    # Run test
    obs, info = env.reset()
    
    data = {
        'times': [], 'angles': [], 'targets': [], 'velocities': [],
        'bicep_actions': [], 'tricep_actions': [], 'errors': [],
        'phase_angles': []  # For phase analysis
    }
    
    for step in range(int(duration * 100)):
        if controller_type == "Simple":
            error = obs[3]
            kp = 4.0
            if error > 0:
                action = [min(kp * error, 1.0), 0.0]
            else:
                action = [0.0, min(kp * abs(error), 1.0)]
        elif controller_type == "SAC":
            action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record all data
        data['times'].append(env.elbow.time)
        data['angles'].append(env.elbow.angle)
        data['targets'].append(env.target_angle)
        data['velocities'].append(env.elbow.velocity)
        data['bicep_actions'].append(action[0])
        data['tricep_actions'].append(action[1])
        data['errors'].append(info['tracking_error'])
        
        # Calculate phase angle (for cycle analysis)
        phase = (2 * np.pi * frequency * env.elbow.time) % (2 * np.pi)
        data['phase_angles'].append(phase)
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def test_step_coordination(controller_type, model, targets, hold_time=2.0):
    """Test muscle coordination during step responses"""
    
    env = ElbowTrackingEnv(max_episode_steps=int(len(targets) * hold_time * 100))
    
    data = {
        'times': [], 'angles': [], 'targets': [], 'velocities': [],
        'bicep_actions': [], 'tricep_actions': [], 'errors': [],
        'step_phases': []  # Track which step we're in
    }
    
    obs, info = env.reset()
    current_target_idx = 0
    step_start_time = 0
    
    for step in range(int(len(targets) * hold_time * 100)):
        # Update target based on time
        elapsed_time = env.elbow.time - step_start_time
        if elapsed_time >= hold_time and current_target_idx < len(targets) - 1:
            current_target_idx += 1
            step_start_time = env.elbow.time
        
        # Set target manually
        if current_target_idx < len(targets):
            env.target_angle = np.radians(targets[current_target_idx])
        
        # Get action
        # Manually construct observation with correct target
        obs = np.array([env.elbow.angle, env.elbow.velocity, env.target_angle, 
                       env.target_angle - env.elbow.angle], dtype=np.float32)
        
        if controller_type == "Simple":
            error = obs[3]
            kp = 4.0
            if error > 0:
                action = [min(kp * error, 1.0), 0.0]
            else:
                action = [0.0, min(kp * abs(error), 1.0)]
        elif controller_type == "SAC":
            action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record data
        data['times'].append(env.elbow.time)
        data['angles'].append(env.elbow.angle)
        data['targets'].append(env.target_angle)
        data['velocities'].append(env.elbow.velocity)
        data['bicep_actions'].append(action[0])
        data['tricep_actions'].append(action[1])
        data['errors'].append(abs(env.target_angle - env.elbow.angle))
        data['step_phases'].append(current_target_idx)
        
        if terminated or truncated:
            break
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def analyze_coactivation_patterns(results_dict):
    """Analyze co-activation patterns across all scenarios"""
    print(f"\n{'='*60}")
    print("CO-ACTIVATION PATTERN ANALYSIS")
    print("="*60)
    
    coactivation_metrics = {}
    
    for scenario_name, data in results_dict.items():
        bicep = data['bicep_actions']
        tricep = data['tricep_actions']
        
        # 1. Co-activation ratio (both muscles active simultaneously)
        both_active = (bicep > 0.1) & (tricep > 0.1)
        coactivation_ratio = np.mean(both_active) * 100
        
        # 2. Antagonistic ratio (only one muscle active)
        only_bicep = (bicep > 0.1) & (tricep <= 0.1)
        only_tricep = (tricep > 0.1) & (bicep <= 0.1)
        antagonistic_ratio = np.mean(only_bicep | only_tricep) * 100
        
        # 3. Rest ratio (neither muscle significantly active)
        both_rest = (bicep <= 0.1) & (tricep <= 0.1)
        rest_ratio = np.mean(both_rest) * 100
        
        # 4. Average co-activation level when both are active
        if np.any(both_active):
            avg_coactivation = np.mean(np.minimum(bicep[both_active], tricep[both_active]))
        else:
            avg_coactivation = 0.0
        
        # 5. Energy efficiency (total activation per unit accuracy)
        total_activation = np.mean(bicep + tricep)
        mean_error = np.degrees(np.mean(data['errors']))
        efficiency = mean_error / total_activation if total_activation > 0 else float('inf')
        
        coactivation_metrics[scenario_name] = {
            'coactivation_ratio': coactivation_ratio,
            'antagonistic_ratio': antagonistic_ratio,
            'rest_ratio': rest_ratio,
            'avg_coactivation_level': avg_coactivation,
            'total_activation': total_activation,
            'efficiency': efficiency,
            'mean_error': mean_error
        }
        
        print(f"{scenario_name}:")
        print(f"  Co-activation:   {coactivation_ratio:.1f}% of time")
        print(f"  Antagonistic:    {antagonistic_ratio:.1f}% of time")
        print(f"  Rest:            {rest_ratio:.1f}% of time")
        print(f"  Avg Co-level:    {avg_coactivation:.3f}")
        print(f"  Total Effort:    {total_activation:.3f}")
        print(f"  Efficiency:      {efficiency:.2f} °/effort")
    
    return coactivation_metrics


def analyze_muscle_timing(data, scenario_name):
    """Analyze muscle activation timing and sequences"""
    bicep = data['bicep_actions']
    tricep = data['tricep_actions']
    angles = data['angles']
    velocities = data['velocities']
    
    print(f"\n🕐 Timing Analysis: {scenario_name}")
    
    # Find muscle switching events
    bicep_dominant = bicep > tricep + 0.1
    tricep_dominant = tricep > bicep + 0.1
    
    switches = []
    current_state = None
    
    for i in range(len(bicep_dominant)):
        if bicep_dominant[i] and current_state != 'bicep':
            switches.append(('bicep', i, angles[i], velocities[i]))
            current_state = 'bicep'
        elif tricep_dominant[i] and current_state != 'tricep':
            switches.append(('tricep', i, angles[i], velocities[i]))
            current_state = 'tricep'
    
    # Analyze switch timing relative to movement phase
    flexion_switches = [s for s in switches if s[0] == 'bicep']
    extension_switches = [s for s in switches if s[0] == 'tricep']
    
    if flexion_switches:
        avg_flex_switch_vel = np.mean([s[3] for s in flexion_switches])
        print(f"  Bicep switches at avg velocity: {avg_flex_switch_vel:.3f} rad/s")
    
    if extension_switches:
        avg_ext_switch_vel = np.mean([s[3] for s in extension_switches])
        print(f"  Tricep switches at avg velocity: {avg_ext_switch_vel:.3f} rad/s")
    
    # Predictive vs reactive timing
    if 'phase_angles' in data:
        # For sine waves, analyze phase relationship
        phases = data['phase_angles']
        bicep_phases = phases[bicep > 0.5]
        tricep_phases = phases[tricep > 0.5]
        
        if len(bicep_phases) > 0:
            avg_bicep_phase = np.mean(bicep_phases) * 180 / np.pi
            print(f"  Bicep most active at phase: {avg_bicep_phase:.0f}°")
        
        if len(tricep_phases) > 0:
            avg_tricep_phase = np.mean(tricep_phases) * 180 / np.pi
            print(f"  Tricep most active at phase: {avg_tricep_phase:.0f}°")


def plot_muscle_coordination_comparison(simple_results, sac_results):
    """Create comprehensive muscle coordination comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Muscle Coordination Analysis: Simple Controller vs SAC Agent', fontsize=16)
    
    # Select representative scenario for detailed analysis
    scenario = "Sine Wave"
    simple_data = simple_results[scenario]
    sac_data = sac_results[scenario]
    
    # Plot 1: Muscle activations over time
    time_range = slice(0, 400)  # First 4 seconds
    axes[0,0].plot(simple_data['times'][time_range], simple_data['bicep_actions'][time_range], 
                   'b-', linewidth=2, label='Simple Bicep', alpha=0.8)
    axes[0,0].plot(simple_data['times'][time_range], simple_data['tricep_actions'][time_range], 
                   'r-', linewidth=2, label='Simple Tricep', alpha=0.8)
    axes[0,0].plot(sac_data['times'][time_range], sac_data['bicep_actions'][time_range], 
                   'b--', linewidth=2, label='SAC Bicep', alpha=0.8)
    axes[0,0].plot(sac_data['times'][time_range], sac_data['tricep_actions'][time_range], 
                   'r--', linewidth=2, label='SAC Tricep', alpha=0.8)
    axes[0,0].set_xlabel('Time (sec)')
    axes[0,0].set_ylabel('Muscle Activation')
    axes[0,0].set_title('Muscle Activation Patterns')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Co-activation scatter plot
    axes[0,1].scatter(simple_data['bicep_actions'][::10], simple_data['tricep_actions'][::10], 
                     c='blue', alpha=0.6, s=20, label='Simple Controller')
    axes[0,1].scatter(sac_data['bicep_actions'][::10], sac_data['tricep_actions'][::10], 
                     c='red', alpha=0.6, s=20, label='SAC Agent')
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal Activation')
    axes[0,1].set_xlabel('Bicep Activation')
    axes[0,1].set_ylabel('Tricep Activation')
    axes[0,1].set_title('Co-activation Patterns')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Activation vs movement phase
    if 'phase_angles' in simple_data:
        phases_deg = simple_data['phase_angles'] * 180 / np.pi
        axes[0,2].scatter(phases_deg[::5], simple_data['bicep_actions'][::5], 
                         c='blue', alpha=0.6, s=15, label='Simple Bicep')
        axes[0,2].scatter(phases_deg[::5], simple_data['tricep_actions'][::5], 
                         c='red', alpha=0.6, s=15, label='Simple Tricep')
        
        sac_phases_deg = sac_data['phase_angles'] * 180 / np.pi
        axes[0,2].scatter(sac_phases_deg[::5], sac_data['bicep_actions'][::5], 
                         c='lightblue', alpha=0.6, s=15, marker='^', label='SAC Bicep')
        axes[0,2].scatter(sac_phases_deg[::5], sac_data['tricep_actions'][::5], 
                         c='lightcoral', alpha=0.6, s=15, marker='^', label='SAC Tricep')
        
        axes[0,2].set_xlabel('Movement Phase (degrees)')
        axes[0,2].set_ylabel('Muscle Activation')
        axes[0,2].set_title('Phase-dependent Activation')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Efficiency comparison across scenarios
    scenarios = list(simple_results.keys())
    simple_efforts = []
    sac_efforts = []
    simple_errors = []
    sac_errors = []
    
    for scenario in scenarios:
        simple_efforts.append(np.mean(simple_results[scenario]['bicep_actions'] + 
                                    simple_results[scenario]['tricep_actions']))
        sac_efforts.append(np.mean(sac_results[scenario]['bicep_actions'] + 
                                 sac_results[scenario]['tricep_actions']))
        simple_errors.append(np.degrees(np.mean(simple_results[scenario]['errors'])))
        sac_errors.append(np.degrees(np.mean(sac_results[scenario]['errors'])))
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    axes[1,0].bar(x_pos - width/2, simple_efforts, width, label='Simple Controller', alpha=0.8, color='blue')
    axes[1,0].bar(x_pos + width/2, sac_efforts, width, label='SAC Agent', alpha=0.8, color='red')
    axes[1,0].set_xlabel('Scenario')
    axes[1,0].set_ylabel('Total Muscle Effort')
    axes[1,0].set_title('Muscle Effort Comparison')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(scenarios, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Error vs Effort efficiency
    axes[1,1].scatter(simple_efforts, simple_errors, c='blue', s=100, alpha=0.8, 
                     label='Simple Controller', marker='o')
    axes[1,1].scatter(sac_efforts, sac_errors, c='red', s=100, alpha=0.8, 
                     label='SAC Agent', marker='s')
    
    for i, scenario in enumerate(scenarios):
        axes[1,1].annotate(scenario, (simple_efforts[i], simple_errors[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,1].annotate(scenario, (sac_efforts[i], sac_errors[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1,1].set_xlabel('Total Muscle Effort')
    axes[1,1].set_ylabel('Mean Tracking Error (degrees)')
    axes[1,1].set_title('Efficiency: Error vs Effort')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Co-activation summary
    coactivation_simple = []
    coactivation_sac = []
    
    for scenario in scenarios:
        simple_coact = np.mean((simple_results[scenario]['bicep_actions'] > 0.1) & 
                              (simple_results[scenario]['tricep_actions'] > 0.1)) * 100
        sac_coact = np.mean((sac_results[scenario]['bicep_actions'] > 0.1) & 
                           (sac_results[scenario]['tricep_actions'] > 0.1)) * 100
        coactivation_simple.append(simple_coact)
        coactivation_sac.append(sac_coact)
    
    axes[1,2].bar(x_pos - width/2, coactivation_simple, width, label='Simple Controller', alpha=0.8, color='blue')
    axes[1,2].bar(x_pos + width/2, coactivation_sac, width, label='SAC Agent', alpha=0.8, color='red')
    axes[1,2].set_xlabel('Scenario')
    axes[1,2].set_ylabel('Co-activation Time (%)')
    axes[1,2].set_title('Muscle Co-activation Patterns')
    axes[1,2].set_xticks(x_pos)
    axes[1,2].set_xticklabels(scenarios, rotation=45)
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    print("🧠💪 Experiment 5: Muscle Co-activation Analysis")
    print("="*60)
    print("THE MOST IMPORTANT EXPERIMENT!")
    print("Reveals what muscle coordination strategies each controller learned")
    
    # Load trained SAC model
    try:
        sac_model = SAC.load("sac_elbow_model")
        print("✅ Loaded trained SAC model")
    except:
        print("❌ Could not load SAC model. Please train it first with elbow_rl_training.py")
        return
    
    # Analyze muscle coordination patterns
    print(f"\n{'='*60}")
    simple_results = analyze_muscle_coordination("Simple")
    sac_results = analyze_muscle_coordination("SAC", sac_model)
    
    # Analyze co-activation patterns
    print(f"\n🔍 SIMPLE CONTROLLER PATTERNS:")
    simple_metrics = analyze_coactivation_patterns(simple_results)
    
    print(f"\n🔍 SAC AGENT PATTERNS:")
    sac_metrics = analyze_coactivation_patterns(sac_results)
    
    # Timing analysis
    for scenario in simple_results.keys():
        analyze_muscle_timing(simple_results[scenario], f"Simple - {scenario}")
        analyze_muscle_timing(sac_results[scenario], f"SAC - {scenario}")
    
    # Create comprehensive plots
    try:
        plot_muscle_coordination_comparison(simple_results, sac_results)
        print("\n📊 Muscle coordination analysis plots displayed!")
    except Exception as e:
        print(f"\n○ Plotting failed: {e}")
    
    # Final insights
    print(f"\n{'='*60}")
    print("🧠 KEY COORDINATION INSIGHTS")
    print("="*60)
    print("💪 Co-activation Strategies: How each controller uses antagonistic muscles")
    print("⚡ Energy Efficiency: Which coordination patterns waste less energy")
    print("🎯 Biological Realism: Which patterns match human muscle control")
    print("🔄 Timing Patterns: Predictive vs reactive muscle activation")
    print("🏆 Optimal Strategy: When to use each coordination approach")
    
    print(f"\n✅ Experiment 5 Complete!")
    print("🎉 ALL EXPERIMENTS FINISHED!")
    print("\nYou now have complete understanding of RL vs Classical Control")
    print("for musculoskeletal systems. Ready to build your 2D arm! 🦾")
    
    return simple_results, sac_results, simple_metrics, sac_metrics


if __name__ == "__main__":
    results = main()