"""
Phase 1: Minimal Viable Musculoskeletal System
Single elbow joint with bicep/tricep muscles for angle tracking

Keep it simple: one file, easy to validate, build incrementally
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

@dataclass
class ElbowParams:
    """Simple parameters for our elbow system"""
    # Joint properties
    joint_inertia: float = 0.5    # kg*m^2 (increased for stability)
    joint_damping: float = 2.0    # N*m*s/rad (more damping)
    
    # Muscle properties (reduced for realistic range)
    bicep_max_force: float = 200.0    # N (reduced)
    tricep_max_force: float = 250.0   # N (reduced) 
    bicep_moment_arm: float = 0.05    # m
    tricep_moment_arm: float = 0.04  # m
    
    # Simulation
    dt: float = 0.01  # 100 Hz simulation

class SimpleElbowSystem:
    """
    Minimal elbow system: one joint, two muscles, basic physics
    Goal: Track angle references with muscle activations
    """
    
    def __init__(self, params=None):
        self.params = params or ElbowParams()
        
        # State variables
        self.angle = 0.0      # Current joint angle (rad)  
        self.velocity = 0.0   # Current joint velocity (rad/s)
        
        # Muscle activations (0-1)
        self.bicep_activation = 0.0
        self.tricep_activation = 0.0
        
        # For tracking
        self.time = 0.0
        
    def compute_muscle_forces(self):
        """Simple muscle force calculation - just activation * max_force"""
        bicep_force = self.bicep_activation * self.params.bicep_max_force
        tricep_force = self.tricep_activation * self.params.tricep_max_force
        return bicep_force, tricep_force
    
    def compute_joint_torque(self):
        """Convert muscle forces to joint torque"""
        bicep_force, tricep_force = self.compute_muscle_forces()
        
        # Bicep flexes (positive torque), tricep extends (negative torque)
        bicep_torque = bicep_force * self.params.bicep_moment_arm
        tricep_torque = -tricep_force * self.params.tricep_moment_arm
        
        total_torque = bicep_torque + tricep_torque
        return total_torque
    
    def step(self, bicep_activation, tricep_activation):
        """
        One simulation step - update muscle activations and physics
        Returns: current angle
        """
        # Clamp muscle activations to valid range
        self.bicep_activation = np.clip(bicep_activation, 0.0, 1.0)
        self.tricep_activation = np.clip(tricep_activation, 0.0, 1.0)
        
        # Compute forces and torques
        joint_torque = self.compute_joint_torque()
        
        # Simple joint dynamics: torque = inertia * acceleration + damping * velocity
        # acceleration = (torque - damping * velocity) / inertia
        acceleration = (joint_torque - self.params.joint_damping * self.velocity) / self.params.joint_inertia
        
        # Integrate to get new velocity and angle
        self.velocity += acceleration * self.params.dt
        self.angle += self.velocity * self.params.dt
        
        # Update time
        self.time += self.params.dt
        
        return self.angle
    
    def get_state(self):
        """Get current state for RL or control"""
        return np.array([self.angle, self.velocity])
    
    def reset(self):
        """Reset to initial state"""
        self.angle = 0.0
        self.velocity = 0.0
        self.bicep_activation = 0.0
        self.tricep_activation = 0.0
        self.time = 0.0

def test_basic_functionality():
    """Test 1: Basic functionality - can we move the elbow?"""
    print("Test 1: Basic elbow movement")
    
    elbow = SimpleElbowSystem()
    angles = []
    times = []
    
    # Test: activate bicep for 1 second, then tricep for 1 second
    for step in range(200):  # 2 seconds at 100Hz
        if step < 100:
            # Activate bicep (should flex - positive angle)
            angle = elbow.step(bicep_activation=0.5, tricep_activation=0.0)
        else:
            # Activate tricep (should extend - negative angle) 
            angle = elbow.step(bicep_activation=0.0, tricep_activation=0.8)
        
        angles.append(angle)
        times.append(elbow.time)
    
    # Simple validation
    max_angle = max(angles)
    min_angle = min(angles)
    
    print(f"  Max angle: {max_angle:.3f} rad ({np.degrees(max_angle):.1f} deg)")
    print(f"  Min angle: {min_angle:.3f} rad ({np.degrees(min_angle):.1f} deg)")
    print(f"  Range: {max_angle - min_angle:.3f} rad ({np.degrees(max_angle - min_angle):.1f} deg)")
    
    # Basic checks
    if max_angle > 0.1:  # Should flex with bicep
        print("  ✓ Bicep activation causes flexion")
    else:
        print("  ✗ Bicep activation failed")
        
    if min_angle < -0.1:  # Should extend with tricep
        print("  ✓ Tricep activation causes extension") 
    else:
        print("  ✗ Tricep activation failed")
    
    return angles, times

def test_sine_tracking():
    """Test 2: Can we manually track a sine wave reference?"""
    print("\nTest 2: Manual sine wave tracking")
    
    elbow = SimpleElbowSystem()
    angles = []
    references = []
    times = []
    errors = []
    
    # Simple proportional controller for testing
    kp = 4.0  # Proportional gain
    
    for step in range(500):  # 5 seconds
        # Generate sine wave reference (amplitude 30 degrees, period 2 seconds)
        reference = 0.5 * np.sin(2 * np.pi * elbow.time / 2.0)
        references.append(reference)
        
        # Simple controller: positive error -> more bicep, negative error -> more tricep
        error = reference - elbow.angle
        errors.append(abs(error))
        
        # Convert error to muscle activations (very simple)
        if error > 0:
            # Need to flex (positive angle)
            bicep_act = min(kp * error, 1.0)
            tricep_act = 0.0
        else:
            # Need to extend (negative angle) 
            bicep_act = 0.0
            tricep_act = min(kp * abs(error), 1.0)
        
        # Step simulation
        angle = elbow.step(bicep_act, tricep_act)
        angles.append(angle)
        times.append(elbow.time)
    
    # Calculate tracking performance
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"  Mean tracking error: {mean_error:.4f} rad ({np.degrees(mean_error):.2f} deg)")
    print(f"  Max tracking error: {max_error:.4f} rad ({np.degrees(max_error):.2f} deg)")
    
    # Success criteria from document: <10% tracking error
    reference_amplitude = 0.5
    error_percentage = (mean_error / reference_amplitude) * 100
    print(f"  Error percentage: {error_percentage:.1f}%")
    
    if error_percentage < 10.0:
        print("  ✓ Tracking performance meets Phase 1 milestone!")
    else:
        print("  ○ Tracking needs improvement (but that's what RL is for)")
    
    return angles, references, times, errors

def visualize_results(angles1, times1, angles2, references, times2):
    """Simple visualization of our tests"""
    plt.figure(figsize=(12, 8))
    
    # Test 1: Basic movement
    plt.subplot(2, 1, 1)
    plt.plot(times1, np.degrees(angles1), 'b-', linewidth=2, label='Elbow Angle')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Switch to Tricep')
    plt.ylabel('Angle (degrees)')
    plt.title('Test 1: Basic Elbow Movement (Bicep then Tricep)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test 2: Sine tracking  
    plt.subplot(2, 1, 2)
    plt.plot(times2, np.degrees(references), 'r--', linewidth=2, label='Reference')
    plt.plot(times2, np.degrees(angles2), 'b-', linewidth=2, label='Actual Angle')
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Time (seconds)')
    plt.title('Test 2: Sine Wave Tracking (Simple Controller)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Phase 1: Simple Elbow System Testing ===")
    print("Goal: Validate basic physics and prepare for RL integration\n")
    
    # Run tests
    angles1, times1 = test_basic_functionality()
    angles2, references, times2, errors = test_sine_tracking()
    
    # Visualize if matplotlib available
    try:
        visualize_results(angles1, times1, angles2, references, times2)
        print("\n✓ Visualization complete - check the plots!")
    except:
        print("\n○ Visualization skipped (matplotlib not available)")
    
    print("\n=== Phase 1 Complete ===")
    print("Next steps:")
    print("1. If tests pass: Add RL environment wrapper")
    print("2. If tests fail: Debug physics or parameters")
    print("3. Once working: Integrate with Stable-Baselines3")