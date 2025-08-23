"""
Phase 3 Week 1: Ankle-Only Balance Physics
Simple inverted pendulum with two ankle muscles
Building directly on Phase 2 success patterns
"""

import numpy as np
import matplotlib.pyplot as plt

class AnkleBalanceSystem:
    """
    Single ankle joint balance system (inverted pendulum)
    Two muscles: plantarflexor (calf) and dorsiflexor (shin)
    """
    
    def __init__(self):
        # Body parameters (more conservative for stability)
        self.body_mass = 70.0          # kg (adult human)
        self.body_height = 0.9         # m (center of mass height from ankle)
        self.body_inertia = 2.0        # kg⋅m² (increased for stability)
        
        # Joint state
        self.ankle_angle = 0.0         # rad (0 = upright, + = forward lean)
        self.ankle_velocity = 0.0      # rad/s
        
        # Muscle parameters (much smaller for gentle control)
        self.plantarflexor_max_force = 50.0      # N (very conservative)
        self.dorsiflexor_max_force = 40.0        # N (very conservative)
        self.plantarflexor_moment_arm = 0.01     # m (small moment arms)
        self.dorsiflexor_moment_arm = 0.01       # m (small moment arms)
        
        # Physics parameters
        self.gravity = 9.81            # m/s²
        self.damping = 5.0             # N⋅m⋅s/rad (increased damping for stability)
        self.dt = 0.01                 # s (100 Hz, same as Phase 2)
        
        # Current muscle activations
        self.plantarflexor_activation = 0.0
        self.dorsiflexor_activation = 0.0
        
    def reset(self, perturbation_angle=None):
        """Reset to perturbed standing position (recovery task setup)"""
        if perturbation_angle is None:
            # Random perturbation: ±10 degrees (like Phase 2 random targets)
            perturbation_angle = np.random.choice([-0.17, 0.17])  # ±10 degrees
        
        self.ankle_angle = perturbation_angle
        self.ankle_velocity = 0.0
        self.plantarflexor_activation = 0.0
        self.dorsiflexor_activation = 0.0
        
    def get_recovery_task_state(self):
        """Get state for recovery task (like Phase 2 reaching state)"""
        target_angle = 0.0  # Always trying to reach upright
        angle_error = self.ankle_angle - target_angle
        
        return {
            'ankle_angle': self.ankle_angle,
            'ankle_velocity': self.ankle_velocity,
            'target_angle': target_angle,
            'angle_error': angle_error,
            'is_recovered': self.is_recovered(),
            'is_fallen': self.is_fallen()
        }
        
    def is_recovered(self, tolerance=0.035):  # ±2 degrees
        """Check if recovered to upright (success condition)"""
        return abs(self.ankle_angle) < tolerance
        
    def is_fallen(self, fall_threshold=0.35):  # ±20 degrees  
        """Check if fallen over (failure condition)"""
        return abs(self.ankle_angle) > fall_threshold
        
    def set_muscle_activations(self, plantarflexor_act, dorsiflexor_act):
        """Set muscle activation levels [0, 1]"""
        self.plantarflexor_activation = np.clip(plantarflexor_act, 0.0, 1.0)
        self.dorsiflexor_activation = np.clip(dorsiflexor_act, 0.0, 1.0)
        
    def compute_muscle_forces(self):
        """Compute muscle forces using simple model (same as Phase 2)"""
        plantarflexor_force = self.plantarflexor_activation * self.plantarflexor_max_force
        dorsiflexor_force = self.dorsiflexor_activation * self.dorsiflexor_max_force
        return plantarflexor_force, dorsiflexor_force
        
    def compute_muscle_torques(self):
        """Convert muscle forces to ankle joint torques"""
        pf_force, df_force = self.compute_muscle_forces()
        
        # Plantarflexor creates forward-leaning torque (positive)
        # Dorsiflexor creates backward-leaning torque (negative)
        plantarflexor_torque = pf_force * self.plantarflexor_moment_arm
        dorsiflexor_torque = -df_force * self.dorsiflexor_moment_arm
        
        total_muscle_torque = plantarflexor_torque + dorsiflexor_torque
        return total_muscle_torque
        
    def compute_gravity_torque(self):
        """Compute gravitational torque (tries to tip body forward)"""
        # Gravity creates torque proportional to sin(angle)
        # Positive angle (forward lean) increases forward torque
        gravity_torque = self.body_mass * self.gravity * self.body_height * np.sin(self.ankle_angle)
        return gravity_torque
        
    def compute_total_torque(self):
        """Sum all torques acting on ankle joint"""
        muscle_torque = self.compute_muscle_torques()
        gravity_torque = self.compute_gravity_torque()
        damping_torque = -self.damping * self.ankle_velocity
        
        total_torque = muscle_torque + gravity_torque + damping_torque
        return total_torque
        
    def step(self):
        """Advance physics simulation by one time step"""
        # Compute angular acceleration from total torque
        total_torque = self.compute_total_torque()
        angular_acceleration = total_torque / self.body_inertia
        
        # Integrate using Euler method (same as Phase 2)
        self.ankle_velocity += angular_acceleration * self.dt
        self.ankle_angle += self.ankle_velocity * self.dt
        
    def get_state(self):
        """Get current system state"""
        return {
            'ankle_angle': self.ankle_angle,
            'ankle_velocity': self.ankle_velocity,
            'plantarflexor_activation': self.plantarflexor_activation,
            'dorsiflexor_activation': self.dorsiflexor_activation,
            'center_of_mass_x': self.body_height * np.sin(self.ankle_angle),  # horizontal CM position
            'is_balanced': self.is_balanced()
        }
        
    def is_balanced(self, angle_threshold=0.3):
        """Check if system is still balanced (not fallen over)"""
        return abs(self.ankle_angle) < angle_threshold  # ~17 degrees (more realistic)
        
    def get_center_of_mass_position(self):
        """Get center of mass position for visualization"""
        cm_x = self.body_height * np.sin(self.ankle_angle)
        cm_y = self.body_height * np.cos(self.ankle_angle)
        return cm_x, cm_y


# Test the physics for recovery task
if __name__ == "__main__":
    print("Testing Ankle Recovery Physics (Simple Task)...")
    
    # Create system
    ankle_system = AnkleBalanceSystem()
    
    # Test 1: Forward lean recovery with dorsiflexor (should pull back)
    print("\nTest 1: Forward lean (+10°) → dorsiflexor recovery")
    ankle_system.reset(perturbation_angle=0.17)  # +10 degrees
    ankle_system.set_muscle_activations(0.0, 0.5)  # dorsiflexor only
    
    for i in range(150):
        ankle_system.step()
        if i % 30 == 0:
            state = ankle_system.get_recovery_task_state()
            angle_deg = np.degrees(state['ankle_angle'])
            print(f"Step {i}: Angle = {angle_deg:.1f}°, Error = {np.degrees(state['angle_error']):.1f}°, Recovered = {state['is_recovered']}")
    
    # Test 2: Backward lean recovery with plantarflexor (should pull forward)  
    print("\nTest 2: Backward lean (-10°) → plantarflexor recovery")
    ankle_system.reset(perturbation_angle=-0.17)  # -10 degrees
    ankle_system.set_muscle_activations(0.5, 0.0)  # plantarflexor only
    
    for i in range(150):
        ankle_system.step()
        if i % 30 == 0:
            state = ankle_system.get_recovery_task_state()
            angle_deg = np.degrees(state['ankle_angle'])
            print(f"Step {i}: Angle = {angle_deg:.1f}°, Error = {np.degrees(state['angle_error']):.1f}°, Recovered = {state['is_recovered']}")
    
    # Test 3: No muscle activation (should fail to recover)
    print("\nTest 3: Forward lean (+10°) → no muscle activation (should fail)")
    ankle_system.reset(perturbation_angle=0.17)  # +10 degrees  
    ankle_system.set_muscle_activations(0.0, 0.0)  # no muscles
    
    for i in range(150):
        ankle_system.step()
        if i % 30 == 0:
            state = ankle_system.get_recovery_task_state()
            angle_deg = np.degrees(state['ankle_angle'])
            print(f"Step {i}: Angle = {angle_deg:.1f}°, Recovered = {state['is_recovered']}, Fallen = {state['is_fallen']}")
    
    print("\nRecovery Physics validation complete! ✓")
    print("Expected: Correct muscle activation should move toward 0°")
    print("Next: Create recovery task environment (ankle_recovery_env.py)")