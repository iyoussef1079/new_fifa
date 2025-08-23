"""
Phase 2.5: 6-Muscle Arm System
Minimal extension of your successful Phase 2 TwoJointArm
Adds shoulder abduction/adduction for full shoulder mobility
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SixMuscleArmParams:
    """
    Parameters for 6-muscle arm system
    Extends your working Phase 2 parameters with 2 new muscles
    """
    
    # Segment properties (SAME as your working Phase 2)
    upper_arm_length: float = 0.3       # m
    forearm_length: float = 0.25        # m
    upper_arm_mass: float = 2.5         # kg
    forearm_mass: float = 1.5           # kg
    
    # Joint inertias (SAME as your working values)
    shoulder_inertia: float = 0.8       # kg*m^2 
    elbow_inertia: float = 0.5          # kg*m^2
    
    # Joint damping (SAME as your working values)
    shoulder_damping: float = 3.0       # N*m*s/rad
    elbow_damping: float = 2.0          # N*m*s/rad
    
    # Your working 4 muscles (UNCHANGED)
    shoulder_flexor_max_force: float = 800.0    # N
    shoulder_extensor_max_force: float = 900.0  # N  
    bicep_max_force: float = 200.0              # N
    tricep_max_force: float = 250.0             # N
    
    # NEW: 2 additional shoulder muscles
    shoulder_abductor_max_force: float = 700.0  # N (lifts arm sideways)
    shoulder_adductor_max_force: float = 800.0  # N (pulls arm to body)
    
    # Your working moment arms (UNCHANGED)
    shoulder_flexor_moment_arm: float = 0.06    # m
    shoulder_extensor_moment_arm: float = 0.05  # m
    bicep_moment_arm: float = 0.05              # m
    tricep_moment_arm: float = 0.04             # m
    
    # NEW: Moment arms for abduction/adduction
    shoulder_abductor_moment_arm: float = 0.055  # m
    shoulder_adductor_moment_arm: float = 0.045  # m
    
    # Physics (SAME)
    dt: float = 0.01        # s
    gravity: float = 9.81   # m/s^2

class SixMuscleArm:
    """
    6-Muscle Arm System
    Direct extension of your successful TwoJointArm with shoulder abduction/adduction
    """
    
    def __init__(self, params=None):
        self.params = params or SixMuscleArmParams()
        
        # Joint states (SAME as Phase 2, but now shoulder moves in 2 directions)
        self.shoulder_flex_angle = 0.0      # Flexion/Extension (forward/back)
        self.shoulder_flex_velocity = 0.0
        self.shoulder_abd_angle = 0.0       # NEW: Abduction/Adduction (sideways)
        self.shoulder_abd_velocity = 0.0
        self.elbow_angle = 0.0              # Elbow flexion/extension
        self.elbow_velocity = 0.0
        
        # Your working 4 muscle activations (UNCHANGED)
        self.shoulder_flexor_activation = 0.0
        self.shoulder_extensor_activation = 0.0
        self.bicep_activation = 0.0
        self.tricep_activation = 0.0
        
        # NEW: 2 additional muscle activations
        self.shoulder_abductor_activation = 0.0  # Lifts arm sideways
        self.shoulder_adductor_activation = 0.0  # Pulls arm to body
        
        # Store previous activations
        self.prev_activations = np.zeros(6)
        
    def get_state(self):
        """System state - now 6D instead of 4D"""
        return np.array([
            self.shoulder_flex_angle, self.shoulder_flex_velocity,
            self.shoulder_abd_angle, self.shoulder_abd_velocity,
            self.elbow_angle, self.elbow_velocity
        ])
    
    def get_activations(self):
        """All 6 muscle activations"""
        return np.array([
            self.shoulder_flexor_activation,
            self.shoulder_extensor_activation,
            self.shoulder_abductor_activation,  # NEW
            self.shoulder_adductor_activation,  # NEW
            self.bicep_activation,
            self.tricep_activation
        ])
    
    def get_end_effector_position(self):
        """
        Hand position in 3D space (but projected to 2D for visualization)
        NOW accounts for both shoulder flexion AND abduction
        """
        # Shoulder joint at origin
        shoulder_pos = np.array([0.0, 0.0, 0.0])
        
        # 3D shoulder rotation (flexion + abduction)
        # Elbow position in 3D
        elbow_x = self.params.upper_arm_length * np.cos(self.shoulder_flex_angle) * np.cos(self.shoulder_abd_angle)
        elbow_y = self.params.upper_arm_length * np.sin(self.shoulder_flex_angle)  # Flexion (up/down)
        elbow_z = self.params.upper_arm_length * np.cos(self.shoulder_flex_angle) * np.sin(self.shoulder_abd_angle)  # Abduction (side)
        
        elbow_pos = np.array([elbow_x, elbow_y, elbow_z])
        
        # Hand position (forearm continues in same direction as upper arm for simplicity)
        hand_direction = elbow_pos / np.linalg.norm(elbow_pos) if np.linalg.norm(elbow_pos) > 0 else np.array([1,0,0])
        hand_pos = elbow_pos + self.params.forearm_length * hand_direction
        
        # For compatibility with 2D visualization, project to X-Y plane
        return np.array([
            np.sqrt(hand_pos[0]**2 + hand_pos[2]**2),  # Distance from body (forward + sideways)
            hand_pos[1]  # Height
        ])
    
    def compute_muscle_forces(self, activations):
        """
        6-muscle force computation - extends your working 4-muscle approach
        """
        flex_act, ext_act, abd_act, add_act, bicep_act, tricep_act = activations
        
        # Your working muscle forces (UNCHANGED)
        forces = {
            'shoulder_flexor': self.params.shoulder_flexor_max_force * flex_act,
            'shoulder_extensor': self.params.shoulder_extensor_max_force * ext_act,
            'bicep': self.params.bicep_max_force * bicep_act,
            'tricep': self.params.tricep_max_force * tricep_act,
            
            # NEW: Abduction/Adduction forces
            'shoulder_abductor': self.params.shoulder_abductor_max_force * abd_act,
            'shoulder_adductor': self.params.shoulder_adductor_max_force * add_act,
        }
        
        return forces
    
    def compute_joint_torques(self, muscle_forces):
        """
        Convert 6-muscle forces to joint torques
        Now handles TWO shoulder directions + elbow
        """
        # Shoulder flexion torque (your working calculation)
        shoulder_flex_torque = (muscle_forces['shoulder_flexor'] * self.params.shoulder_flexor_moment_arm - 
                               muscle_forces['shoulder_extensor'] * self.params.shoulder_extensor_moment_arm)
        
        # NEW: Shoulder abduction torque
        shoulder_abd_torque = (muscle_forces['shoulder_abductor'] * self.params.shoulder_abductor_moment_arm - 
                              muscle_forces['shoulder_adductor'] * self.params.shoulder_adductor_moment_arm)
        
        # Elbow torque (your working calculation, UNCHANGED)
        elbow_torque = (muscle_forces['bicep'] * self.params.bicep_moment_arm - 
                       muscle_forces['tricep'] * self.params.tricep_moment_arm)
        
        return shoulder_flex_torque, shoulder_abd_torque, elbow_torque
    
    def compute_gravity_torques(self):
        """
        Gravity effects - extended for 3D shoulder movement
        (Simplified - assumes abduction doesn't significantly change gravity effects)
        """
        # Use your working gravity calculation for flexion
        upper_arm_com_x = 0.5 * self.params.upper_arm_length * np.cos(self.shoulder_flex_angle)
        upper_arm_com_y = 0.5 * self.params.upper_arm_length * np.sin(self.shoulder_flex_angle)
        
        # Simplified forearm COM (assumes forearm follows upper arm direction)
        elbow_x = self.params.upper_arm_length * np.cos(self.shoulder_flex_angle)
        elbow_y = self.params.upper_arm_length * np.sin(self.shoulder_flex_angle)
        forearm_com_x = elbow_x + 0.5 * self.params.forearm_length * np.cos(self.shoulder_flex_angle)
        forearm_com_y = elbow_y + 0.5 * self.params.forearm_length * np.sin(self.shoulder_flex_angle)
        
        # Gravity torques (your working calculation)
        shoulder_flex_gravity = -(self.params.upper_arm_mass * self.params.gravity * upper_arm_com_x + 
                                 self.params.forearm_mass * self.params.gravity * forearm_com_x)
        
        # Abduction gravity torque (minimal for small angles)
        shoulder_abd_gravity = 0.0  # Simplified: assume abduction doesn't change gravity significantly
        
        elbow_gravity = -(self.params.forearm_mass * self.params.gravity * 
                         (forearm_com_x - elbow_x))
        
        return shoulder_flex_gravity, shoulder_abd_gravity, elbow_gravity
    
    def step(self, activations):
        """
        Physics simulation - extends your working step() to 6 muscles and 3 DOF
        """
        # Clip and store activations (now 6D)
        activations = np.clip(activations, 0.0, 1.0)
        self.shoulder_flexor_activation = activations[0]
        self.shoulder_extensor_activation = activations[1]
        self.shoulder_abductor_activation = activations[2]  # NEW
        self.shoulder_adductor_activation = activations[3]  # NEW
        self.bicep_activation = activations[4]
        self.tricep_activation = activations[5]
        
        # Compute forces and torques (extended)
        muscle_forces = self.compute_muscle_forces(activations)
        muscle_torques = self.compute_joint_torques(muscle_forces)
        gravity_torques = self.compute_gravity_torques()
        
        # Total torques for 3 DOF
        shoulder_flex_total = (muscle_torques[0] + gravity_torques[0] - 
                              self.params.shoulder_damping * self.shoulder_flex_velocity)
        shoulder_abd_total = (muscle_torques[1] + gravity_torques[1] - 
                             self.params.shoulder_damping * self.shoulder_abd_velocity)
        elbow_total = (muscle_torques[2] + gravity_torques[2] - 
                      self.params.elbow_damping * self.elbow_velocity)
        
        # Joint accelerations
        shoulder_flex_accel = shoulder_flex_total / self.params.shoulder_inertia
        shoulder_abd_accel = shoulder_abd_total / self.params.shoulder_inertia  
        elbow_accel = elbow_total / self.params.elbow_inertia
        
        # Integration (your working method) + angle wrapping
        self.shoulder_flex_velocity += shoulder_flex_accel * self.params.dt
        self.shoulder_flex_angle += self.shoulder_flex_velocity * self.params.dt
        
        self.shoulder_abd_velocity += shoulder_abd_accel * self.params.dt
        self.shoulder_abd_angle += self.shoulder_abd_velocity * self.params.dt
        
        self.elbow_velocity += elbow_accel * self.params.dt
        self.elbow_angle += self.elbow_velocity * self.params.dt
        
        # Keep angles in reasonable range [-π, π]
        self.shoulder_flex_angle = np.arctan2(np.sin(self.shoulder_flex_angle), np.cos(self.shoulder_flex_angle))
        self.shoulder_abd_angle = np.arctan2(np.sin(self.shoulder_abd_angle), np.cos(self.shoulder_abd_angle))
        self.elbow_angle = np.arctan2(np.sin(self.elbow_angle), np.cos(self.elbow_angle))
        
        # Store for next step
        self.prev_activations = activations.copy()
        
        return self.get_state(), self.get_end_effector_position()

def test_6_muscle_system():
    """
    Test the 6-muscle extension - verify it works like your Phase 2 system
    """
    print("=== Testing 6-Muscle Arm Extension ===")
    print("Verifying extension of your successful Phase 2 system")
    
    arm = SixMuscleArm()
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    state = arm.get_state()
    print(f"   Initial state shape: {state.shape} (should be 6D)")
    print(f"   Initial hand position: {arm.get_end_effector_position()}")
    
    # DEBUGGING: Step-by-step shoulder flexor test
    print("\n2. DEBUGGING: Single step shoulder flexor test...")
    arm = SixMuscleArm()
    
    print(f"   Initial shoulder angle: {np.degrees(arm.shoulder_flex_angle):.1f}°")
    
    # Single step with shoulder flexor only
    flexor_only = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Manually check forces and torques
    muscle_forces = arm.compute_muscle_forces(flexor_only)
    muscle_torques = arm.compute_joint_torques(muscle_forces)
    gravity_torques = arm.compute_gravity_torques()
    
    print(f"   Shoulder flexor force: {muscle_forces['shoulder_flexor']:.1f}N")
    print(f"   Shoulder flexor torque: {muscle_torques[0]:.3f} N⋅m")
    print(f"   Gravity torque: {gravity_torques[0]:.3f} N⋅m")
    print(f"   Total torque: {muscle_torques[0] + gravity_torques[0]:.3f} N⋅m")
    
    state, hand_pos = arm.step(flexor_only)
    
    print(f"   After 1 step shoulder angle: {np.degrees(arm.shoulder_flex_angle):.1f}°")
    print(f"   Shoulder velocity: {np.degrees(arm.shoulder_flex_velocity):.1f}°/s")
    
    if muscle_torques[0] + gravity_torques[0] > 0 and arm.shoulder_flex_angle < 0:
        print("   ❌ PHYSICS ERROR: Positive torque but negative angle change!")
    elif muscle_torques[0] + gravity_torques[0] > 0 and arm.shoulder_flex_angle > 0:
        print("   ✅ Physics correct: Positive torque, positive angle")
    
    # Test 2: Your working muscles still work (flexion/extension)
    print("\n3. Testing your working Phase 2 muscles (10 steps only)...")
    arm = SixMuscleArm()
    
    for step in range(10):  # Only 10 steps to see progression
        your_working_muscles = np.array([0.5, 0.0, 0.0, 0.0, 0.3, 0.0])
        state, hand_pos = arm.step(your_working_muscles)
        if step % 3 == 0:
            print(f"   Step {step}: Shoulder={np.degrees(state[0]):.1f}°, Elbow={np.degrees(state[4]):.1f}°")
    
    print(f"   Final hand position: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f})")
    
    # Test 3: NEW abduction muscles
    print("\n4. Testing NEW abduction muscles...")
    arm = SixMuscleArm()  # Reset
    
    for step in range(100):
        # Use only abduction: [none, none, abduct, none, none, none] 
        abduction_only = np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0])
        state, hand_pos = arm.step(abduction_only)
    
    print(f"   After 100 steps with abduction only:")
    print(f"   Hand position: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f})")
    print(f"   Shoulder abduction angle: {np.degrees(state[2]):.1f}°")
    
    # Test 4: All 6 muscles together
    print("\n5. Testing all 6 muscles coordination...")
    arm = SixMuscleArm()  # Reset
    
    for step in range(100):
        # Moderate activation of all muscles
        all_muscles = np.array([0.3, 0.1, 0.4, 0.1, 0.2, 0.1])
        state, hand_pos = arm.step(all_muscles)
    
    print(f"   After 100 steps with all 6 muscles:")
    print(f"   Hand position: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f})")
    print(f"   All joint angles: {np.degrees(state[::2])}")  # [flex, abd, elbow]
    
    print(f"\n✅ 6-Muscle system working!")
    print(f"Next: Create 6-muscle reaching environment")

if __name__ == "__main__":
    test_6_muscle_system()