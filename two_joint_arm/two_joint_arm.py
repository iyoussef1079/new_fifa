"""
Phase 2.1: Minimal 2D Arm System
Extends Phase 1 elbow system with ONE shoulder joint
Focus: Test gravity integration and multi-joint coordination
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TwoJointArmParams:
    """Parameters for 2D arm system - conservative starting values"""
    
    # Segment lengths (human-like proportions)
    upper_arm_length: float = 0.3   # m (shoulder to elbow)
    forearm_length: float = 0.25    # m (elbow to hand)
    
    # Segment masses (realistic but simple)
    upper_arm_mass: float = 2.5     # kg
    forearm_mass: float = 1.5       # kg
    
    # Joint inertias (from your validated elbow values)
    shoulder_inertia: float = 0.8   # kg*m^2 (larger than elbow)
    elbow_inertia: float = 0.5      # kg*m^2 (your working value)
    
    # Joint damping (from your validated values)
    shoulder_damping: float = 3.0   # N*m*s/rad (higher for stability)
    elbow_damping: float = 2.0      # N*m*s/rad (your working value)
    
    # Muscle forces (FIXED: stronger shoulder muscles to overcome gravity)
    shoulder_flexor_max_force: float = 800.0    # N (much stronger to lift arm+forearm)
    shoulder_extensor_max_force: float = 900.0  # N
    bicep_max_force: float = 200.0              # N (your working value)
    tricep_max_force: float = 250.0             # N (your working value)
    
    # Moment arms (balanced torques like your elbow)
    shoulder_flexor_moment_arm: float = 0.06    # m
    shoulder_extensor_moment_arm: float = 0.05  # m (balanced torque ~15 N⋅m)
    bicep_moment_arm: float = 0.05              # m (your working value)
    tricep_moment_arm: float = 0.04             # m (your working value)
    
    # Physics
    dt: float = 0.01        # s (100 Hz like Phase 1)
    gravity: float = 9.81   # m/s^2 (NEW: critical for 2D)

class TwoJointArm:
    """
    Minimal 2D arm: shoulder + elbow joints
    Direct extension of your working SimpleElbowSystem
    """
    
    def __init__(self, params=None):
        self.params = params or TwoJointArmParams()
        
        # Joint states (angles in radians)
        self.shoulder_angle = 0.0    # 0 = arm horizontal forward
        self.shoulder_velocity = 0.0
        self.elbow_angle = 0.0       # 0 = forearm aligned with upper arm
        self.elbow_velocity = 0.0
        
        # Muscle activations (0-1, like your elbow system)
        self.shoulder_flexor_activation = 0.0   # Lifts arm up
        self.shoulder_extensor_activation = 0.0 # Pulls arm down
        self.bicep_activation = 0.0             # Your working bicep
        self.tricep_activation = 0.0            # Your working tricep
        
        # Track previous actions (for smoothness penalties)
        self.prev_activations = np.zeros(4)
        
    def get_state(self):
        """Current system state"""
        return np.array([
            self.shoulder_angle, self.shoulder_velocity,
            self.elbow_angle, self.elbow_velocity
        ])
    
    def get_activations(self):
        """Current muscle activations"""
        return np.array([
            self.shoulder_flexor_activation,
            self.shoulder_extensor_activation, 
            self.bicep_activation,
            self.tricep_activation
        ])
    
    def get_end_effector_position(self):
        """Hand position in 2D space (critical for reaching tasks)"""
        # Shoulder joint at origin
        shoulder_x = 0.0
        shoulder_y = 0.0
        
        # Elbow position
        elbow_x = shoulder_x + self.params.upper_arm_length * np.cos(self.shoulder_angle)
        elbow_y = shoulder_y + self.params.upper_arm_length * np.sin(self.shoulder_angle)
        
        # Hand position (end effector)
        hand_x = elbow_x + self.params.forearm_length * np.cos(self.shoulder_angle + self.elbow_angle)
        hand_y = elbow_y + self.params.forearm_length * np.sin(self.shoulder_angle + self.elbow_angle)
        
        return np.array([hand_x, hand_y])
    
    def compute_muscle_forces(self, activations):
        """
        Compute muscle forces using your validated Hill model approach
        Direct extension from your working elbow system
        """
        shoulder_flexor_act, shoulder_extensor_act, bicep_act, tricep_act = activations
        
        # Use your working muscle force calculation (simplified Hill model)
        shoulder_flexor_force = self.params.shoulder_flexor_max_force * shoulder_flexor_act
        shoulder_extensor_force = self.params.shoulder_extensor_max_force * shoulder_extensor_act
        bicep_force = self.params.bicep_max_force * bicep_act
        tricep_force = self.params.tricep_max_force * tricep_act
        
        return {
            'shoulder_flexor': shoulder_flexor_force,
            'shoulder_extensor': shoulder_extensor_force,
            'bicep': bicep_force,
            'tricep': tricep_force
        }
    
    def compute_joint_torques(self, muscle_forces):
        """
        Convert muscle forces to joint torques
        Uses your validated moment arm approach
        """
        # Shoulder torque (flexor positive, extensor negative)
        shoulder_torque = (muscle_forces['shoulder_flexor'] * self.params.shoulder_flexor_moment_arm - 
                          muscle_forces['shoulder_extensor'] * self.params.shoulder_extensor_moment_arm)
        
        # Elbow torque (your working calculation)
        elbow_torque = (muscle_forces['bicep'] * self.params.bicep_moment_arm - 
                       muscle_forces['tricep'] * self.params.tricep_moment_arm)
        
        return shoulder_torque, elbow_torque
    
    def compute_gravity_torques(self):
        """
        NEW: Gravity effects on joints (critical for realistic 2D behavior)
        """
        # Upper arm center of mass
        upper_arm_com_x = 0.5 * self.params.upper_arm_length * np.cos(self.shoulder_angle)
        upper_arm_com_y = 0.5 * self.params.upper_arm_length * np.sin(self.shoulder_angle)
        
        # Forearm center of mass (relative to elbow)
        elbow_x = self.params.upper_arm_length * np.cos(self.shoulder_angle)
        elbow_y = self.params.upper_arm_length * np.sin(self.shoulder_angle)
        forearm_com_x = elbow_x + 0.5 * self.params.forearm_length * np.cos(self.shoulder_angle + self.elbow_angle)
        forearm_com_y = elbow_y + 0.5 * self.params.forearm_length * np.sin(self.shoulder_angle + self.elbow_angle)
        
        # Gravity torques (negative y direction)
        shoulder_gravity_torque = -(self.params.upper_arm_mass * self.params.gravity * upper_arm_com_x + 
                                   self.params.forearm_mass * self.params.gravity * forearm_com_x)
        
        elbow_gravity_torque = -(self.params.forearm_mass * self.params.gravity * 
                                (forearm_com_x - elbow_x))
        
        return shoulder_gravity_torque, elbow_gravity_torque
    
    def step(self, activations):
        """
        Physics simulation step - extends your working elbow step() method
        """
        # Clip and store activations
        activations = np.clip(activations, 0.0, 1.0)
        self.shoulder_flexor_activation = activations[0]
        self.shoulder_extensor_activation = activations[1] 
        self.bicep_activation = activations[2]
        self.tricep_activation = activations[3]
        
        # Compute forces and torques
        muscle_forces = self.compute_muscle_forces(activations)
        muscle_torques = self.compute_joint_torques(muscle_forces)
        gravity_torques = self.compute_gravity_torques()
        
        # Total torques
        shoulder_total_torque = (muscle_torques[0] + gravity_torques[0] - 
                                self.params.shoulder_damping * self.shoulder_velocity)
        elbow_total_torque = (muscle_torques[1] + gravity_torques[1] - 
                             self.params.elbow_damping * self.elbow_velocity)
        
        # Joint accelerations (F = ma)
        shoulder_acceleration = shoulder_total_torque / self.params.shoulder_inertia
        elbow_acceleration = elbow_total_torque / self.params.elbow_inertia
        
        # Integrate (Euler method like your elbow system)
        self.shoulder_velocity += shoulder_acceleration * self.params.dt
        self.shoulder_angle += self.shoulder_velocity * self.params.dt
        
        self.elbow_velocity += elbow_acceleration * self.params.dt
        self.elbow_angle += self.elbow_velocity * self.params.dt
        
        # Store for next step
        self.prev_activations = activations.copy()
        
        return self.get_state(), self.get_end_effector_position()

def test_gravity_effects():
    """
    CRITICAL TEST: Validate gravity integration
    This is the key new component that could break everything
    """
    print("Testing gravity effects on 2D arm...")
    
    arm = TwoJointArm()
    states = []
    positions = []
    
    # Test: Start arm horizontal, no muscle activation
    # Should fall under gravity
    for step in range(500):  # 5 seconds
        no_activation = np.zeros(4)  # No muscle forces
        state, hand_pos = arm.step(no_activation)
        states.append(state.copy())
        positions.append(hand_pos.copy())
        
        if step % 100 == 0:
            print(f"Step {step}: Shoulder={state[0]:.3f}°, Elbow={state[2]:.3f}°, Hand=({hand_pos[0]:.3f}, {hand_pos[1]:.3f})")
    
    # Analyze results
    final_shoulder = np.degrees(states[-1][0])
    final_hand_y = positions[-1][1]
    
    print(f"\nResults after 5 seconds:")
    print(f"Final shoulder angle: {final_shoulder:.1f}° (should be negative - arm fell down)")
    print(f"Final hand height: {final_hand_y:.3f}m (should be negative - below shoulder)")
    
    if final_shoulder < -10 and final_hand_y < -0.1:
        print("✅ GRAVITY TEST PASSED: Arm falls naturally under gravity")
        return True
    else:
        print("❌ GRAVITY TEST FAILED: Check physics parameters")
        return False

def test_muscle_control():
    """
    TEST: Validate muscle control extends from your working elbow system
    """
    print("\nTesting muscle control...")
    
    arm = TwoJointArm()
    
    # Test 1: Shoulder flexor only (should lift arm up)
    print("Test 1: Shoulder flexor activation")
    for step in range(200):
        shoulder_flexor_only = np.array([0.5, 0.0, 0.0, 0.0])  # 50% shoulder flexor
        state, hand_pos = arm.step(shoulder_flexor_only)
        
        if step % 50 == 0:
            print(f"  Step {step}: Shoulder={np.degrees(state[0]):.1f}°, Hand height={hand_pos[1]:.3f}m")
    
    shoulder_lift = np.degrees(state[0])
    if shoulder_lift > 5:
        print("✅ Shoulder flexor working: Lifted arm up")
    else:
        print("❌ Shoulder flexor problem: Arm didn't lift")
    
    # Reset arm
    arm = TwoJointArm()
    
    # Test 2: Your working elbow muscles (should still work)
    print("\nTest 2: Elbow bicep activation (your working muscle)")
    for step in range(200):
        bicep_only = np.array([0.0, 0.0, 0.5, 0.0])  # 50% bicep
        state, hand_pos = arm.step(bicep_only)
        
        if step % 50 == 0:
            print(f"  Step {step}: Elbow={np.degrees(state[2]):.1f}°, Hand pos=({hand_pos[0]:.3f}, {hand_pos[1]:.3f})")
    
    elbow_flex = np.degrees(state[2])
    if elbow_flex > 5:
        print("✅ Elbow bicep working: Your Phase 1 muscle control preserved")
        return True
    else:
        print("❌ Elbow bicep problem: Phase 1 functionality lost")
        return False

if __name__ == "__main__":
    print("=== Phase 2.1: Two-Joint Arm System ===")
    print("Minimal extension of your working Phase 1 elbow system")
    
    # Run critical validation tests
    gravity_ok = test_gravity_effects()
    muscle_ok = test_muscle_control()
    
    if gravity_ok and muscle_ok:
        print("\n🎉 SUCCESS: Ready for Phase 2.2 (Gymnasium environment)")
        print("Next step: Create TwoJointReachingEnv extending your ElbowTrackingEnv")
    else:
        print("\n⚠️  ISSUES FOUND: Fix physics before proceeding")
        print("Debug: Check parameters, validate one component at a time")