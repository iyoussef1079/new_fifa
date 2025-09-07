# balance_env.py (Corrected for Observation Shape)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer
import os

class BalanceEnv(gym.Env):
    """
    Custom Gymnasium environment for the musculoskeletal single-leg balance task.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, model_path, render_mode=None):
        super().__init__()

        # --- Load the MuJoCo Model ---
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            raise IOError(f"Error loading model: {e}")
            
        self.data = mujoco.MjData(self.model)
        
        # --- Get Body and Sensor IDs for easy access ---
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.foot_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'foot_sensor')

        try:
            self.r_foot_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'r_foot')
            self.l_foot_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'l_foot') 
        except:
            print("Warning: Foot sensors not found, using contact-based CoP calculation")
            self.r_foot_sensor_id = -1
            self.l_foot_sensor_id = -1

        # --- Define Reward Weights ---
        self.reward_weight_pose = 2.0
        self.reward_weight_sway = -1.0
        self.reward_weight_effort = -0.1
        self.alive_bonus = 0.5

        # --- Define Action and Observation Spaces ---
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # --- THIS SECTION IS CORRECTED ---
        # Define the observation space size based on the actual components we use.
        qpos_size = len(self.data.qpos)
        qvel_size = len(self.data.qvel)
        
        # Get the dimension of our specific foot sensor, not all sensors.
        foot_force_dim = self.model.sensor_dim[self.foot_sensor_id]
        
        observation_size = qpos_size + qvel_size + foot_force_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64
        )
        
        # --- Rendering Setup ---
        self.render_mode = render_mode
        self.viewer = None
        
        print(f"BalanceEnv initialized. Observation space size: {observation_size}, Action space size: {self.model.nu}")

        self.all_muscles, self.trunk_muscles, self.leg_muscles, self.other_muscles = self._identify_all_muscles()

    def _get_obs(self):
        """
        Constructs the observation vector from the simulation state.
        """
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # --- THIS SECTION IS CORRECTED ---
        # Read sensor data robustly using its specific address and dimension.
        sensor_adr = self.model.sensor_adr[self.foot_sensor_id]
        sensor_dim = self.model.sensor_dim[self.foot_sensor_id]
        foot_force = self.data.sensordata[sensor_adr : sensor_adr + sensor_dim]

        # Concatenate all parts into a single observation vector
        return np.concatenate([qpos, qvel, foot_force])
    
    def _calculate_cop_com(self):
        """Calculate Center of Pressure from contacts and Center of Mass"""
        
        # Center of Pressure from ground contacts
        total_force = 0
        weighted_position = np.zeros(2)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_pos = contact.pos
            
            # Check if contact is with ground (low z-position)
            if contact_pos[2] < 0.15:  # Ground level threshold
                # Get contact force
                force_6d = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force_6d)
                normal_force = abs(force_6d[2])  # Z-component
                
                if normal_force > 0.5:  # Force threshold  
                    total_force += normal_force
                    weighted_position += contact_pos[:2] * normal_force
        
        if total_force > 0:
            cop = weighted_position / total_force
        else:
            # Default to pelvis projection if no contacts
            pelvis_pos = self.data.xpos[self.pelvis_body_id]
            cop = pelvis_pos[:2]
        
        # Center of Mass (simplified - pelvis projection)
        pelvis_pos = self.data.xpos[self.pelvis_body_id]
        com = pelvis_pos[:2]
        
        return cop, com
    
    def _compute_full_foot_contact_reward(self):
        """
        Reward full foot contact (heel + midfoot + forefoot)
        Penalize toe-only balancing
        """
        
        # Get all foot sensor values using the IDs from your debug output
        r_heel = self.data.sensordata[0]      # r_heel
        r_foot = self.data.sensordata[1]      # r_foot  
        r_forefoot = self.data.sensordata[2]  # r_forefoot
        r_toes = self.data.sensordata[3]      # r_toes
        
        l_heel = self.data.sensordata[4]      # l_heel
        l_foot = self.data.sensordata[5]      # l_foot
        l_forefoot = self.data.sensordata[6]  # l_forefoot
        l_toes = self.data.sensordata[7]      # l_toes
        
        contact_threshold = 1.0  # Based on your sensor values showing 13-188
        
        # Count active regions for each foot
        r_heel_contact = r_heel > contact_threshold
        r_mid_contact = r_foot > contact_threshold
        r_fore_contact = r_forefoot > contact_threshold
        r_toes_contact = r_toes > contact_threshold
        
        l_heel_contact = l_heel > contact_threshold
        l_mid_contact = l_foot > contact_threshold  
        l_fore_contact = l_forefoot > contact_threshold
        l_toes_contact = l_toes > contact_threshold
        
        # Reward full foot contact, penalize toe-only
        r_foot_score = 0
        l_foot_score = 0
        
        # Right foot scoring
        if r_heel_contact and r_mid_contact and r_fore_contact:
            r_foot_score = 1.0  # Perfect full foot contact
        elif (r_heel_contact or r_mid_contact) and not r_toes_contact:
            r_foot_score = 0.7  # Good heel/mid contact, no toes
        elif r_toes_contact and not (r_heel_contact or r_mid_contact):
            r_foot_score = -0.5  # Bad: toe-only contact
        else:
            r_foot_score = 0.0
        
        # Left foot scoring (same logic)
        if l_heel_contact and l_mid_contact and l_fore_contact:
            l_foot_score = 1.0
        elif (l_heel_contact or l_mid_contact) and not l_toes_contact:
            l_foot_score = 0.7
        elif l_toes_contact and not (l_heel_contact or l_mid_contact):
            l_foot_score = -0.5
        else:
            l_foot_score = 0.0
        
        total_reward = (r_foot_score + l_foot_score) * 0.5
        
        return total_reward
    
    def _compute_core_tonic_reward(self, action):
        """
        Reward tonic activation in key postural stabilizers only
        Focus on muscles that maintain spinal stability and prevent 'floppy' posture
        """
        
        # Key postural muscles that should maintain baseline activation
        key_stabilizers = {
            # Major abdominals
            'rect_abd_r': 0.10,    # 8% baseline - main abs
            'rect_abd_l': 0.10,    
            
            # External obliques (just middle segments)
            'EO3_r': 0.1,         # 6% baseline - mid trunk
            'EO4_r': 0.1,         
            'EO3_l': 0.1,
            'EO4_l': 0.1,
            
            # Internal obliques (just middle segments)  
            'IO3_r': 0.1,         # 5% baseline - deep stability
            'IO4_r': 0.1,
            'IO3_l': 0.1,
            'IO4_l': 0.1,
            
            # Key multifidus (lumbar region - most important for posture)
            'MF_m3s_r': 0.1,      # 7% baseline - L3 level
            'MF_m4s_r': 0.1,      # L4 level  
            'MF_m5s_r': 0.1,      # L5 level
            'MF_m3s_l': 0.1,
            'MF_m4s_l': 0.1, 
            'MF_m5s_l': 0.1,
            
            # Hip flexors (already identified)
            'iliacus_r': 0.08,     # 4% baseline
            'psoas_r': 0.10,       # 6% baseline
            'iliacus_l': 0.08,
            'psoas_l': 0.10,
        }
        
        total_reward = 0
        muscle_count = 0
        
        for muscle_name, target_activation in key_stabilizers.items():
            try:
                muscle_idx = self.all_muscles.index(muscle_name)
                current_activation = action[muscle_idx]
                
                # Gaussian reward around target (sigma = 0.03)
                error = abs(current_activation - target_activation)
                muscle_reward = np.exp(-(error / 0.03)**2)
                
                total_reward += muscle_reward
                muscle_count += 1
                
            except ValueError:
                # Muscle not found - skip
                continue
        
        # Average reward across found muscles
        if muscle_count > 0:
            return total_reward / muscle_count
        else:
            return 0.0

    def _calculate_reward(self, action):
        # Existing rewards
        pelvis_orientation_matrix = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        up_vector = pelvis_orientation_matrix[:, 2]
        vertical_alignment = up_vector[1]
        pose_reward = np.exp(-10.0 * (1.0 - vertical_alignment)**2)

        pelvis_lin_vel = self.data.cvel[self.pelvis_body_id, 3:]
        pelvis_ang_vel = self.data.cvel[self.pelvis_body_id, :3]
        sway_penalty = np.sum(np.square(pelvis_lin_vel)) + 0.5 * np.sum(np.square(pelvis_ang_vel))

        effort_penalty = np.sum(np.square(action))

        # CoP-CoM balance reward
        cop, com = self._calculate_cop_com()
        cop_com_distance = np.linalg.norm(cop - com)
        balance_reward = np.exp(-5.0 * cop_com_distance)
        
        # Full foot contact reward
        foot_contact_reward = self._compute_full_foot_contact_reward()
        
        # NEW: Core tonic activation reward
        core_tonic_reward = self._compute_core_tonic_reward(action)
        
        total_reward = (self.reward_weight_pose * pose_reward +
                        self.reward_weight_sway * sway_penalty +
                        self.reward_weight_effort * effort_penalty +
                        0.5 * balance_reward +
                        0.6 * foot_contact_reward +      # Aggressive foot contact
                        0.3 * core_tonic_reward +        # NEW: Core stability
                        self.alive_bonus)
        
        return total_reward

    def _is_terminated(self):
        """
        Checks if the episode should be terminated (e.g., the model has fallen).
        """
        pelvis_height = self.data.xpos[self.pelvis_body_id, 2]
        return pelvis_height < 0.7
    
    def _debug_foot_geometry(self):
        """Debug ALL contacts, sensors, AND ground setup"""
        print(f"\n--- MODEL POSITION DEBUG ---")
        
        # Check pelvis height (root body position)
        pelvis_pos = self.data.qpos[:3]  # First 3 are usually root position
        print(f"Pelvis position: {pelvis_pos}")
        print(f"Pelvis height (z): {pelvis_pos[2]:.4f}")
        
        # Check if there's a ground plane
        print(f"\nTotal geoms in model: {self.model.ngeom}")
        for i in range(min(5, self.model.ngeom)):  # Check first 5 geoms
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            geom_type = self.model.geom_type[i]
            geom_pos = self.model.geom_pos[i]
            print(f"  Geom {i}: '{geom_name}', type={geom_type}, pos={geom_pos}")
        
        print(f"\n--- CONTACTS DEBUG ---")
        print(f"Total contacts: {self.data.ncon}")
        
        if self.data.ncon == 0:
            print("ERROR: No contacts detected!")
            
            # Check foot positions
            try:
                r_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'calcn_r')
                l_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'calcn_l')
                
                r_foot_pos = self.data.xpos[r_foot_body_id]
                l_foot_pos = self.data.xpos[l_foot_body_id]
                
                print(f"Right foot position: {r_foot_pos}")
                print(f"Left foot position: {l_foot_pos}")
                print(f"Right foot height: {r_foot_pos[2]:.4f}")
                print(f"Left foot height: {l_foot_pos[2]:.4f}")
                
            except Exception as e:
                print(f"Could not get foot positions: {e}")
        
        # Rest of your existing sensor code...
        print(f"\n--- FOOT SENSORS DEBUG ---")
        foot_sensors = ['r_heel', 'r_foot', 'r_forefoot', 'r_toes',
                        'l_heel', 'l_foot', 'l_forefoot', 'l_toes']
        
        for sensor_name in foot_sensors:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                sensor_value = self.data.sensordata[sensor_id]
                contact_status = "CONTACT" if sensor_value > 0.01 else "no contact"
                print(f"  {sensor_name}: Value={sensor_value:.4f} ({contact_status})")
            except Exception as e:
                print(f"  {sensor_name}: ERROR - {e}")
        
        print("--- END DEBUG ---\n")

    def _identify_all_muscles(self):
        """List ALL muscles in the model with better core muscle identification"""
        
        print(f"\n--- ALL MUSCLES IN MODEL ---")
        print(f"Total actuators: {self.model.nu}")
        
        all_muscles = []
        trunk_muscles = []
        leg_muscles = []
        other_muscles = []
        
        for i in range(self.model.nu):
            try:
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name:
                    name = actuator_name.decode() if isinstance(actuator_name, bytes) else actuator_name
                    all_muscles.append(name)
                    
                    # Enhanced categorization for trunk muscles
                    if any(pattern in name.lower() for pattern in 
                        ['rect_abd',        # Rectus abdominis
                            'eo1', 'eo2', 'eo3', 'eo4', 'eo5', 'eo6',  # External obliques
                            'io1', 'io2', 'io3', 'io4', 'io5', 'io6',  # Internal obliques  
                            'mf_',             # Multifidus
                            'ql_',             # Quadratus lumborum
                            'ps_',             # Psoas components
                            'il_',             # Iliocostalis  
                            'ltpt_', 'ltpl_',  # Longissimus
                            'psoas', 'iliacus']):
                        trunk_muscles.append(name)
                    elif any(pattern in name.lower() for pattern in 
                            ['quad', 'ham', 'gas', 'soleus', 'tibialis', 'glut', 'hip', 
                            'vastus', 'rectus_fem', 'biceps_fem', 'semiten', 'semimem',
                            'bflh', 'bfsh', 'recfem', 'vasint', 'vaslat', 'vasmed',
                            'glmax', 'glmed', 'glmin', 'add']):
                        leg_muscles.append(name)
                    else:
                        other_muscles.append(name)
                        
            except:
                name = f"actuator_{i}"
                all_muscles.append(name)
                other_muscles.append(name)
        
        print(f"\nTRUNK/CORE MUSCLES ({len(trunk_muscles)}):")
        for i, muscle in enumerate(trunk_muscles):
            print(f"  {i:3d}: {muscle}")
        
        print(f"\nLEG MUSCLES ({len(leg_muscles)}):")  
        for i, muscle in enumerate(leg_muscles[:20]):  # Show first 20 only
            print(f"  {i:3d}: {muscle}")
        if len(leg_muscles) > 20:
            print(f"  ... and {len(leg_muscles)-20} more")
        
        print(f"--- END MUSCLE CATEGORIZATION ---\n")
        
        return all_muscles, trunk_muscles, leg_muscles, other_muscles


    def reset(self, seed=None, options=None):
        """
        Resets the environment and adds slight randomization to the initial pose.
        """
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        qpos_init = self.model.qpos0.copy()
        qvel_init = np.zeros(self.model.nv)
        
        noise = self.np_random.uniform(low=-0.05, high=0.05, size=qpos_init.shape)
        qpos_init[7:] += noise[7:]
        
        self.data.qpos[:] = qpos_init
        self.data.qvel[:] = qvel_init
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()
        
        # self._debug_foot_geometry()
            
        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        self.data.ctrl[:] = action
        
        n_frames = 10
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_obs()
        reward = self._calculate_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None