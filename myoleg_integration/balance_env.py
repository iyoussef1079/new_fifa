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

    def _calculate_reward(self, action):
        """
        Calculates the multi-component reward for the current state and action.
        """
        pelvis_orientation_matrix = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        up_vector = pelvis_orientation_matrix[:, 2]
        vertical_alignment = up_vector[1]
        pose_reward = np.exp(-10.0 * (1.0 - vertical_alignment)**2)

        pelvis_lin_vel = self.data.cvel[self.pelvis_body_id, 3:]
        pelvis_ang_vel = self.data.cvel[self.pelvis_body_id, :3]
        sway_penalty = np.sum(np.square(pelvis_lin_vel)) + 0.5 * np.sum(np.square(pelvis_ang_vel))

        effort_penalty = np.sum(np.square(action))

        total_reward = (self.reward_weight_pose * pose_reward +
                        self.reward_weight_sway * sway_penalty +
                        self.reward_weight_effort * effort_penalty +
                        self.alive_bonus)
        
        return total_reward

    def _is_terminated(self):
        """
        Checks if the episode should be terminated (e.g., the model has fallen).
        """
        pelvis_height = self.data.xpos[self.pelvis_body_id, 2]
        return pelvis_height < 0.7

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