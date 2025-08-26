# balance_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
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

        # --- Define Action and Observation Spaces ---
        # Action space: 80 muscle activations, continuous values between 0 and 1.
        # This aligns with the MyoLeg model which has 80 muscle actuators.[3, 4]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Observation space (simplified for now)
        # We will include: joint positions (qpos), joint velocities (qvel),
        # and torso orientation (quaternion) and angular velocity.
        # This will be expanded later.
        observation_size = len(self.data.qpos) + len(self.data.qvel)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64
        )
        
        # --- Rendering Setup ---
        self.render_mode = render_mode
        self.viewer = None
        
        print("BalanceEnv initialized successfully.")

    def _get_obs(self):
        """Helper function to construct the observation vector."""
        # Concatenate joint positions and velocities
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed) # Required for gym compatibility
        
        # Reset the simulation data
        mujoco.mj_resetData(self.model, self.data)
        
        # TODO: Implement a more robust initial state, e.g., a slightly perturbed standing pose.
        # For now, we start from the default pose.
        
        observation = self._get_obs()
        info = {} # Placeholder for additional info

        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # --- Apply Action ---
        # The action is a vector of muscle activations.
        self.data.ctrl[:] = action
        
        # --- Step the Simulation ---
        # We step multiple times for each 'step' call to match control frequency
        # with simulation frequency. This is a common practice.
        n_frames = 10 # Number of physics steps per environment step
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

        # --- Get Observation ---
        observation = self._get_obs()

        # --- Calculate Reward (Placeholder) ---
        # TODO: Implement the multi-component reward function.
        reward = 1.0 # Placeholder: simple "alive" bonus

        # --- Check for Termination (Placeholder) ---
        # TODO: Implement termination conditions (e.g., pelvis height falls below a threshold).
        terminated = False
        
        # --- Other Info ---
        truncated = False # Not time-limited for now
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            return

        if self.viewer is None:
            # Lazily initialize the viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Sync the viewer with the current simulation state
        self.viewer.sync()

    def close(self):
        """
        Clean up the environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None