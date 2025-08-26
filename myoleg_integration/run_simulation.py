# run_simulation.py

import mujoco
import mujoco.viewer
import time
import os

import numpy as np
from balance_env import BalanceEnv

# --- Step 1: Load the Musculoskeletal Model ---
# Ensure you have downloaded the MyoLeg model from the MyoSim repository
# and placed it in an 'assets' folder.
# https://github.com/MyoHub/myo_sim
model_path = os.path.join('./myo_sim', 'leg', 'myoLegs.xml')

env = BalanceEnv(model_path=model_path, render_mode="human")

# --- Step 2: Run a Simple "Do Nothing" Test ---
# This test verifies that the environment loop works correctly.
# We apply zero action, so the model should just collapse.
print("\n--- Running 'Do Nothing' Test ---")
observation, info = env.reset()
terminated = False
episode_length = 0

while not terminated and episode_length < 600:
    # Apply zero action (no muscle activation)
    action = np.zeros(env.action_space.shape)
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    episode_length += 1
    
    # A small delay to make the visualization easier to follow
    time.sleep(0.01) 

print("Test finished. Model collapsed as expected.")
env.close()