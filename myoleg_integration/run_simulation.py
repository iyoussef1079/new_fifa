# run_simulation.py

import os
from balance_env import BalanceEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Setup Directories ---
# It's good practice to have separate directories for models and logs
log_dir = "logs/"
model_dir = "leg_chain_torso_chain"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# --- Step 1: Initialize the Custom Environment ---
model_path = os.path.join('./myo_sim', 'body', 'myobody.xml')
# Set render_mode to None for fast training
env = BalanceEnv(model_path=model_path, render_mode="human") 

# --- Step 2: Setup Checkpoint Callback ---
# This will save the model every 50,000 steps
# It also saves the replay buffer, which is critical for resuming SAC training
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=model_dir,
  name_prefix="sac_balance_model",
  save_replay_buffer=True,
  save_vecnormalize=True, # Save environment statistics if applicable
)

# --- Step 3: Initialize or Load the Agent ---
# Path to the latest model checkpoint
latest_model_path = os.path.join(model_dir, "sac_balance_model_interrupted.zip") # Example path

if os.path.exists(latest_model_path):
    print(f"--- Loading existing model from {latest_model_path} ---")
    model = SAC.load(latest_model_path, env=env)
    # If you saved the replay buffer, you can load it now
    # model.load_replay_buffer(os.path.join(model_dir, "sac_replay_buffer.pkl"))
else:
    print("--- No existing model found, initializing a new one ---")
    model = SAC('MlpPolicy', env, verbose=1, device="auto")

# --- Step 4: Train the Agent ---
# The callback will save checkpoints automatically. You can stop this script
# with Ctrl+C at any time and resume from the last saved model.
print("\n--- Starting training ---")
try:
    # Set reset_num_timesteps=False to continue the timestep counter from the loaded model
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback, reset_num_timesteps=False)
    print("\n--- Training complete ---")
    model.save(os.path.join(model_dir, "sac_balance_model_final"))
except KeyboardInterrupt:
    print("\n--- Training interrupted by user. Saving model... ---")
    model.save(os.path.join(model_dir, "sac_balance_model_interrupted"))
finally:
    env.close()