# evaluate_model.py

import os
import time
from balance_env import BalanceEnv
from stable_baselines3 import SAC

# --- Parameters ---
model_dir = "RL_agent_weight_after_train"
# IMPORTANT: Change this to the specific model checkpoint you want to view
model_to_load = "sac_balance_model_interrupted.zip" 
model_path = os.path.join(model_dir, 'leg_chain_torso_chain', model_to_load)

# --- Load Environment and Model ---
xml_path = os.path.join('./myo_sim', 'body', 'myobody.xml')
# IMPORTANT: Set render_mode to "human" to see the visualization
eval_env = BalanceEnv(model_path=xml_path, render_mode="human")

print(f"Loading model from: {model_path}")
model = SAC.load(model_path, env=eval_env)

# --- Run Evaluation Loop ---
obs, info = eval_env.reset()
for _ in range(2000): # Run for 2000 steps
    # Use deterministic=True for the model's best guess action
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    if terminated:
        print("Episode finished. Resetting.")
        obs, info = eval_env.reset()
    
    time.sleep(0.01) # Small delay to make it watchable

eval_env.close()