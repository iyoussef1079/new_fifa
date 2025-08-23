"""
MuJoCo Setup and Installation Guide
Phase 3 Week 1: Getting MuJoCo + RL working
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for MuJoCo RL training"""
    
    print("Installing MuJoCo + RL requirements...")
    
    packages = [
        "mujoco>=3.0.0",           # MuJoCo physics engine
        "stable-baselines3[extra]", # RL algorithms
        "gymnasium[mujoco]",        # Environment interface
        "matplotlib",               # Plotting
        "numpy",                   # Numerical computing
        "torch"                    # PyTorch (for SAC)
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_mujoco_installation():
    """Test MuJoCo installation"""
    print("\nTesting MuJoCo installation...")
    
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        
        import gymnasium as gym
        print(f"✓ Gymnasium version: {gym.__version__}")
        
        from stable_baselines3 import SAC
        print(f"✓ Stable-Baselines3 imported successfully")
        
        # Test basic MuJoCo functionality
        print("\nTesting basic MuJoCo functionality...")
        
        # Create simple test model
        test_xml = """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 1"/>
            <body name="box" pos="0 0 1">
              <joint type="free"/>
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(test_xml)
        data = mujoco.MjData(model)
        
        # Run a few simulation steps
        for _ in range(10):
            mujoco.mj_step(model, data)
        
        print("✓ MuJoCo simulation test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ MuJoCo test failed: {e}")
        return False

def create_project_structure():
    """Create project directory structure"""
    print("\nCreating project structure...")
    
    directories = [
        "logs/sac_mujoco_balance",
        "models",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def verify_files():
    """Verify all required files are present"""
    print("\nVerifying project files...")
    
    required_files = [
        "simple_balance.xml",
        "mujoco_balance_env.py", 
        "train_mujoco_balance.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_quick_test():
    """Run quick environment test"""
    print("\nRunning quick environment test...")
    
    try:
        # This should work if everything is set up correctly
        from mujoco_balance_env import MuJoCoBalanceEnv
        
        print("Creating environment...")
        env = MuJoCoBalanceEnv("simple_balance.xml")
        
        print("Testing reset...")
        obs, info = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        print("Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful, reward: {reward:.2f}")
        
        env.close()
        print("✓ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("MuJoCo Balance RL Setup")
    print("=" * 40)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements. Please check error messages above.")
        return
    
    # Step 2: Test MuJoCo
    if not test_mujoco_installation():
        print("\n❌ MuJoCo installation test failed.")
        return
    
    # Step 3: Create project structure  
    create_project_structure()
    
    # Step 4: Verify files
    if not verify_files():
        print("\n❌ Missing required files. Please ensure all files are in current directory.")
        return
    
    # Step 5: Quick test
    if not run_quick_test():
        print("\n❌ Environment test failed.")
        return
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete! Ready to train.")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Run: python train_mujoco_balance.py")
    print("2. Expected: >80% success rate after 50k steps")
    print("3. Compare with Phase 2 results")
    print("\nThis validates: MuJoCo + muscles + SAC approach")

if __name__ == "__main__":
    main()