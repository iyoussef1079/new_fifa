import os
import mujoco
import mujoco.viewer
import time

def view_fullbody_model():
    """
    Simple script to load and visualize the MyoSuite full body model.
    """
    
    # Path to the full body model XML
    model_path = os.path.join('./myo_sim', 'body', 'myobody.xml')

    # body: myobody.xml myobody_simpleupper.xml
    # arm: myoarm.xml
    
    try:
        print(f"Loading model from: {model_path}")
        
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print(f"Model loaded successfully!")
        print(f"- Bodies: {model.nbody}")
        print(f"- Joints: {model.njnt}")
        print(f"- Actuators: {model.nu}")
        print(f"- Degrees of freedom: {model.nv}")
        
        # Reset to the keyframe pose (if defined in the XML)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
            
        # Forward dynamics to update the simulation state
        mujoco.mj_forward(model, data)
        
        print("\nOpening viewer... Press ESC to exit.")
        
        # Launch the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Keep the viewer open
            while viewer.is_running():
                # Step the simulation slowly
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)  # Small delay for smooth viewing
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTips:")
        print("1. Make sure the model path is correct")
        print("2. Check that all included files exist in the myo_sim directory")
        print("3. Verify the XML file structure")

if __name__ == "__main__":
    view_fullbody_model()