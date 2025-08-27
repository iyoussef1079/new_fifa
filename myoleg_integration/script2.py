import argparse
import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R
import re

def transform_vector(vec_str, attribute_name):
    """
    Reflects a 3D vector string across the XY plane.
    The transformation rule depends on whether the vector represents a position or a rotation axis.
    """
    if not vec_str:
        return vec_str
    
    vec = np.fromstring(vec_str, sep=' ')
    
    if attribute_name == 'axis':
        # For joint axes, the reflection of the body's coordinate system (across XY plane)
        # means the sense of rotation is flipped. To counteract this and keep the motion
        # intuitive (e.g., positive angle still means flexion), we reflect the axis
        # across the Z-axis.
        # Rule: [x, y, z] -> [-x, -y, z]
        if vec.size == 3:
            vec[0] *= -1
            vec[1] *= -1
    else: # Handles 'pos' and 'fromto'
        # For positions, we perform a direct reflection across the XY plane.
        # Rule: [x, y, z] -> [x, y, -z]
        if vec.size == 3: # pos
            vec[2] *= -1
        elif vec.size == 6: # fromto
            vec[2] *= -1
            vec[5] *= -1
            
    return ' '.join(map(str, vec))

def transform_euler(euler_str):
    """
    Reflects Euler angles across the XY plane. Based on the provided example,
    this primarily involves flipping the rotation around the Z-axis.
    Note: This is a simplification. For complex, chained rotations, converting
    to quaternions or matrices before transformation is more robust.
    """
    if not euler_str:
        return euler_str
    vec = np.fromstring(euler_str, sep=' ')
    if vec.size == 3:
        # Rule from example: [x, y, z] -> [x, y, -z]
        vec[2] *= -1
    return ' '.join(map(str, vec))


def transform_quaternion(quat_str):
    """
    Applies a principled mirroring transformation to an orientation quaternion
    for reflection across the XY plane.
    """
    if not quat_str:
        return qu_str

    # MuJoCo quat is [w, x, y, z]
    q_orig_mujoco = np.fromstring(quat_str, sep=' ')
    
    # Scipy expects [x, y, z, w]
    q_orig_scipy = np.array([q_orig_mujoco[1], q_orig_mujoco[2], q_orig_mujoco[3], q_orig_mujoco[0]])

    # Convert to a rotation matrix
    M_orig = R.from_quat(q_orig_scipy).as_matrix()

    # To mirror across the XY plane and preserve "handedness" of the coordinate frame,
    # we apply a composite transformation matrix. This reflects the local Y and Z axes.
    # T_orient = diag(1, -1, -1)
    T_orient = np.diag([1, -1, -1])

    # Apply the transformation: M_new = T_orient * M_orig
    M_new = T_orient @ M_orig

    # Convert the new matrix back to a quaternion
    q_new_scipy = R.from_matrix(M_new).as_quat()

    # Convert back to MuJoCo quaternion format [w, x, y, z]
    q_new_mujoco = np.array([q_new_scipy[3], q_new_scipy[0], q_new_scipy[1], q_new_scipy[2]])

    return ' '.join(map(str, q_new_mujoco))

def replace_name(name, append_if_neutral=True):
    """
    Replaces right-hand identifiers with left-hand ones.
    If append_if_neutral is True and no identifier is found, it appends '_l'
    to create a unique left-side name.
    """
    if not name:
        return name
    
    original_name = name
    new_name = name
    was_replaced = False

    # Define patterns for right-side identifiers
    patterns = {
        r'right': 'left',
        r'_r': '_l',
        r'\.r': '.l'
    }

    for pattern, replacement in patterns.items():
        if re.search(pattern, new_name, flags=re.IGNORECASE):
            new_name = re.sub(pattern, replacement, new_name, flags=re.IGNORECASE)
            was_replaced = True

    # Handle prefixes like "R." -> "L." or "RSJC" -> "LSJC"
    # The \b ensures we only match at the beginning of a word.
    if re.search(r'\bR([A-Z0-9.])', new_name):
        new_name = re.sub(r'\bR([A-Z0-9.])', r'L\1', new_name)
        was_replaced = True

    # If no specific right-side pattern was matched and the flag is set,
    # append "_l" to the original name to make it unique.
    if not was_replaced and append_if_neutral:
        new_name = f"{original_name}_l"

    return new_name

def mirror_model(input_file, output_file):
    """
    Parses a MuJoCo XML file and applies mirroring transformations to convert
    a right-sided model to a left-sided one across the XY-plane.
    """
    print(f"Parsing input file: {input_file}")
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(input_file, parser)
    root = tree.getroot()

    # Attributes to transform
    vector_attrs = ['pos', 'axis', 'fromto']
    name_attrs = ['name', 'childclass', 'class', 'mesh']

    print("\n--- Starting Model Transformation ---")
    # Iterate over every element in the XML tree
    for elem in root.xpath('//*'):
        tag_name = elem.tag
        original_name_attr = elem.get('name', 'N/A')
        print(f"\nProcessing <{tag_name}> with name='{original_name_attr}'")

        # 1. Rename element and any other relevant attributes
        for attr_name in name_attrs:
            if attr_name in elem.attrib:
                original_value = elem.attrib[attr_name]
                
                # Determine if we should append '_l' for neutral names.
                # We only do this for the 'name' attribute itself, which defines an element.
                # For references like 'class', 'childclass', or 'mesh', we only
                # want to replace existing 'right' identifiers, not create new names.
                should_append = (attr_name == 'name')
                
                new_value = replace_name(original_value, append_if_neutral=should_append)
                
                if original_value != new_value:
                    print(f"  - Renaming '{attr_name}': '{original_value}' -> '{new_value}'")
                    elem.attrib[attr_name] = new_value

        # 2. Transform vector attributes
        for attr in vector_attrs:
            if attr in elem.attrib:
                original_value = elem.attrib[attr]
                new_value = transform_vector(original_value, attr)
                if original_value != new_value:
                    print(f"  - Transforming '{attr}': '{original_value}' -> '{new_value}'")
                    elem.attrib[attr] = new_value

        # 3. Transform orientation attributes
        if 'quat' in elem.attrib:
            original_value = elem.attrib['quat']
            new_value = transform_quaternion(original_value)
            if original_value != new_value:
                print(f"  - Transforming 'quat': '{original_value}' -> '{new_value}'")
                elem.attrib['quat'] = new_value
        elif 'euler' in elem.attrib:
            original_value = elem.attrib['euler']
            new_value = transform_euler(original_value)
            if original_value != new_value:
                print(f"  - Transforming 'euler': '{original_value}' -> '{new_value}'")
                elem.attrib['euler'] = new_value

    print("\n--- Model Transformation Complete ---")
    # Write the transformed tree to the output file
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"Successfully converted model and saved to: {output_file}")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        prog="MuJoCo Model Mirror",
        description="Converts a right-sided MuJoCo XML model to its left-sided equivalent by mirroring it across the XY plane.",
        epilog="This script handles positions, orientations (quaternions/euler), joint axes, and names."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input XML file (the right-sided model)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the new output XML file (the left-sided model)."
    )

    args = parser.parse_args()

    try:
        mirror_model(args.input_file, args.output_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")