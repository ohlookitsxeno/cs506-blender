import bpy
import pickle
import numpy as np

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Path to the pickled file
pickle_file_path = "/home/xeno/Documents/cs506/cs506-blender/neuralnetwork/hidden_space_and_points.pkl"

# Load the pickled data
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)

# Create materials
def create_material(name, base_color, opacity):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add Principled BSDF
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.location = (0, 0)
    output.location = (200, 0)
    links.new(shader.outputs['BSDF'], output.inputs['Surface'])

    # Configure opacity and blend mode
    shader.inputs['Alpha'].default_value = opacity
    mat.blend_method = 'BLEND' if opacity < 1.0 else 'OPAQUE'

    # Set base color
    shader.inputs['Base Color'].default_value = (*base_color, 1.0)

    return mat


# Extract the original points and hidden space trajectory
original_points = np.array(data["original_points"])  # Shape (n, 2)
hidden_space_trajectory = np.array(data["hidden_space_trajectory"])  # Shape (epochs, n, hidden_dim)

# Print information to verify
print(f"Loaded {original_points.shape[0]} original points with shape {original_points.shape}")
print(f"Loaded hidden space trajectory with shape {hidden_space_trajectory.shape}")


# Create materials for pass and fail
pass_material = create_material("Pass_Material", (0.0, 1.0, 0.0), 1.0)  # Green for pass
fail_material = create_material("Fail_Material", (1.0, 0.0, 0.0), 1.0)  # Red for fail

# Compute pass_fail for original points
cw1, cw2, cb = -1, 1, -1
sw1, sw2, sb = -1, 1, 1

C = np.array([0 if cw1 * x[0] + cw2 * x[1] + cb >= 0 else 1 for x in original_points])
S = np.array([0 if sw1 * x[0] + sw2 * x[1] + sb >= 0 else 1 for x in original_points])
pass_fail = C ^ S

# Add points to the Blender scene and assign materials
objects = []  # Store references to the created objects
for i, point in enumerate(original_points):
    # Create a sphere for each point
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(point[0], point[1], 0))
    obj = bpy.context.object  # Reference to the newly created object

    # Assign material based on pass_fail
    if pass_fail[i] == 0:
        obj.data.materials.append(pass_material)
    else:
        obj.data.materials.append(fail_material)

    # Set the original point as the first keyframe
    obj.location = (point[0], point[1], 0)
    obj.keyframe_insert(data_path="location", frame=1)

    objects.append(obj)

# Add keyframes for each epoch
start_frame = 10  # Start animating at frame 10
frame_step = 10  # Frames between each epoch

for epoch_index, hidden_positions in enumerate(hidden_space_trajectory):
    current_frame = start_frame + epoch_index * frame_step
    for i, obj in enumerate(objects):
        # Update object location based on hidden space position
        hidden_position = hidden_positions[i]
        obj.location = (hidden_position[0], hidden_position[1], hidden_position[2] if hidden_position.shape[0] > 2 else 0)
        obj.keyframe_insert(data_path="location", frame=current_frame)

print(f"Animation keyframes added for {len(hidden_space_trajectory)} epochs.")
