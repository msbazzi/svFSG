import numpy as np
import pyvista as pv

# Path to the VTK file
vtk_file = "/home/bazzi/repo/svFSIplus/tests/cases/gr/mesh/solid-mesh-complete/mesh-complete.mesh.vtp"

# Load the VTK file
mesh = pv.read(vtk_file)

# Check if 'GlobalElementID' exists
if 'GlobalElementID' not in mesh.cell_data:
    raise ValueError(f"'GlobalElementID' data not found in VTK file: {vtk_file}")

# Get the 'GlobalElementID' array
global_element_id = mesh.cell_data['GlobalElementID']

# Check the data type and correct if necessary
if global_element_id.dtype != np.int32:
    print(f"Correcting 'GlobalElementID' data type from {global_element_id.dtype} to Int32")
    global_element_id = global_element_id.astype(np.int32)
    mesh.cell_data['GlobalElementID'] = global_element_id

# Save the modified VTK file
mesh.save(vtk_file)
print(f"Saved corrected VTK file: {vtk_file}")