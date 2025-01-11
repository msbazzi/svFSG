import meshio
import numpy as np

# Load VTU files
mesh1 = meshio.read("/home/bazzi/repo/svFSG/mesh/solid-mesh-complete/mesh-complete.mesh.vtu")
mesh2 = meshio.read("/home/bazzi/repo/svFSGnew/svFSG/mesh/solid-mesh-complete/mesh-complete.mesh.vtu")

# Extract array data from the files
# Replace 'array_name' with the actual name of the array you want to compare
array1 = mesh1.cell_data['varWallProps'][0]
array2 = mesh2.cell_data['varWallProps'][0]

# Check if the arrays are the same size
if len(array1) == len(array2):
    # Perform comparison
    difference = np.isclose(array1, array2, atol=1e-6)  # Use a tolerance level if needed
    diff_array = np.where(difference, 0, 1)  # Create diff array with 0 if equal, 1 if not

    if np.all(difference):
        print("Arrays are identical within the tolerance level.")
    else:
        print("Arrays differ.")
        diff_positions = np.where(diff_array == 1)[0]  # Get positions where elements differ
        print(f"Positions with differences: {diff_positions}")
else:
    print("Arrays have different sizes, cannot compare directly.")