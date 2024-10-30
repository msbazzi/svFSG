import os
import subprocess
import numpy as np
import pyvista as pv

numProcessorsSolid = 1
simulationExecutable = "/home/bazzi/repo/svFSIplus/build/svFSI-build/bin/svFSI"
simulationInputDirectory = "/home/bazzi/repo/svFSG"
vtk_file = os.path.join(simulationInputDirectory, "pipe/wall-mesh-complete/mesh-complete.mesh.vtu")


#vtk_file = os.path.join(simulationInputDirectory, "pipe/wall-mesh-complete/mesh-complete.mesh.vtu")

# Check if the VTK file exists
if not os.path.isfile(vtk_file):
    raise FileNotFoundError(f"Required VTK file not found: {vtk_file}")

'''
# Load the VTK file
try:
    mesh = pv.read(vtk_file)
    print(f"Successfully read the VTK file: {vtk_file}")
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of cells: {mesh.n_cells}")
    print("Point data arrays:", mesh.point_data.keys())
    print("Cell data arrays:", mesh.cell_data.keys())

    if 'varWallProps' in mesh.cell_data:
        var_wall_props = mesh.cell_data['varWallProps']
        print(f"varWallProps shape: {var_wall_props.shape}")
        num_columns = var_wall_props.shape[1] if var_wall_props.ndim > 1 else 1
        print(f"Number of columns in varWallProps: {num_columns}")
    else:
        print("varWallProps data array not found in the VTK file.")  
except Exception as e:
    print(f"Error reading VTK file: {e}") '''

# Run the simulation and capture output
try:
    result = subprocess.run(
        ["mpirun", "-n", str(numProcessorsSolid), simulationExecutable, f"{simulationInputDirectory}/FolderSimulationInputFiles/svFSI_pipe.xml"],
        check=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error running simulation: {e}")
    print(f"Standard Output: {e.stdout}")
    print(f"Standard Error: {e.stderr}")