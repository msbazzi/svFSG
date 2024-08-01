import os
import subprocess
import numpy as np
import pyvista as pv

numProcessorsSolid = 1
simulationExecutable = "/home/bazzi/repo/svFSIplus/build/svFSI-build/bin/svFSI"
simulationInputDirectory = "/home/bazzi/repo/svFSG"
vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-complete.mesh.vtu")

# Check if the VTK file exists
if not os.path.isfile(vtk_file):
    raise FileNotFoundError(f"Required VTK file not found: {vtk_file}")



# Run the simulation and capture output
try:
    result = subprocess.run(
        ["mpirun", "-n", str(numProcessorsSolid), simulationExecutable, f"{simulationInputDirectory}/FolderSimulationInputFiles/svFSI.xml"],
        check=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error running simulation: {e}")
    print(f"Standard Output: {e.stdout}")
    print(f"Standard Error: {e.stderr}")