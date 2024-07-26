import os
import vtk
import subprocess

numProcessorsSolid = 1
simulationExecutable = "/home/bazzi/repo/svFSIplus/build/svFSI-build/bin/svFSI"
simulationInputDirectory = "/home/bazzi/repo/svFSG"
vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-complete.mesh.vtu")

#Check if the VTK file exists
if not os.path.isfile(vtk_file):
    raise FileNotFoundError(f"Required VTK file not found: {vtk_file}")

# Read the VTK file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtk_file)
reader.Update()
# Get the output of the reader
output = reader.GetOutput()

# Check if 'GlobalElementID' data exists and is of type Int32
global_element_id_array = output.GetCellData().GetArray('GlobalElementID')
if not global_element_id_array:
    raise ValueError(f"'GlobalElementID' data not found in VTK file: {vtk_file}")
    vol.GetCellData().AddArray(global_element_id_array)
#
 #   raise ValueError(f"'GlobalElementID' data is not of type Int32 in VTK file: {vtk_file}")
# Additional logging for debugging
print("VTK mesh updated successfully.")
print(f"Number of cells: {vol.GetNumberOfCells()}")
print(f"Number of points: {vol.GetNumberOfPoints()}")

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