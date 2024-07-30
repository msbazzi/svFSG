import vtk
import numpy as np
import pyvista as pv
import os


# Path to the VTK file
simulationInputDirectory = "/home/bazzi/repo/svFSG"
vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-complete.mesh.vtu")
#vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-surfaces/ring_outlet.vtp")


reader = vtk.vtkXMLUnstructuredGridReader() # for vtu files    
#reader = vtk.vtkXMLPolyDataReader() # for vtp files
reader.SetFileName(vtk_file)
reader.Update()
# Get the output of the reader
output = reader.GetOutput()

# Assuming vol is your vtkUnstructuredGrid object
numCells = output.GetNumberOfCells()

# Add CellStructureID array
cell_structure_id_array = pv.convert_array(np.linspace(1, numCells, numCells).astype(int), name="CellStructureID")
output.GetCellData().AddArray(cell_structure_id_array)

# Check if 'GlobalElementID' data exists and is of type Int32
global_element_id_array = output.GetCellData().GetArray('GlobalElementID')
global_element_id_array = pv.convert_array(np.array(global_element_id_array).astype(np.int32), name="GlobalElementID")
output.GetCellData().AddArray(global_element_id_array)


# Additional logging for debugging
print("VTK mesh updated successfully.")
print(f"Number of cells: {output.GetNumberOfCells()}")
print(f"Number of points: {output.GetNumberOfPoints()}")

# Save the updated VTK file
writer = vtk.vtkXMLUnstructuredGridWriter()
#writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(vtk_file)
writer.SetInputData(output)
writer.Write()

