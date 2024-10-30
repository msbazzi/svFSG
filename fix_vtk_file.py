import vtk
import numpy as np
import os

files = ["ring_outlet.vtp", "ring_inlet.vtp", "wall_inlet.vtp", "wall_outlet.vtp", "wall_inner.vtp", "wall_outer.vtp"]

# Path to the VTK file
simulationInputDirectory = "/home/bazzi/repo/svFSG"

vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-complete.mesh.vtu")

reader = vtk.vtkXMLUnstructuredGridReader()  # for vtu files
reader.SetFileName(vtk_file)
reader.Update()

# Get the output of the reader
output = reader.GetOutput()

# Get the number of cells
numCells = output.GetNumberOfCells()

# Add CellStructureID array
cell_structure_id_array = vtk.vtkIntArray()
cell_structure_id_array.SetName("CellStructureID")
cell_structure_id_array.SetNumberOfComponents(1)
cell_structure_id_array.SetNumberOfTuples(numCells)
for i in range(numCells):
    cell_structure_id_array.SetValue(i, i + 1)
output.GetCellData().AddArray(cell_structure_id_array)

# Check if 'GlobalElementID' data exists and is of type Int32
global_element_id_array = output.GetCellData().GetArray('GlobalElementID')
if global_element_id_array is not None:
    # Create a new array with type Int32 and copy the data
    new_global_element_id_array = vtk.vtkIntArray()
    new_global_element_id_array.SetName("GlobalElementID")
    new_global_element_id_array.SetNumberOfComponents(1)
    new_global_element_id_array.SetNumberOfTuples(global_element_id_array.GetNumberOfTuples())
    for i in range(global_element_id_array.GetNumberOfTuples()):
        new_global_element_id_array.SetValue(i, int(global_element_id_array.GetValue(i)))
    output.GetCellData().RemoveArray('GlobalElementID')  # Remove the old array
    output.GetCellData().AddArray(new_global_element_id_array)  # Add the new array

# Additional logging for debugging
print("VTK mesh updated successfully.")
print(f"Number of cells: {output.GetNumberOfCells()}")
print(f"Number of points: {output.GetNumberOfPoints()}")

# Save the updated VTK file
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(vtk_file)
writer.SetInputData(output)
writer.Write()

# Process surface files
for filename in files:
    vtk_file = os.path.join(simulationInputDirectory, "mesh/solid-mesh-complete/mesh-surfaces", filename)

    reader = vtk.vtkXMLPolyDataReader()  # for vtp files
    reader.SetFileName(vtk_file)
    reader.Update()

    # Get the output of the reader
    output = reader.GetOutput()

    # Get the number of cells
    numCells = output.GetNumberOfCells()

    # Add CellStructureID array
    cell_structure_id_array = vtk.vtkIntArray()
    cell_structure_id_array.SetName("CellStructureID")
    cell_structure_id_array.SetNumberOfComponents(1)
    cell_structure_id_array.SetNumberOfTuples(numCells)
    for i in range(numCells):
        cell_structure_id_array.SetValue(i, i + 1)
    output.GetCellData().AddArray(cell_structure_id_array)

    # Check if 'GlobalElementID' and 'GlobalNodeID' data exists
    global_element_id_array = output.GetCellData().GetArray('GlobalElementID')
    if global_element_id_array is not None:
        # Create a new array with type Int32 and copy the data
        new_global_element_id_array = vtk.vtkIntArray()
        new_global_element_id_array.SetName("GlobalElementID")
        new_global_element_id_array.SetNumberOfComponents(1)
        new_global_element_id_array.SetNumberOfTuples(global_element_id_array.GetNumberOfTuples())
        for i in range(global_element_id_array.GetNumberOfTuples()):
            new_global_element_id_array.SetValue(i, int(global_element_id_array.GetValue(i)))
        output.GetCellData().RemoveArray('GlobalElementID')  # Remove the old array
        output.GetCellData().AddArray(new_global_element_id_array)  # Add the new array

    global_node_id_array = output.GetPointData().GetArray('GlobalNodeID')
    if global_node_id_array is not None:
        # Create a new array with type Int32 and copy the data
        new_global_node_id_array = vtk.vtkIntArray()
        new_global_node_id_array.SetName("GlobalNodeID")
        new_global_node_id_array.SetNumberOfComponents(1)
        new_global_node_id_array.SetNumberOfTuples(global_node_id_array.GetNumberOfTuples())
        for i in range(global_node_id_array.GetNumberOfTuples()):
            new_global_node_id_array.SetValue(i, int(global_node_id_array.GetValue(i)))
        output.GetPointData().RemoveArray('GlobalNodeID')  # Remove the old array
        output.GetPointData().AddArray(new_global_node_id_array)  # Add the new array

    # Additional logging for debugging
    print(f"VTK surface mesh {filename} updated successfully.")
    print(f"Number of cells: {output.GetNumberOfCells()}")
    print(f"Number of points: {output.GetNumberOfPoints()}")

    # Save the updated VTK file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_file)
    writer.SetInputData(output)
    writer.Write()
