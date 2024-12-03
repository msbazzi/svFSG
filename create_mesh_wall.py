import vtk
import pyvista as pv
import numpy as np
import os
from scipy.spatial import KDTree

def extract_faces(unstructured_grid, output_file):
    # Use vtkGeometryFilter to extract the surface
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()

    # Get the output surface as vtkPolyData
    surface = geometry_filter.GetOutput()

        # Check if GlobalNodeID exists
    global_node_id_array = unstructured_grid.GetPointData().GetArray("GlobalNodeID")
    if global_node_id_array:
        print("GlobalNodeID found. Propagating to the surface...")
        surface.GetPointData().AddArray(global_node_id_array)
    else:
        print("GlobalNodeID not found in the volumetric mesh.")

    # Write the surface to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(surface)
    writer.Write()

    print(f"Extracted surface written to: {output_file}")
    return surface


# Read the .vtp file
def point_normals(surface):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(surface)
    reader.Update()
    polydata = reader.GetOutput()

    # Validate input data
    num_points = polydata.GetNumberOfPoints()
    num_cells = polydata.GetNumberOfCells()

    print(f"Number of points: {num_points}")
    print(f"Number of cells: {num_cells}")

    if num_points == 0 or num_cells == 0:
        print("The mesh is empty or invalid. Check the .vtp file.")
        exit()  # Stop execution if the file is invalid

    # Check for pre-computed normals
    if polydata.GetPointData().GetNormals():
        print("Normals already exist in the file.")
        normals = polydata.GetPointData().GetNormals()
    else:
        print("Normals not found in the file. Computing normals...")
        # Compute normals
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()  # Focus only on point normals
        normals_filter.Update()

        # Get the output with computed normals
        polydata = normals_filter.GetOutput()

    # Validate normals after computation
    normals = polydata.GetPointData().GetNormals()
    if normals is None:
        print("Failed to compute normals.")
        exit()  # Stop if normals computation failed
    else:
        print(f"Normals computed successfully. Number of points with normals: {polydata.GetNumberOfPoints()}")

    # Convert to PyVista
    mesh = pv.wrap(polydata)

    if mesh is None or mesh.n_points == 0:
        print("The mesh is empty after wrapping with PyVista.")
        exit()

    if "Normals" not in mesh.point_data:
        print("Normals not found in the PyVista mesh.")
        exit()

    # # Plot the normals
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color="white", show_edges=True, opacity=0.8)  # Add surface
    # plotter.add_arrows(mesh.points, mesh["Normals"], mag=0.1, color="red")  # Add normals
    # plotter.show()
    normal_point = np.zeros((len(mesh.points),3))
    for i in range(len(mesh.points)):
        normal_point[i,:] = normals.GetTuple(i)


    return polydata, normals, normal_point  

def plot_points(points):
    # Create a PyVista point cloud from the NumPy array
    point_cloud = pv.PolyData(points)

    # Plot the points
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="blue", point_size=5, render_points_as_spheres=True)
    plotter.show()


def generate_prismatic_mesh_vtu(polydata, normals, thickness, radial_layers, output_file):
    # Create an array to store the new points
    
    num_points = polydata.GetNumberOfPoints()
    all_points = np.zeros(((radial_layers+1)*num_points, 7))
    ids = np.zeros(((radial_layers+1)*num_points),  dtype=int)
    prism_connectivity = []
    global_id_counter = 0

    for i in range(num_points):

        normal = normals.GetTuple(i)
        global_id_array = polydata.GetPointData().GetArray("GlobalNodeID").GetValue(i)
        points_coordinates = polydata.GetPoint(i)
        all_points[i,0]= global_id_array
        all_points[i,1] = points_coordinates[0]
        all_points[i,2] = points_coordinates[1]
        all_points[i,3] = points_coordinates[2]
        all_points[i,4] = normal[0]
        all_points[i,5] = normal[1]
        all_points[i,6] = normal[2]
        ids[i] = 1


    num_cells = polydata.GetNumberOfCells()
    cell_point_ids = [global_id_counter + i for i in range(num_points)]
    
    # Generate new points for each radial layer
    for i in range(radial_layers):
        c = i+1
        factor = c / (radial_layers)  # Scale factor for radial layers
        
        for j in range(num_points):

            point = polydata.GetPoint(j)
            normal = normals.GetTuple(j)

            # Calculate the offset point
            xPt = point[0] + factor * thickness * (-normal[0])
            yPt = point[1] + factor * thickness * (-normal[1])
            zPt = point[2] + factor * thickness * (-normal[2])
            
            all_points[num_points*c+j,0] = all_points[num_points*i+j,0]+num_points*c
            all_points[num_points*c+j,1] = xPt
            all_points[num_points*c+j,2] = yPt
            all_points[num_points*c+j,3] = zPt
            all_points[num_points*c+j,4] = normal[0]
            all_points[num_points*c+j,5] = normal[1]
            all_points[num_points*c+j,6] = normal[2]
        
            if i == radial_layers - 1:
                ids[num_points*c+j] = 2
    
    # Generate connectivity for prismatic cells
    for i in range(num_cells):
        
        cell = polydata.GetCell(i)
        num_cell_points = cell.GetNumberOfPoints()
        cell_point_ids = [cell.GetPointId(j) for j in range(num_cell_points)]

        for j in range(radial_layers - 1):
            base_layer = j * num_points
            next_layer = (j + 1) * num_points

            # Create prismatic connectivity
            if num_cell_points==3: # Triangular cells
                p0 = cell_point_ids[0] + base_layer
                p1 = cell_point_ids[1] + base_layer
                p2 = cell_point_ids[2] + base_layer

                p3 = cell_point_ids[0] + next_layer
                p4 = cell_point_ids[1] + next_layer
                p5 = cell_point_ids[2] + next_layer
                # Add a prism cell: 6 points per cell
                prism_connectivity.append([p0, p1, p2, p3, p4, p5])
             
            elif len(num_cell_points)==4: # Quadrilateral cells
                print("Quadrilateral cells is not implemented yet")              
                    # Convert all_points to a NumPy array
    
    
   
    # Convert points to VTK format
    all_points_coordinates  = all_points[:,1:4]
    #plot_points(all_points_coordinates)
    vtk_points = vtk.vtkPoints()
    for point in all_points_coordinates:
        vtk_points.InsertNextPoint(point)

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(vtk_points)

    # Add prism cells to the unstructured grid
    for prism in prism_connectivity:
        vtk_cell = vtk.vtkIdList()
        for pid in prism:
            vtk_cell.InsertNextId(pid)
        unstructured_grid.InsertNextCell(vtk.VTK_WEDGE, vtk_cell)
    global_ids = all_points[:,0] # Global IDs for surface points
    unstructured_grid.GetPointData().AddArray(pv.convert_array((ids).astype(np.int32),name="FaceId"))
    unstructured_grid.GetPointData().AddArray(pv.convert_array((global_ids).astype(np.int32),name="GlobalNodeID"))
    # Write to VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()
    
    print(f"VTU file written to: {output_file}")

    return unstructured_grid, ids, all_points

def main():
    surface = "/home/bazzi/repo/svFSG/pipe/wall-mesh-complete_marisa/mesh-surfaces/inner.vtp"
    output_file="/home/bazzi/repo/svFSG/pipe/wall-mesh-complete_marisa/wall.vtu"
    faces_file = "/home/bazzi/repo/svFSG/pipe/wall-mesh-complete_marisa/wall_faces.vtp"
    output_dir = "/home/bazzi/repo/svFSG/pipe/wall-mesh-complete_marisa/split_faces"
    polydata, normals, normal_point = point_normals(surface)
    vol, ids,all_points = generate_prismatic_mesh_vtu(polydata, normals, 0.5, 4, output_file)
    # Print normals and their corresponding points
   
    # Extract faces and save them to a file
    extracted_faces = extract_faces(vol, faces_file)
    print(f"Extracted surface has {extracted_faces.GetNumberOfPoints()} points and {extracted_faces.GetNumberOfCells()} cells.")
    mesh = pv.read(faces_file)
    mesh_points = mesh.points

    global_ids = all_points[:,0] # Global IDs for surface points
 
    tree = KDTree(all_points[:,1:4])
    distances, indices = tree.query(mesh_points)

    polydata2, normals2, normal_point2 = point_normals(faces_file)
    # Add a scalar field based on the Z coordinate (or another relevant axis)
    
    surface_points = all_points[indices, 1:4]
    tree2 = KDTree(mesh_points)
    # Step 2: Query the tree for mesh_points
    distances, indices2 = tree2.query(surface_points)
    normals_surface = normal_point2[indices2]
    mesh["Normals"] = normals_surface   # Add Z-normal as a scalar field
    z_normals = normals_surface[:, 2]  # Extract Z-component of normals
    mesh["Z_Normals"] = z_normals  # Add Z-normal as a scalar field
      # # Plot the normals

    #       # Create a PyVista PolyData object
    # point_cloud = pv.PolyData(mesh_points)

    # # Add the ids array to the point data
    # point_cloud["z_normals"] = z_normals  # Ensure the ids array matches the number of points
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color="white", show_edges=True, opacity=0.8)  # Add surface
    # plotter.add_mesh(point_cloud, scalars="z_normals", point_size=10, render_points_as_spheres=True)
    # plotter.add_arrows(mesh.points, mesh["Normals"], mag=0.1, color="red")  # Add normals
    # plotter.show()

    # Filter for points with normals pointing outward (near inlet)
    inlet = mesh.threshold(value=0.98, scalars="Z_Normals", invert=False)
    inlet_ids = np.where(mesh["Z_Normals"] >= 0.99)[0]
    positions_inlet = np.where(np.isin(global_ids, inlet_ids))[0]
    inlet_global_ids =np.where(np.isin(all_points[:,0], positions_inlet))[0]
    ids[inlet_global_ids] = 2
   
    # Filter for points with normals pointing outward (near inlet)
    outlet = mesh.threshold(value=-0.1, scalars="Z_Normals", invert=True)
    outlet_ids = np.where(mesh["Z_Normals"] <= -0.1)[0]
    positions_outlet = np.where(np.isin(global_ids, outlet_ids))[0]
    outlet_global_ids = np.where(np.isin(all_points[:,0], positions_outlet))[0]
    ids[outlet_global_ids] = 3

    # Create a PyVista PolyData object
    point_cloud = pv.PolyData(all_points[:,1:4])

    # Add the ids array to the point data
    point_cloud["ids"] = ids  # Ensure the ids array matches the number of points

    # Plot the points and color by ids
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="ids", point_size=10, render_points_as_spheres=True)
    plotter.show()

    # Visualize the refined inlet
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightgray", label="Original Mesh")
    plotter.add_mesh(inlet, color="blue", label="Inlet")
    plotter.add_mesh(outlet, color="green", label="Outlet")
    #plotter.add_mesh(outer_surface, color="red", label="outer_surface")
    plotter.add_legend()
    plotter.show()
if __name__ == "__main__":
    main()