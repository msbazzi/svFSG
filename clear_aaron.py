import pyvista as pv
import os



# Path to mesh-surfaces folder
mesh_surfaces_dir = "/home/bazzi/repo/svFSG/mesh/solid-mesh-complete/mesh-surfaces"

# For each mesh in the folder, clear the arrays and save the mesh
for mesh_file in os.listdir(mesh_surfaces_dir):
    if mesh_file.endswith(".vtp"):
        # Load the mesh
        mesh = pv.read(os.path.join(mesh_surfaces_dir, mesh_file))

        # Clear the arrays
        mesh.clear_data()

        # Save the mesh
        mesh.save(os.path.join(mesh_surfaces_dir, mesh_file))