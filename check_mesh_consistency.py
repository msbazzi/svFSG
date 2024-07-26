import pyvista as pv

# Path to the VTK file
vtk_file = "/home/bazzi/repo/svFSIplus/tests/cases/gr/mesh/solid-mesh-complete/mesh-complete.mesh.vtu"

# Load the VTK file
mesh = pv.read(vtk_file)

# Check for basic mesh consistency
def check_mesh_consistency(mesh):
    n_points = mesh.n_points
    cells = mesh.cells

    i = 0
    while i < len(cells):
        num_points_in_cell = cells[i]
        point_ids = cells[i + 1:i + 1 + num_points_in_cell]
        for point_id in point_ids:
            if point_id >= n_points:
                raise ValueError(f"Invalid point ID {point_id}")
        i += 1 + num_points_in_cell

    print("Mesh consistency check passed.")

try:
    check_mesh_consistency(mesh)
except ValueError as e:
    print(f"Mesh consistency error: {e}")

# Save the mesh if needed
# mesh.save(vtk_file)