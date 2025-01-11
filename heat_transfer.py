import meshio
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()  # Turn on interactive mode


# Function to run OpenFOAM simulation
def run_simulation(numProcessorsSolid, simulationExecutable, simulationInputDirectory):
    subprocess.run(
        ["mpirun", "-n", str(numProcessorsSolid), simulationExecutable, f"{simulationInputDirectory}/svFSI_heat.inp"],
        check=True,
        capture_output=True,
        text=True )


# Function to plot and save heat flux
def save_heat_flux(results, output_text_file):
    try:
        mesh_data = meshio.read(results)

        points = mesh_data.points
        heat_flux = mesh_data.point_data.get('Heat_flux', None)

        if heat_flux is None:
            print("No heat flux data found in the VTK file.")
            return None, None

        with open(output_text_file, "w") as f:
            f.write("X, Y, Z, Fx, Fy, Fz\n")
            for point, flux in zip(points, heat_flux):
                f.write(f"{point[0]}, {point[1]}, {point[2]}, {flux[0]}, {flux[1]}, {flux[2]}\n")
        print(f"Heat flux vectors saved to: {output_text_file}")

        return points, heat_flux

    except Exception as e:
        print(f"Error in saving heat flux: {e}")
        return None, None


def plot_heat_flux(points, heat_flux, skip=10):
    try:
        if points is None or heat_flux is None:
            print("No data to plot.")
            return

        # Debugging prints to check data
        print(f"Points shape: {points.shape}")
        print(f"Heat Flux shape: {heat_flux.shape}")

        # Skip points (take every nth point)
        points = points[::skip]
        heat_flux = heat_flux[::skip]

        # Enable interactive mode
        plt.ion()

        # Plot the heat flux vectors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot quiver plot with dynamic capability
        ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                  heat_flux[:, 0], heat_flux[:, 1], heat_flux[:, 2],
                  length=0.5, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Heat Flux Vectors')

        # Show the plot and make it interactive (rotate, zoom)
        plt.show(block=True)  # Keeps the plot open until manually closed

        # Allow interactive rotation and manipulation
        ax.mouse_init()  # Initializes mouse controls for interactive 3D plot
        plt.draw()       # Updates the figure
    except Exception as e:
        print(f"Error in plotting heat flux: {e}")

def read_heat_flux(input_text_file):
    points  = []
    heat_flux = []

    
    with open(input_text_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip header line
        for line in lines:
            data = line.strip().split(',')  # Split line by commas
            # Extract points and heat flux values from the file
            x, y, z = map(float, data[:3])
            fx, fy, fz = map(float, data[3:])
            points.append([x, y, z])
            heat_flux.append([fx, fy, fz])
    
    points = np.array(points)
    heat_flux = np.array(heat_flux)
    return points, heat_flux


def main():
    '''mesh_folder = "/home/bazzi/repo/svFSG/pipe/wall/"
    numProcessorsSolid = 1
    simulationExecutable = "/home/bazzi/repo/svFSI/build/svFSI-build/bin/svFSI"
    simulationInputDirectory = "/home/bazzi/repo/svFSG"
    InputFiles = "/home/bazzi/repo/svFSG/FolderSimulationInputFiles"
    vtk_file = os.path.join(simulationInputDirectory, "pipe/wall-mesh-complete/mesh-complete.mesh.vtu")
    results = os.path.join(simulationInputDirectory, "Axial/result_040.vtu")
    output_text_file = os.path.join(simulationInputDirectory, "heat_flux_axial.txt")
  
    # Save heat flux data and get points and heat_flux arrays
    points, heat_flux = save_heat_flux(results, output_text_file)'''
    radial_heat_flux = "/home/bazzi/repo/svFSG/heat_flux_radial.txt"
    axial_heat_flux = "/home/bazzi/repo/svFSG/heat_flux_axial.txt"

    points_radial, heat_flux_radial = read_heat_flux(radial_heat_flux)
    points_axial, heat_flux_axial = read_heat_flux(axial_heat_flux)

    # Plot heat flux vectors
    plot_heat_flux(points_radial, heat_flux_radial, skip=50)

    # Plot heat flux vectors
    #plot_heat_flux(points_axial, heat_flux_axial, skip=50)

if __name__ == "__main__":
    main() 