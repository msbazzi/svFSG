import numpy as np
import pyvista as pv
import vtk
import os
import sys
from utils import *
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

def compute_cell_centers(mesh):
    # Get the cell data (connectivity) and point data (coordinates)
    points = mesh.points
    cells = mesh.cells_dict[10]  # Assuming tetrahedral elements; adjust as needed

    cell_centers = []
    for cell in cells:
        # Compute the center of the cell by averaging the coordinates of its vertices
        cell_center = np.mean(points[cell], axis=0)
        cell_centers.append(cell_center)

    return np.array(cell_centers)

def interpolate_heat_flux_at_cell_centers(mesh):
    # Get the point data (heat flux)
    heat_flux = mesh.point_data.get('Heat_flux', None)

    if heat_flux is None:
        print("No heat flux data found in the VTK file.")
        return None

    # Compute the cell centers
    cell_centers = compute_cell_centers(mesh)
    
    # Interpolate the heat flux at the cell centers
    # Here we use the same method for simplicity, by averaging the heat flux at the points of each cell
    cells = mesh.cells_dict[10]  # Assuming tetrahedral elements; adjust as needed
    cell_heat_flux = []
    for cell in cells:
        # Compute the average heat flux for the cell
        cell_flux = np.mean(heat_flux[cell], axis=0)
        cell_heat_flux.append(cell_flux)

    return np.array(cell_heat_flux), cell_centers

def create_varProp_array(vtu_mesh, configuration_file):
    
        print("Initializing sheep model...")

        #Read in the vtu file
        vol = pv.read(vtu_mesh) 
        radial_heat_flux = pv.read("/home/bazzi/repo/svFSG/Radial/result_040.vtu")
        axial_heat_flux = pv.read("/home/bazzi/repo/svFSG/Axial/result_040.vtu")

        cell_heat_flux_radial, cell_center_radial = interpolate_heat_flux_at_cell_centers(radial_heat_flux)
        cell_heat_flux_axial, cell_center_axial = interpolate_heat_flux_at_cell_centers(axial_heat_flux)
        
        with open(configuration_file) as f:
            lines = f.readlines()[1:]
            nativeIn = []
            for line in lines:
                nativeIn.append([float(x) for x in line.split()])

        #Build material array (constant for now)
        materialArray = [ nativeIn[7][0],nativeIn[4][0],nativeIn[4][1],nativeIn[4][2],nativeIn[2][0]*10.0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][2],nativeIn[5][2],nativeIn[2][4]*10.0,nativeIn[2][5],0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][3],nativeIn[5][3],nativeIn[2][6]*10.0,nativeIn[2][7],0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][4],nativeIn[5][4],nativeIn[2][8]*10.0,nativeIn[2][9],0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][5],nativeIn[5][5],nativeIn[2][10]*10.0,nativeIn[2][11],0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][1],nativeIn[5][1],nativeIn[2][2]*10.0,nativeIn[2][3],0,0,0,0,0,0,0,0,0,0,\
                          nativeIn[7][2],nativeIn[16][0],nativeIn[14][2],nativeIn[14][1],0,0,0,0,0,0,0,0,0,0]

        #********************
        
        # angles for the diagonal fibers
        ang1 = nativeIn[3][4]
        ang2 = nativeIn[3][5] 


        numCells = vol.GetNumberOfCells()
        e_ma = np.zeros((numCells,98))
        e_r = np.zeros((numCells,3))
        e_t = np.zeros((numCells,3))
        e_z = np.zeros((numCells,3))
        tevgValue = np.zeros(numCells)
        
        for i in range(numCells):
            
            xPt = cell_center_radial[i,0]
            yPt = cell_center_radial[i,1]
            zPt = cell_center_radial[i,2]


            # calculating the local coordinate system
            vr = cell_heat_flux_radial[i,:]/np.linalg.norm(cell_heat_flux_radial[i,:])
            vz = cell_heat_flux_axial[i,:]/np.linalg.norm(cell_heat_flux_axial[i,:])
            vt = np.cross(vr,vz)/np.linalg.norm(np.cross(vr,vz))
            
            e_r[i,:] = vr
            e_t[i,:] = vt
            e_z[i,:] = vz

            e_ma[i,:] = materialArray
            
            
            e_ma[i,5:8] = vr
            e_ma[i,8:11] = vt
            e_ma[i,11:14] = vz

            
            e_ma[i,18:21] = np.cos(90.0*np.pi/180.0)*vz + np.sin(90.0*np.pi/180.0)*vt
            e_ma[i,32:35] = np.cos(0.0*np.pi/180.0)*vz + np.sin(0.0*np.pi/180.0)*vt

            e_ma[i,46:49] =  np.cos(ang1*np.pi/180.0)*vz + np.sin(ang1*np.pi/180.0)*vt
            e_ma[i,60:63] =  np.cos(ang2*np.pi/180.0)*vz + np.sin(ang2*np.pi/180.0)*vt

            e_ma[i,74:77] =  np.cos(90.0*np.pi/180.0)*vz + np.sin(90.0*np.pi/180.0)*vt
            e_ma[i,88:91] =  np.cos(90.0*np.pi/180.0)*vz + np.sin(90.0*np.pi/180.0)*vt 

            
            # assuming is TEVG in the whole domain
            #tevgValue[i] = 1 #getTEVGValue([xPt,yPt,zPt], self.radius)
               # if tevgValue[k + j*self.numRad + i*self.numRad*self.numCirc] > 0:
            e_ma[i,4] = 200000000
            e_ma[i,1] = 1
            e_ma[i,2] = 1
            e_ma[i,3] = 1
            e_ma[i,14] = 0
            e_ma[i,28] = 0
            e_ma[i,42] = 0
            e_ma[i,56] = 0
            e_ma[i,70] = 0
            e_ma[i,84] = 0          

        # Add the new data array to the volume mesh
        e_ma_new = e_ma[:,1:10]
        vol.cell_data["varWallProps"] = e_ma_new.astype(float)

        # Write the vtu file
        vol.save(vtu_mesh)
        
        return vol


def main(vtu_mesh,configuration_file):
    create_varProp_array(vtu_mesh, configuration_file)

if __name__ == "__main__":
    
    simulation_executable = "/home/bazzi/repo/svFSIplus/build/svFSI-build/bin/svFSI"
    configuration_file = "/home/bazzi/repo/svFSG/FolderVesselConfigurationFiles/Native_in_handshake_"
    vtu_mesh = "/home/bazzi/repo/svFSG/pipe/wall-mesh-complete/mesh-complete.mesh.vtu"
        
    main(vtu_mesh, configuration_file)
