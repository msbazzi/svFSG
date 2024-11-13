from utils import *
import time
from cvessel import cvessel
#import sympy
import pyvista as pv
import heat_transfer

# This class provides functionality to running an FSG simulation
class Vessel():

    def __init__(self, **kwargs):

        self.mesh= kwargs.get("mesh","mesh/solid-mesh-complete/mesh-complete.mesh.vtu")
        self.vtp_full= kwargs.get("vtp_full","mesh/solid-mesh-complete/mesh-complete.mesh.vtp")
        self.surfaces_folder= kwargs.get("surfaces_folder","mesh/solid-mesh-complete/mesh-surfaces")
        self.mesh= kwargs.get("mesh","mesh/solid-mesh-complete/mesh-complete.mesh.vtu")
        # self.surfaces= kwargs.get("surfaces",["pipe/solid-mesh-complete/mesh-surfaces/inner.vtp",
        #                                        "pipe/solid-mesh-complete/mesh-surfaces/outer.vtp",
        #                                         "pipe/solid-mesh-complete/mesh-surfaces/outlet.vtp",
        #                                         "pipe/solid-mesh-complete/mesh-surfaces/inlet.vtp" ])
        self.surfaces= kwargs.get("surfaces",["mesh/solid-mesh-complete/mesh-surfaces/wall_inner.vtp",
                                               "mesh/solid-mesh-complete/mesh-surfaces/wall_outer.vtp",
                                                "mesh/solid-mesh-complete/mesh-surfaces/wall_outlet.vtp",
                                                "mesh/solid-mesh-complete/mesh-surfaces/wall_inlet.vtp" ])
        self.radial_heat_flux =  kwargs.get("radial_heat_flux","/home/bazzi/repo/svFSG/simulationResults/heat_radial.vtu")
        self.axial_heat_flux =  kwargs.get("axial_heat_flux","/home/bazzi/repo/svFSG/simulationResults/heat_axial.vtu")
        self.configuration_file =  kwargs.get("configuration_file","/home/bazzi/repo/svFSG/FolderVesselConfigurationFiles/Native_in_handshake_")
        self.vesselReference = None
        self.vesselSolid = None #kwargs.get("vesselSolid","pipe/wall-mesh-complete/mesh-complete.mesh.vtu")
        self.vesselFluid = pv.read("mesh/lumen-mesh-complete/mesh-complete.mesh.vtu")
       #self.vesselFluid = pv.read("mesh/lumen-mesh-complete/mesh-complete.mesh.vtu")
        self.solidResult = None
        self.fluidResult = None
        self.sigma_h = None
        self.tau_h = None

        self.nG = 4
        self.total_time_steps = 0
        self.gnr_max_days = kwargs.get("gnr_max_days", 720)
        self.gnr_step_size = kwargs.get("gnr_step_size", 1.0)
        self.prefix = kwargs.get("prefix", "./")
        self.aneurysm = kwargs.get("aneurysm", 0)
        self.tevg = kwargs.get("tevg", 0)
        self.timeStep = 0
        self.timeIter = 0
        self.omega = 0.5
        self.inletFlow = kwargs.get("inletFlow", -20)
        self.outletPressure = kwargs.get("outletPressure", 6150)
        self.residual = 1.0
        self.residualType = None
        self.viscosity = kwargs.get("viscosity", 0.04)
        self.simulationInputDirectory = kwargs.get("simulationInputDirectory", "FolderSimulationInputFiles")
        self.simulationExecutable = kwargs.get("simulationExecutable","~/svFSI-build/svFSI-build/mysvfsi")
        self.vesselName = kwargs.get("vesselName", "vessel")
        self.resultNum = kwargs.get("resultNum",100)
        self.resultDir = kwargs.get("resultDir","results")
        self.resultDirHeatTR = kwargs.get("resultDirHeatTR","results/heatTR")
        self.resultDirHeatTA = kwargs.get("resultDirHeatTA","results/heatTA")
        self.fluidDir = kwargs.get("fluidDir","fluid-results")
        self.outputDir = kwargs.get("outputDir","Outputs")
        self.estimateWSS = kwargs.get("estimateWSS", False)
        self.predictMethod = kwargs.get("predictMethod", "aitken")
        self.damping = kwargs.get("damping", 1e4)
        self.penalty = kwargs.get("penalty", 1e8)
        self.startTime = 0.0
        self.currTime = 0.0
        self.tolerance = kwargs.get("tolerance", 1e-2)
        self.rotation_matrix = np.eye(3)
        self.scaling_value = 1.0
        self.numProcessorsSolid = None
        self.numProcessorsFluid = None
        self.numProcessorsFluidSolid = None
        self.zcenter = 0.0
        self.nq = 8
        self.iq_eps = 1e-12
        self.mat_W = []
        self.mat_V = []
        self.mat_D = []
        self.cvessels = []
        self.flipContours = False
        self.flipInlet = False
        self.averageStress = True
        self.averageVolume = True
        self.solidLinearSolverType = "GMRES"
        self.smoothAttributesValue = 0.0
        self.skippedFluid = False

    def propogateIQNILS(self):
        self.mat_W = []
        self.mat_V = []
        self.mat_D = []

        pointLocatorSolid = vtk.vtkPointLocator()
        pointLocatorSolid.SetDataSet(self.solidResult)
        pointLocatorSolid.BuildLocator()

        numPts = self.vesselReference.GetNumberOfPoints()
        numCells = self.vesselReference.GetNumberOfCells()

        dcurr = []
        dprev = []

        for q in range(numPts):
            originalCoordinate = np.array(self.vesselReference.GetPoint(q))
            displacement_prev = np.array(self.vesselReference.GetPointData().GetArray("displacements").GetTuple3(q))
            currentCoordinate = originalCoordinate + displacement_prev

            pointIdSolid = pointLocatorSolid.FindClosestPoint(currentCoordinate)
            solidCoordinate = np.array(self.solidResult.GetPoint(pointIdSolid))
            solidDispacement = np.array(self.solidResult.GetPointData().GetArray("Displacement").GetTuple3(pointIdSolid))

            displacement_curr = solidCoordinate + solidDispacement - originalCoordinate
            residual_curr = displacement_curr - displacement_prev
            self.vesselReference.GetPointData().GetArray("residual_curr").SetTuple(q, residual_curr)

            dcurr.append(displacement_curr)
            dprev.append(displacement_prev)


        rcurr = np.array(self.vesselReference.GetPointData().GetArray("residual_curr")).flatten()
        rprev = np.array(self.vesselReference.GetPointData().GetArray("residual_prev")).flatten()

        dcurr = np.array(dcurr).flatten()
        dprev = np.array(dprev).flatten()

        self.mat_W = []
        self.mat_V = []
        self.mat_D = []
        self.mat_D.append(dcurr)
        vnew =  dprev + 0.5*rcurr
        dnew = vnew.reshape((-1,3))

        self.timeIter = 1


        return

    def writeStatus(self, currTime, extra=""):
        with open('svDriverIterations','a') as f:
            print("%d %d %.3e %s %5.4f %5.2f %s" %(self.timeStep,self.timeIter,self.residual, self.residualType, self.omega, currTime, extra), file=f)

    def setTime(self, timeVal):
        self.time = timeVal
        return

    def incrementTimestep(self):
        self.timeStep+=1
        return

    def incrementIteration(self):
        self.timeIter+=1
        return

    def runHeatTransfer(self):
        if   self.timeStep == 0 and self.timeIter == 0:
            self.runHeatT()

    def runDiffusionWSS(self):
        self.runDiffWSS()
        
    def runFluidIteration(self):
        if   self.timeStep == 0 and self.timeIter == 0:
            self.runFluid()
            self.updateSolid()
            self.updateFluidResults()
            self.appendIterfaceResult()
        
        else:
            self.updateSolid()
            self.plot_vessel_solid()
            self.saveSolid()
            self.updateFluid()
            self.saveFluid()
            self.runFluid()
            self.updateFluidResults()
            self.appendIterfaceResult()
            self.skippedFluid = False
            return

    def skipFluidIteration(self):
        numCells = self.vesselReference.GetNumberOfCells()
        for q in range(numCells):
            wss_prev = self.vesselReference.GetCellData().GetArray('wss_curr').GetTuple1(q)
            self.vesselReference.GetCellData().GetArray('wss_prev').SetTuple1(q,wss_prev)
        self.skippedFluid = True
        return

    def runFluidSolidIteration(self):
        self.updateSolid()
        self.plot_vessel_solid()
        self.saveSolid()
        self.updateFluid()
        self.saveFluid()
        self.runFluidSolid()
        self.updateFluidSolidResults()
        self.appendSolidResult()
        self.appendIterfaceResult()
        self.skippedFluid = False
        return

    def runSolidIteration(self):
        self.updateSolid()
        self.saveSolid()
        self.runSolid()

        ''' with open(self.resultDir+'/histor.dat') as f:
            if 'NaN' in f.read():
                print("Simulation has NaN! Reducing omega.", file=sys.stderr)
                self.appendReducedResult()
                return '''

        self.updateSolidResults()
        self.appendSolidResult()
        return

    def runMaterialIteration(self):
        self.updateMaterial()
        self.updateReference()
        self.saveReference()
        self.checkResidual()
        return

    def checkResidual(self):
        tolTypes = ["disp", "sigma_inv", "wss", "jac"]
        tol1 = 100.0*np.max(np.linalg.norm(self.vesselReference.get_array('residual_curr'),axis=1))
        tol2 = np.max(abs((self.vesselReference.get_array('inv_prev')-self.vesselReference.get_array('inv_curr'))/self.sigma_h))
        tol3 = np.max(abs((self.vesselReference.get_array('wss_prev')-self.vesselReference.get_array('wss_curr'))/self.tau_h))
        tol4 = np.max(abs(self.vesselReference.get_array('varWallProps')[:,37::45] - 1.0))
        tolVals = np.array([tol1,tol2,tol3,tol4])
        self.residual = np.max(tolVals)
        self.residualType = tolTypes[tolVals.argmax()]
        return

    def runSolid(self):
        os.system('rm -rf '+self.resultDir +'/*')
        if self.timeIter == 0 and self.timeStep == 0:
            if self.numProcessorsSolid is not None:
                os.system("mpirun -n " + str(self.numProcessorsSolid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/solid_mm_struct.mfs")
            else:
                print("check for the executable",self.simulationExecutable)
                print("check for input dir", self.simulationInputDirectory)
                os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/solid_mm_struct.mfs")
        else:
            if self.numProcessorsSolid is not None:
                os.system("mpirun -n " + str(self.numProcessorsSolid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/solid_aniso.mfs")
            else:
                os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/solid_aniso.mfs")
        print("Solid simulation finished.")
        os.system('cp ' + self.resultDir + '/result_'+str(self.resultNum)+'.vtu simulationResults/solid__struct_' + str(self.timeStep) + '.vtu')

        return

    def runFluid(self):
        os.system('rm -rf '+self.resultDir +'/*')
        if self.numProcessorsFluid is not None:
            os.system("mpirun -n " + str(self.numProcessorsFluid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_fluid.mfs")
        else:
            os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_fluid.mfs")
        with open(self.resultDir+'/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("Fluid simulation finished.")
        os.system('cp ' + self.resultDir + '/result_'+str(self.resultNum)+'.vtu simulationResults/fluid__struct_' + str(self.timeStep) + '.vtu')
        return

    def runHeatT(self):
        os.system('rm -rf 1-procs/*')
        if self.numProcessorsFluid is not None:
            os.system("mpirun -n " + str(self.numProcessorsFluid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_axial.inp")
        else:
            os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_axial.inp")
        with open('1-procs/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("Heat transfer axial simulation finished.")
        os.system('cp 1-procs/result_050.vtu simulationResults/heat_axial.vtu')
        
        
        os.system('rm -rf 1-procs/*')
        if self.numProcessorsFluid is not None:
            os.system("mpirun -n " + str(self.numProcessorsFluid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_radial.inp")
        else:
            os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_radial.inp")
        with open('1-procs/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("Heat transfer radial simulation finished.")
        os.system('cp 1-procs/result_050.vtu simulationResults/heat_radial.vtu')
        return

    def runDiffWSS(self):
        os.system('rm -rf 1-procs/*')
        
        if self.numProcessorsFluid is not None:
            os.system("mpirun -n " + str(self.numProcessorsFluid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_wss_variable.inp")
        else:
            os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_axial.inp")
        with open('1-procs/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("Wss diffusion simulation finished.")
        os.system('cp 1-procs/result_050.vtu simulationResults/wss_' + str(self.timeStep) + '.vtu')
        
        
        ''' os.system('rm -rf 1-procs/*')
        if self.numProcessorsFluid is not None:
            os.system("mpirun -n " + str(self.numProcessorsFluid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_wss_constant.inp")
        else:
            os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/svFSI_heat_radial.inp")
        with open('1-procs/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("Heat transfer radial simulation finished.")
        os.system('cp 1-procs/result_050.vtu simulationResults/wss_' + str(self.timeStep) + '.vtu') '''
        return

    def runFluidSolid(self):
        os.system('rm -rf '+self.resultDir +'/*')
        if self.timeIter == 0 and self.timeStep == 0:
            if self.numProcessorsFluidSolid is not None:
                os.system("mpirun -n " + str(self.numProcessorsFluidSolid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_mm.mfs")
            else:
                os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_mm.mfs")
        else:
            if self.numProcessorsFluidSolid is not None:
                os.system("mpirun -n " + str(self.numProcessorsFluidSolid) + " " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_aniso.mfs")
            else:
                os.system("mpirun " + self.simulationExecutable + " " + self.simulationInputDirectory + "/input_aniso.mfs")
        with open(self.resultDir+'/histor.dat') as f:
            if 'NaN' in f.read():
                raise RuntimeError("Simulation has NaN!")
        print("FSI simulation finished.")
        os.system('cp ' + self.resultDir + '/result_'+str(self.resultNum)+'.vtu simulationResults/fluidsolid_' + str(self.timeStep) + '.vtu')

        return
        
    def initializeVessel(self):
       
        self.vesselReference = self.create_varProp_array()
        #self.estimateIterfaceResult()
        self.saveReference()
        
        self.total_time_steps = self.gnr_max_days//self.gnr_step_size
        self.cvessels = [cvessel() for i in range(self.vesselReference.GetNumberOfCells() * self.nG)]

        return

    def updateMaterial(self):
        numCells = self.vesselReference.GetNumberOfCells()
        input_array = []

        for q in range(numCells):
            sigma_inv = self.vesselReference.GetCellData().GetArray('inv_curr').GetTuple1(q)
            tauw_wss = self.vesselReference.GetCellData().GetArray('wss_curr').GetTuple1(q)
            
            #Rotate into GnR membrane frame
            e_r = self.vesselReference.GetCellData().GetArray('e_r').GetTuple(q)
            e_t = self.vesselReference.GetCellData().GetArray('e_t').GetTuple(q)
            e_z = self.vesselReference.GetCellData().GetArray('e_z').GetTuple(q)
            Q = np.array((e_r,e_t,e_z))

            defGrad_mem = []
            #Gauss point values
            for p in range(self.nG): #loop over the number of gauss points

                defGrad = self.vesselReference.GetCellData().GetArray('defGrad').GetTuple(q)
                defGrad_g = defGrad[p*9:(p+1)*9]
                defGrad_g = np.reshape(defGrad_g, (3,3))
                defGrad_mem_g = np.matmul(Q,np.matmul(defGrad_g, np.transpose(Q)))
                defGrad_mem_g = np.reshape(defGrad_mem_g,9)

                #Queue process

                self.cvessels[q*self.nG + p].prefix = self.outputDir
                self.cvessels[q*self.nG + p].name = 'python_'+str(q)+'_'+str(p)
                self.cvessels[q*self.nG + p].number = q*self.nG + p
                self.cvessels[q*self.nG + p].restart = self.timeStep
                self.cvessels[q*self.nG + p].iteration = self.timeIter
                self.cvessels[q*self.nG + p].num_days = self.gnr_max_days
                self.cvessels[q*self.nG + p].step_size = self.gnr_step_size
                self.cvessels[q*self.nG + p].sigma_inv = sigma_inv
                self.cvessels[q*self.nG + p].tauw_wss = tauw_wss

                for i in range(9):
                    self.cvessels[q*self.nG+ p].F[i] = defGrad_mem_g[i]

                defGrad_mem = np.hstack((defGrad_mem,defGrad_mem_g))

            self.vesselReference.GetCellData().GetArray('defGrad_mem').SetTuple(q,defGrad_mem)


        numrank = int(np.loadtxt('numrank'))
        cvessels_split = np.array_split(self.cvessels,numrank)
        for i in range(numrank):
            cvessel_file = open("materialResults/cvessel_array_out_"+str(i)+".dat","wb")
            pickle.dump(cvessels_split[i], cvessel_file)
            cvessel_file.close()

        print("Running points...")
        time1 = time.time()
        os.system("mpiexec python3 utils_run_vessel.py")
        time2 = time.time()
        print("Time to run_vessel: " + str(time2 - time1))

        for i in range(numrank):
            cvessel_file = open("materialResults/cvessel_array_in_"+str(i)+".dat","rb")
            cvessels_temp = pickle.load(cvessel_file)
            for vess_temp in cvessels_temp:
                self.cvessels[vess_temp.number] = vess_temp
            cvessel_file.close()
        
        print("Parsing points...")

        for q in range(numCells):
            stiffness_mem_g = []
            sigma_mem_g = []
            J_target_g = []
            J_curr_g = []
            for p in range(self.nG):

                output_data = self.cvessels[q*self.nG + p].out_array

                J_curr =  np.linalg.det(np.reshape(output_data[48:57], (3,3)))
                J_target = output_data[46]/output_data[47]

                stiffness = output_data[1:37]
                sigma = output_data[37:46]

                # Dont forget units 
                stiffness = np.array(stiffness)*10.0
                sigma = np.array(sigma)*10.0

                sigma_inv = output_data[57]
                wss = output_data[58]

                stiffness_mem_g = stiffness_mem_g + stiffness.tolist()
                sigma_mem_g = sigma_mem_g + sigma.tolist()
                J_target_g = J_target_g + [J_target]
                J_curr_g = J_curr_g + [J_curr]


            self.vesselReference.GetCellData().GetArray('stiffness_mem').SetTuple(q,stiffness_mem_g)
            self.vesselReference.GetCellData().GetArray('sigma_mem').SetTuple(q,sigma_mem_g)
            self.vesselReference.GetCellData().GetArray('J_target').SetTuple(q,J_target_g)
            self.vesselReference.GetCellData().GetArray('J_curr').SetTuple(q,J_curr_g)

        return

    def updateSolid(self):
        self.vesselSolid = self.vesselReference.warp_by_vector("displacements")
        arrayNames = self.vesselReference.array_names
        for name in arrayNames:
            if name not in ["GlobalNodeID", "varWallProps", "GlobalElementID", "InnerRegionID", "OuterRegionID", "DistalRegionID", "ProximalRegionID", "StructureID", "Pressure", "Coordinate"]:
                if name in self.vesselSolid.point_data:
                    self.vesselSolid.point_data.remove(name)
                if name in self.vesselSolid.cell_data:
                    self.vesselSolid.cell_data.remove(name)
        return

    def updateFluid(self):
        surf = self.vesselSolid.extract_surface()
        inner = thresholdModel(surf, 'InnerRegionID',0.5,1.5)
        if self.fluidResult is not None:
            tempFluid = self.generateFluidMesh(inner)
            self.vesselFluid = interpolateSolution(self.fluidResult, tempFluid)
        else:
            self.vesselFluid = self.generateFluidMesh(inner)

    def saveSolid(self):
        vol = self.vesselSolid

        # os.makedirs(self.prefix + 'pipe/wall-mesh-complete/mesh-surfaces', exist_ok=True)
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-complete.mesh.vtu',vol)
        # surf = vol.extract_surface()
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-complete.mesh.vtp',surf)
        # outer = thresholdModel(surf, 'OuterRegionID',0.5,1.5)
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-surfaces/outer.vtp',outer)
        # distal = thresholdModel(surf, 'DistalRegionID',0.5,1.5)
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-surfaces/inlet.vtp',distal)
        # proximal = thresholdModel(surf, 'ProximalRegionID',0.5,1.5)
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-surfaces/outlet.vtp',proximal)
        # inner = thresholdModel(surf, 'InnerRegionID',0.5,1.5)
        # save_data(self.prefix + 'pipe/wall-mesh-complete/mesh-surfaces/inner.vtp',inner)


        os.makedirs(self.prefix + self.surfaces_folder, exist_ok=True)
        save_data(self.prefix + self.mesh,vol)
        surf = vol.extract_surface()
        save_data(self.prefix + self.vtp_full,surf)
        outer = thresholdModel(surf, 'OuterRegionID',0.5,1.5)
        save_data(self.prefix + self.surfaces[1],outer)
        distal = thresholdModel(surf, 'DistalRegionID',0.5,1.5)
        save_data(self.prefix + self.surfaces[3],distal)
        proximal = thresholdModel(surf, 'ProximalRegionID',0.5,1.5)
        save_data(self.prefix + self.surfaces[2],proximal)
        inner = thresholdModel(surf, 'InnerRegionID',0.5,1.5)
        save_data(self.prefix + self.surfaces[0],inner)

        return

    def saveFluid(self):
        fluidVol = self.vesselFluid

        os.makedirs(self.prefix + 'mesh/lumen-mesh-complete/mesh-surfaces', exist_ok=True)
        # Add global node id to inner volume
        numPtsVol = fluidVol.GetNumberOfPoints()
        numCellsVol = fluidVol.GetNumberOfCells()

        # Need to order ids to match globalNodeID order
        fluidVol.GetPointData().AddArray(pv.convert_array(np.linspace(1,numPtsVol,numPtsVol).astype(np.int32),name="GlobalNodeID"))
        fluidVol.GetCellData().AddArray(pv.convert_array(np.linspace(1,numCellsVol,numCellsVol).astype(np.int32),name="GlobalElementID"))

        save_data(self.prefix + 'mesh/lumen-mesh-complete/mesh-complete.mesh.vtu',fluidVol)
        innerVolSurf = pv.wrap(fluidVol).extract_surface()
        save_data(self.prefix + 'mesh/lumen-mesh-complete/mesh-complete.mesh.vtp',innerVolSurf)

        innerProx = thresholdModel(innerVolSurf, 'ProximalRegionID',0.5,1.5)
        save_data(self.prefix + 'mesh/lumen-mesh-complete/mesh-surfaces/lumen_inlet.vtp',pv.wrap(innerProx))
        innerDist = thresholdModel(innerVolSurf, 'DistalRegionID',0.5,1.5)
        save_data(self.prefix + 'mesh/lumen-mesh-complete/mesh-surfaces/lumen_outlet.vtp',innerDist)
        innerWall = thresholdModel(innerVolSurf, 'OuterRegionID',0.5,1.5)
        save_data(self.prefix + 'mesh/lumen-mesh-complete/mesh-surfaces/lumen_wall.vtp',innerWall)

        return

    def saveReference(self):
        save_data(self.prefix + 'meshIterations/mesh_' + str(self.timeStep) + '_' + str(self.timeIter) + '.vtu', self.vesselReference)
        save_data(self.prefix + 'meshResults/mesh_' + str(self.timeStep) + '.vtu', self.vesselReference)
        return

    def updateFluidSolidResults(self):
        resultdata = read_data(self.resultDir+'/result_'+str(self.resultNum)+'.vtu', file_format="vtu")
        self.solidResult = thresholdModel(resultdata,'Domain_ID', 1.5, 2.5, extract=False, cell=True)
        self.fluidResult = thresholdModel(resultdata,'Domain_ID', 0.5, 1.5, extract=False, cell=True)

        return

    def updateSolidResults(self):
        self.solidResult = read_data(self.resultDir+'/result_'+str(self.resultNum)+'.vtu', file_format="vtu")
        return

    def updateFluidResults(self):
        self.fluidResult = read_data(self.resultDir+'/result_'+str(self.resultNum)+'.vtu', file_format="vtu")
        return

    def appendIterfaceResult(self):
        """
        Add results of fluid simulation
        """
        numCells = self.vesselReference.GetNumberOfCells()

        fluidSurface = self.fluidResult.extract_surface()
        pointLocatorFluid = vtk.vtkPointLocator()
        pointLocatorFluid.SetDataSet(fluidSurface)
        pointLocatorFluid.BuildLocator()

        innerReference = thresholdModel(self.vesselReference.extract_surface(), 'InnerRegionID',0.5,1.5)
        numPointsInner = innerReference.GetNumberOfPoints()

        arrayNames = innerReference.array_names
        for name in arrayNames:
            if name not in ["displacements", "WSS", "Pressure"]:
                if name in innerReference.point_data:
                    innerReference.point_data.remove(name)
                if name in self.vesselSolid.cell_data:
                    innerReference.cell_data.remove(name)

        for q in range(numPointsInner):
            fluidCoordinate = np.array(innerReference.GetPoint(q)) + np.array(innerReference.GetPointData().GetArray("displacements").GetTuple3(q))
            fluidId = int(pointLocatorFluid.FindClosestPoint(fluidCoordinate))
            innerReference.GetPointData().GetArray('WSS').SetTuple1(q,np.linalg.norm(fluidSurface.GetPointData().GetArray('WSS').GetTuple3(fluidId)))
            innerReference.GetPointData().GetArray('Pressure').SetTuple1(q,fluidSurface.GetPointData().GetArray('Pressure').GetTuple1(fluidId))

        if self.smoothAttributesValue:
            innerReference = smoothAttributes(innerReference, self.smoothAttributesValue, 100)

        pointLocatorInner = vtk.vtkPointLocator()
        pointLocatorInner.SetDataSet(innerReference)
        pointLocatorInner.BuildLocator()

        # Save innerReference as a .vtu file
        inner_reference_vtp_path = "simulationResults/wss.vtp"
        innerReference.save(inner_reference_vtp_path)

        numCellsFluidDomain = self.vesselFluid.GetNumberOfCells()

        for q in range(numCells):
            #fluidStressId = int(self.vesselReference.GetCellData().GetArray('fluidStressQueryID').GetTuple1(q))
            cell = self.vesselReference.GetCell(q)
            cellPts = cell.GetPointIds()

            # For each cell, use trapezoidal integration to compute WSS
            cellWSS = 0.0
            numberOfPoints = 0
            for p in range(cellPts.GetNumberOfIds()):
                ptId = cellPts.GetId(p)
                if self.vesselReference.GetPointData().GetArray('InnerRegionID').GetTuple1(ptId) == 1:
                    innerCoordinate = np.array(self.vesselReference.GetPoint(p))
                    innerId = int(pointLocatorInner.FindClosestPoint(innerCoordinate))
                    self.vesselReference.GetPointData().GetArray('Pressure').SetTuple1(ptId,innerReference.GetPointData().GetArray('Pressure').GetTuple1(innerId))
                    cellWSS += innerReference.GetPointData().GetArray('WSS').GetTuple1(innerId)
                    numberOfPoints += 1
            #Average WSS in cell -- fix this -- assumes it is only different the zero for the inner wall
            if numberOfPoints > 0:
                cellWSS *= 1/float(numberOfPoints)
            else: 
                cellWSS = 0.0
           
            wss_prev = self.vesselReference.GetCellData().GetArray('wss_curr').GetTuple1(q)
            self.vesselReference.GetCellData().GetArray('wss_prev').SetTuple1(q,wss_prev)
            self.vesselReference.GetCellData().GetArray('wss_curr').SetTuple1(q,cellWSS)

        return
            
    def appendReducedResult(self):

        numPts = self.vesselReference.GetNumberOfPoints()
        numCells = self.vesselReference.GetNumberOfCells()

        self.omega = self.omega/2.0

        # Calculate cauchy green tensor
        for q in range(numPts):
            rcurr = np.array(self.vesselReference.GetPointData().GetArray("residual_curr").GetTuple3(q))
            dcurr = np.array(self.vesselReference.GetPointData().GetArray("displacements").GetTuple3(q))
            displacement = dcurr - self.omega*rcurr

            self.vesselReference.GetPointData().GetArray("displacements").SetTuple(q, displacement)

        self.vesselReference = computeGaussValues(self.vesselReference,"displacements")

        return

    def appendSolidResult(self):

        if self.smoothAttributesValue:
            self.solidResult = smoothAttributes(self.solidResult,self.smoothAttributesValue,100)

        pointLocatorSolid = vtk.vtkPointLocator()
        pointLocatorSolid.SetDataSet(self.solidResult)
        pointLocatorSolid.BuildLocator()

        numPts = self.vesselReference.GetNumberOfPoints()
        numCells = self.vesselReference.GetNumberOfCells()

        time1 = time.time()

        dcurr = []
        dprev = []

        for q in range(numPts):
            originalCoordinate = np.array(self.vesselReference.GetPoint(q))
            displacement_prev = np.array(self.vesselReference.GetPointData().GetArray("displacements").GetTuple3(q))
            currentCoordinate = originalCoordinate + displacement_prev

            pointIdSolid = pointLocatorSolid.FindClosestPoint(currentCoordinate)
            solidCoordinate = np.array(self.solidResult.GetPoint(pointIdSolid))
            solidDispacement = np.array(self.solidResult.GetPointData().GetArray("Displacement").GetTuple3(pointIdSolid))

            displacement_curr = solidCoordinate + solidDispacement - originalCoordinate
            residual_curr = displacement_curr - displacement_prev
            self.vesselReference.GetPointData().GetArray("residual_curr").SetTuple(q, residual_curr)

            dcurr.append(displacement_curr)
            dprev.append(displacement_prev)


        rcurr = np.array(self.vesselReference.GetPointData().GetArray("residual_curr")).flatten()
        rprev = np.array(self.vesselReference.GetPointData().GetArray("residual_prev")).flatten()

        dcurr = np.array(dcurr).flatten()
        dprev = np.array(dprev).flatten()



        if self.predictMethod == "none":
            dnew = dcurr.reshape((-1, 3))


        elif self.predictMethod == "aitken":
            if self.timeIter > 1:
                diff = rcurr - rprev
                self.omega = -self.omega*np.dot(rprev,diff)/np.dot(diff,diff)
                #if self.skippedFluid == False:
                #    self.omega = 0.25
            elif self.timeStep == 0 and self.timeIter == 0:
                self.omega = 1.0
            else:
                self.omega = 0.5

            if self.omega > 2.0:
                self.omega = 2.0
            elif self.omega < 0.1:
                self.omega = 0.1

            vnew =  dprev + self.omega*rcurr
            dnew = vnew.reshape((-1,3))


        elif self.predictMethod == "iqnils":

            if self.timeIter == 0:
                self.mat_W = []
                self.mat_V = []
                self.mat_D = []
                self.mat_D.append(dcurr)
                vnew =  dprev + 0.5*rcurr
                dnew = vnew.reshape((-1,3))
            elif self.timeIter == 1:
                self.mat_D.append(dcurr)
                vnew =  dprev + 0.5*rcurr
                dnew = vnew.reshape((-1,3))
            else:
                self.mat_D.append(dcurr)
                self.mat_W.append(self.mat_D[-1] - self.mat_D[-2])
                self.mat_V.append(rcurr - rprev)

                # trim to max number of considered vectors
                self.mat_V = self.mat_V[-self.nq:]
                self.mat_W = self.mat_W[-self.nq:]

                # remove linearly dependent vectors
                while True:
                    # QR decomposition
                    qq, rr = np.linalg.qr(np.array(self.mat_V[:self.nq]).T)

                    # tolerance for redundant vectors
                    i_eps = np.where(
                        np.abs(np.diag(rr)) < self.iq_eps
                    )[0]
                    if not np.any(i_eps):
                        break

                    print("Filtering " + str(len(i_eps)) + " time steps")
                    for i in reversed(i_eps):
                        self.mat_V.pop(i)
                        self.mat_W.pop(i)

                # solve for coefficients
                bb = np.linalg.solve(rr.T, -np.dot(np.array(self.mat_V), rcurr))
                cc = np.linalg.solve(rr, bb)

                # update
                vnew = dcurr + np.dot(np.array(self.mat_W).T, cc)
                dnew = vnew.reshape((-1, 3))


        # Calculate cauchy green tensor
        for q in range(numPts):
            rcurr = np.array(self.vesselReference.GetPointData().GetArray("residual_curr").GetTuple3(q))
            self.vesselReference.GetPointData().GetArray("residual_prev").SetTuple(q, rcurr)
            self.vesselReference.GetPointData().GetArray("displacements").SetTuple(q, dnew[q])

        self.vesselReference = computeGaussValues(self.vesselReference,"displacements")

        # Get stress invariant
        for q in range(numCells):
            cell = self.solidResult.GetCell(q)
            cellPts = cell.GetPointIds()

    
            sigma_inv = 0.0

            stressRange = range(q, q+1)

            for p in stressRange:
                cell = self.solidResult.GetCell(p)
                cellPts = cell.GetPointIds()
                cellSigma = 0.0
                for r in range(cellPts.GetNumberOfIds()):
                    ptId = cellPts.GetId(r)
                    pointSigma = self.solidResult.GetPointData().GetArray('Cauchy_stress').GetTuple6(ptId)           
                    cellSigma += (pointSigma[0]+pointSigma[1]+pointSigma[2])
                cellSigma *= 1/float(cellPts.GetNumberOfIds())
                sigma_inv = sigma_inv + cellSigma

            sigma_inv = 0.1*sigma_inv

            inv_prev = self.vesselReference.GetCellData().GetArray("inv_curr").GetTuple1(q)
            self.vesselReference.GetCellData().GetArray("inv_prev").SetTuple1(q, inv_prev)
            self.vesselReference.GetCellData().GetArray("inv_curr").SetTuple1(q, sigma_inv)

        return

    def appendFluidResult(self):

        numPoints = self.vesselFluid.GetNumberOfPoints()

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(self.fluidResult)
        pointLocator.BuildLocator()

        velocity_array = np.zeros((numPoints,3))
        pressure_array = np.zeros((numPoints,1))

        for q in range(numPoints):
            coordinate = self.vesselFluid.GetPoint(q)
            pointIdSource = pointLocator.FindClosestPoint(coordinate)

            velocity_array[q,:] = self.fluidResult.GetPointData().GetArray('Velocity').GetTuple(pointIdSource)
            pressure_array[q,:] = self.fluidResult.GetPointData().GetArray('Pressure').GetTuple1(pointIdSource)

        self.vesselFluid.GetPointData().AddArray(pv.convert_array(np.array(velocity_array),name="Velocity"))
        self.vesselFluid.GetPointData().AddArray(pv.convert_array(np.array(self.outletPressure),name="Pressure"))

        return

    def updateReference(self):

        numCells = self.vesselReference.GetNumberOfCells()
        numPoints = self.vesselReference.GetNumberOfPoints()

        array_names = self.vesselReference.array_names

        dv_max = 0

        if "temp_array" in array_names:
            self.vesselReference.rename_array("varWallProps","material_global")
            self.vesselReference.rename_array("temp_array","varWallProps")

        for q in range(numCells):
            e_r = self.vesselReference.GetCellData().GetArray('e_r').GetTuple(q)
            e_t = self.vesselReference.GetCellData().GetArray('e_t').GetTuple(q)
            e_z = self.vesselReference.GetCellData().GetArray('e_z').GetTuple(q)
            Q = np.array((e_r,e_t,e_z))

            J_curr = self.vesselReference.GetCellData().GetArray('J_curr').GetTuple(q)
            inv_curr = self.vesselReference.GetCellData().GetArray('inv_curr').GetTuple1(q)

            sigma_gnr_g = []
            stiffness_g = []
            varWallProps_g = []
            p_est_g = []

            J_c = 0.0
            p_est_c = 0.0
            for p in range(self.nG):
                J_c += J_curr[p]/self.nG

                sigma_gnr_mem = self.vesselReference.GetCellData().GetArray('sigma_mem').GetTuple(q)[p*9:(p+1)*9]
                sigma_gnr = rotateStress(sigma_gnr_mem,Q)
                sigma_gnr_g = sigma_gnr_g + sigma_gnr.tolist()

                p_est = (10.0*inv_curr-(sigma_gnr[0]+sigma_gnr[1]+sigma_gnr[2]))/3.0
                p_est_g = p_est_g + [p_est]
                p_est_c += p_est/self.nG


            for p in range(self.nG):
                J_target = self.vesselReference.GetCellData().GetArray('J_target').GetTuple(q)[p]
                stiffness_mem = self.vesselReference.GetCellData().GetArray('stiffness_mem').GetTuple(q)[p*36:(p+1)*36]
                stiffness = rotateStiffness(stiffness_mem,Q)
                stiffness_g = stiffness_g + stiffness.tolist()

                if self.averageVolume:
                    J_final = J_target/J_c
                else:
                    J_final = J_target/J_curr[p]

                varWallProps_g = np.hstack((varWallProps_g,np.hstack((stiffness,p_est_g[p],J_final,sigma_gnr_g[p*6:(p+1)*6],p))))

                if np.abs(J_final - 1) > dv_max:
                    dv_max = np.abs(J_final - 1)

            self.vesselReference.GetCellData().GetArray('varWallProps').SetTuple(q,varWallProps_g)
            self.vesselReference.GetCellData().GetArray('p_est').SetTuple(q,p_est_g)
            self.vesselReference.GetCellData().GetArray('sigma_gnr').SetTuple(q,sigma_gnr_g)

            self.vesselReference.GetCellData().GetArray('J_curr').SetTuple(q,J_curr)
            self.vesselReference.GetCellData().GetArray('J_c').SetTuple1(q,J_c)
            self.vesselReference.GetCellData().GetArray('p_est_c').SetTuple1(q,p_est_c)

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

    def compute_cell_centers(self,heat_flux_data):
        # Get the cell data (connectivity) and point data (coordinates)
        points = heat_flux_data.points
        cells = heat_flux_data.cells_dict[10]  # Assuming tetrahedral elements; adjust as needed
        #cells = heat_flux_data.cells_dict[12]  # Assuming hexa elements; adjust as needed

        cell_centers = []
        for cell in cells:
            # Compute the center of the cell by averaging the coordinates of its vertices
            cell_center = np.mean(points[cell], axis=0)
            cell_centers.append(cell_center)

        return np.array(cell_centers)

    def interpolate_heat_flux_at_cell_centers(self,heat_flux_data):
        # Get the point data (heat flux)
        heat_flux = heat_flux_data.point_data.get('Heat_flux', None)

        if heat_flux is None:
            print("No heat flux data found in the VTK file.")
            return None

        # Compute the cell centers
        cell_centers = self.compute_cell_centers(heat_flux_data)
        
        # Interpolate the heat flux at the cell centers
        # Here we use the same method for simplicity, by averaging the heat flux at the points of each cell
        cells = heat_flux_data.cells_dict[10]  # Assuming tetrahedral elements; adjust as needed
        #cells = heat_flux_data.cells_dict[12]  # Assuming hexA elements; adjust as needed
        cell_heat_flux = []
        for cell in cells:
            # Compute the average heat flux for the cell
            cell_flux = np.mean(heat_flux[cell], axis=0)
            cell_heat_flux.append(cell_flux)

        return np.array(cell_heat_flux), cell_centers

    def create_varProp_array(self):
    
        print("Initializing sheep model...")

        #Read in the vtu file
        vol = pv.read(self.mesh) 
        radial_heat_flux = pv.read(self.radial_heat_flux)
        axial_heat_flux = pv.read(self.axial_heat_flux)

        cell_heat_flux_radial, cell_center_radial = self.interpolate_heat_flux_at_cell_centers(radial_heat_flux)
        cell_heat_flux_axial, cell_center_axial = self.interpolate_heat_flux_at_cell_centers(axial_heat_flux)
        
        with open(self.configuration_file) as f:
            lines = f.readlines()[1:]
            nativeIn = []
            for line in lines:
                nativeIn.append([float(x) for x in line.split()])
        
        self.sigma_h = nativeIn[13][0]
        self.tau_h = nativeIn[13][1]

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
        innerIds = np.zeros(numCells)
        #for j in range(numPoints):
   

             
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
            # e_ma[i,4] = 200000000
            # e_ma[i,1] = 1
            # e_ma[i,2] = 1
            # e_ma[i,3] = 1
            # e_ma[i,14] = 0
            # e_ma[i,28] = 0
            # e_ma[i,42] = 0
            # e_ma[i,56] = 0
            # e_ma[i,70] = 0
            # e_ma[i,84] = 0          
        
        
        # cell_center_radial = pv.PolyData(cell_center_radial) 

        # cell_center_radial['ema'] =  e_ma[:,1]
        # plotter = pv.Plotter()
        # plotter.add_mesh(cell_center_radial, scalars='ema', point_size=10)
        # plotter.show()
        #heat_transfer.plot_heat_flux(cell_center_radial, -e_r, skip=5)
        ids = [ 'InnerRegionID', 'OuterRegionID', 'DistalRegionID', 'ProximalRegionID']
        numPoints = vol.GetNumberOfPoints()
        distance = np.zeros(numPoints)
        for i in range(len(self.surfaces)):
            surface_mesh = pv.read(self.surfaces[i])
            print("Processing surface mesh", self.surfaces[i])
        
            locator = pv.PolyData(surface_mesh.points)
            surface_points_ids = np.zeros(numPoints, dtype=int)

            for j in range(numPoints):
                point = vol.points[j]
            
                # Find the closest point on the surface and check if it's within the tolerance
                closest_point_id = locator.find_closest_point(point)
                closest_point = surface_mesh.points[closest_point_id]
                
                distance [j] = np.linalg.norm(point - closest_point)
              
                if distance [j] < 1e-2:
                    surface_points_ids[j] = 1 
            vol.GetPointData().AddArray(pv.convert_array((surface_points_ids).astype(np.int32),name=ids[i]))
 
       
        numCells = vol.GetNumberOfCells()
        vol.GetCellData().AddArray(pv.convert_array(np.zeros((numCells,45*self.nG)).astype(float),name="temp_rray"))
        vol.GetCellData().AddArray(pv.convert_array(np.linspace(1,numCells,numCells).astype(np.int32),name="aGlobalElementID"))
        vol.GetCellData().AddArray(pv.convert_array(np.linspace(1,numCells,numCells).astype(np.int32),name="CellStructureID"))
        vol.GetCellData().AddArray(pv.convert_array(e_ma.astype(float),name="varWallProps"))
       
        #Exchange these to gauss point quantities
        #Matrices/arrays
      
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile(np.array([1,0,0,0,1,0,0,0,1]),self.nG),(numCells,1)).astype(float),name="defGrad"))
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile(np.array([1,0,0,0,1,0,0,0,1]),self.nG),(numCells,1)).astype(float),name="defGrad_mem"))

        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile(np.zeros(36),self.nG),(numCells,1)).astype(float),name="stiffness_mem"))
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile(np.zeros(9),self.nG),(numCells,1)).astype(float),name="sigma_mem"))
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile(np.zeros(6),self.nG),(numCells,1)).astype(float),name="sigma_gnr"))
        #Scalar values
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile([1],self.nG),(numCells,1)).astype(float),name="J_curr"))
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile([1],self.nG),(numCells,1)).astype(float),name="J_target"))
        vol.GetCellData().AddArray(pv.convert_array(np.tile(np.tile([0],self.nG),(numCells,1)).astype(float),name="p_est"))

        vol.GetCellData().AddArray(pv.convert_array(np.ones(numCells).astype(float),name="J_c"))
        vol.GetCellData().AddArray(pv.convert_array(np.zeros(numCells).astype(float),name="p_est_c"))
        vol.GetCellData().AddArray(pv.convert_array(nativeIn[13][1]*np.ones(numCells).astype(float),name="wss_curr"))
        vol.GetCellData().AddArray(pv.convert_array(nativeIn[13][1]*np.ones(numCells).astype(float),name="wss_prev"))

        vol.GetCellData().AddArray(pv.convert_array(nativeIn[13][0]*np.ones(numCells).astype(float),name="inv_curr"))
        vol.GetCellData().AddArray(pv.convert_array(nativeIn[13][0]*np.ones(numCells).astype(float),name="inv_prev"))


        vol.GetCellData().AddArray(pv.convert_array(e_r.astype(float),name="e_r"))
        vol.GetCellData().AddArray(pv.convert_array(e_t.astype(float),name="e_t"))
        vol.GetCellData().AddArray(pv.convert_array(e_z.astype(float),name="e_z"))
        vol.GetCellData().AddArray(pv.convert_array(e_ma.astype(float),name="material_global"))
        
        numPts = vol.GetNumberOfPoints()
        vol.GetPointData().AddArray(pv.convert_array(np.linspace(1,numPts,numPts).astype(np.int32),name="GlobalNodeID"))
        vol.GetPointData().AddArray(pv.convert_array(np.tile(np.zeros(3),(numPts,1)).astype(float),name="displacements"))
        vol.GetPointData().AddArray(pv.convert_array(np.tile(np.zeros(3),(numPts,1)).astype(float),name="residual_curr"))
        vol.GetPointData().AddArray(pv.convert_array(np.tile(np.zeros(3),(numPts,1)).astype(float),name="residual_prev"))
        vol.GetPointData().AddArray(pv.convert_array(np.zeros(numPts).astype(float),name="Pressure"))
        vol.GetPointData().AddArray(pv.convert_array(nativeIn[13][1]*np.ones(numPts).astype(float),name="WSS"))
        vol.GetPointData().AddArray(pv.convert_array(vol.points.astype(float),name="Coordinate"))

        return vol

    def generateFluidMesh(self, surf):
        print("Making inner volume...")

        wall_mesh = pv.read(self.mesh)

        # Visualize the wall mesh (optional)
        wall_mesh.plot(show_edges=True)

        # Extract the inner surface (assuming you can threshold the scalars or point data to extract it)
        # You might need a specific condition here based on how the wall mesh is structured
        # Example: Using thresholding to separate inner and outer surfaces
        #inner_surface = pv.read(surf)

        # Visualize the extracted inner surface (optional)
        surf.plot(show_edges=True)

        # Generate the lumen volume (Delaunay 3D)
        lumen_mesh = surf.delaunay_3d()

        # Visualize the lumen mesh (optional)
        lumen_mesh.plot(show_edges=True)

        # Save the lumen mesh to a file
        lumen_mesh.save('lumen_mesh.vtk')

        print("Lumen mesh generated and saved as 'lumen_mesh.vtk'")


        numPts = lumen_mesh.GetNumberOfPoints()
        lumen_mesh.GetPointData().AddArray(pv.convert_array(np.tile(np.zeros(3),(numPts,1)).astype(float),name="Velocity"))
        lumen_mesh.GetPointData().AddArray(pv.convert_array(np.zeros(numPts).astype(float),name="Pressure"))
        return lumen_mesh

    def setInputFileValues(self):

        with open(self.simulationInputDirectory + '/input_aniso.mfs', 'r') as file:
            data = file.readlines()
        data[86] = "      Penalty parameter: " + str(self.penalty) + "\n"
        data[88] = "      Mass damping: " + str(self.damping) + "\n"
        data[118] = "      Value: " + str(self.inletFlow) + "\n"
        data[130] = "      Value: " + str(- self.outletPressure / self.inletFlow) + "\n"
        with open(self.simulationInputDirectory + '/input_aniso.mfs', 'w') as file:
            file.writelines(data)

        with open(self.simulationInputDirectory + '/input_mm.mfs', 'r') as file:
            data = file.readlines()
        data[89] = "      Penalty parameter: " + str(self.penalty) + "\n"
        data[91] = "      Mass damping: " + str(self.damping) + "\n"
        data[120] = "      Value: " + str(self.inletFlow) + "\n"
        data[132] = "      Value: " + str(- self.outletPressure / self.inletFlow) + "\n"
        with open(self.simulationInputDirectory + '/input_mm.mfs', 'w') as file:
            file.writelines(data)

        with open(self.simulationInputDirectory + '/input_fluid.mfs', 'r') as file:
            data = file.readlines()
        data[73] = "      Value: " + str(self.inletFlow) + "\n"
        data[81] = "      Value: " + str(- self.outletPressure / self.inletFlow) + "\n"
        with open(self.simulationInputDirectory + '/input_fluid.mfs', 'w') as file:
            file.writelines(data)

        with open(self.simulationInputDirectory + '/solid_aniso.mfs', 'r') as file:
            data = file.readlines()
        data[59] = "   Penalty parameter: " + str(self.penalty) + "\n"
        data[61] = "   Mass damping: " + str(self.damping) + "\n"
        data[63] = "   LS type: " + str(self.solidLinearSolverType) + " {\n"
        if self.solidLinearSolverType == "GMRES":
            data[65] = "      Max iterations: 10\n"
            data[66] = "      Krylov space dimension: 50\n"
        else:
            data[65] = "      Max iterations: 1000\n"
            data[66] = "      Krylov space dimension: 250\n"
        with open(self.simulationInputDirectory + '/solid_aniso.mfs', 'w') as file:
            file.writelines(data)

        with open(self.simulationInputDirectory + '/solid_mm.mfs', 'r') as file:
            data = file.readlines()
        data[60] = "   Penalty parameter: " + str(self.penalty) + "\n"
        data[62] = "   Mass damping: " + str(self.damping) + "\n"
        #data[58] = "   LS type: " + str(self.solidLinearSolverType) + " {\n"
        # if self.solidLinearSolverType == "GMRES":
        #     data[60] = "      Max iterations: 10\n"
        #     data[61] = "      Krylov space dimension: 50\n"
        # else:
        #     data[60] = "      Max iterations: 1000\n"
        #     data[61] = "      Krylov space dimension: 250\n"
        with open(self.simulationInputDirectory + '/solid_mm.mfs', 'w') as file:
            file.writelines(data)


    """
    def setPressure()

    def setTimestep()

    def setDampening()

    def setPenalty()

    """
    def plot_vessel_solid(self):
        # Ensure self.vesselSolid is a pyvista-compatible mesh
        if not isinstance(self.vesselSolid, pv.core.pointset.PointSet):
            print("self.vesselSolid is not a pyvista PointSet. Cannot plot.")
            return

        plotter = pv.Plotter()
        plotter.add_mesh(self.vesselSolid, color='lightblue', show_edges=True)
        plotter.add_axes()
        plotter.show()