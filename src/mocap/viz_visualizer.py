import numpy as np
import pinocchio
from pinocchio.utils import *
from pinocchio.rpy import matrixToRpy, rpyToMatrix, rotate
from robot_properties_solo.config import SoloConfig
from mocap.loader import loadHumanoidBall, loadHumanoidWoBall, loadHumanoidBullet
import subprocess
import threading
import time
import meshcat
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

def m2a(m): return np.array(m.flat)
def a2m(a): return np.matrix(a).T

class VisualModel():
    def __init__(self,display=False):
        self.display=display

        self.robot_o = loadHumanoidBall()
        self.robot = loadHumanoidWoBall()
        self.model_o = self.robot_o.model
        self.model = self.robot.model
        self.data_o = self.model_o.createData()
        self.data = self.model.createData()
        
        self.robot_bullet = loadHumanoidBullet()
        self.model_bullet = self.robot_bullet.model
        self.data_bullet = self.model_bullet.createData()
        server_args = []
        proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=server_args)

        viewer = meshcat.Visualizer(zmq_url=zmq_url)
        
#         viewer1 = meshcat.Visualizer(zmq_url=zmq_url)

        # Setup the visualizer
        self.viz = viz = pinocchio.visualize.MeshcatVisualizer(
            self.robot_o.model, self.robot_o.collision_model, self.robot_o.visual_model
        )
#         self.viz1 = viz1 = pinocchio.visualize.MeshcatVisualizer(
#             self.robot_o.model, self.robot_o.collision_model, self.robot_o.visual_model
#         )
        
        viz.initViewer(viewer)
        viz.loadViewerModel()
        
#         viz1.initViewer(viewer1)
#         viz1.loadViewerModel()
        
        lSole = 'l_foot'
        rSole = 'r_foot'
        torso = 'base_link'
        lGripper = 'l_gripper'
        rGripper = 'r_gripper'

        self.lSoleId_o = self.model_o.getFrameId(lSole)
        self.rSoleId_o = self.model_o.getFrameId(rSole)
        self.torsoId_o = self.model_o.getFrameId(torso)
        self.lGripperId_o = self.model_o.getFrameId(lGripper)
        self.rGripperId_o = self.model_o.getFrameId(rGripper)
        
        self.lSoleId = self.model.getFrameId(lSole)
        self.rSoleId = self.model.getFrameId(rSole)
        self.torsoId = self.model.getFrameId(torso)
        self.lGripperId = self.model.getFrameId(lGripper)
        self.rGripperId = self.model.getFrameId(rGripper)

        self.nq_o = self.model_o.nq
        self.nq = self.model.nq
        self.nv_o = self.model_o.nv
        self.nv = self.model.nv

        assert abs(self.nq-self.nq_o)<1.e-6, "[Error]Models verification error[nq]!"
        assert abs(self.nv-self.nv_o)<1.e-6, "[Error]Models verification error[nv]!"
        # assert abs(self.lSoleId-self.lSoleId_o)<1.e-6, "[Error]Models verification error[lSoleId]!"
        # assert abs(self.rSoleId-self.rSoleId_o)<1.e-6, "[Error]Models verification error[rSoleId]!"
        # assert abs(self.torsoId-self.torsoId_o)<1.e-6, "[Error]Models verification error[torsoId]!"
        # assert abs(self.lGripperId-self.lGripperId_o)<1.e-6, "[Error]Models verification error[lGripperId]!"
        # assert abs(self.rGripperId-self.rGripperId_o)<1.e-6, "[Error]Models verification error[rGripperId]!"

        self.na = self.nq-7
        self.dq0 = pinocchio.utils.zero(self.nv)
        self.q0 = self.model.referenceConfigurations["standing"]

        # Calibrate origin at center of knee points
        self.calibrateOrigin()

        # Reset hand and leg pose
        self.clearPose()

        

        ###############################Pinocchio Manual Tuning(Joint ID table)##############################################
        #                                                  TROSO
        #                                          0, 1, 2, 3, 4, 5, 6
        #                                        ------------------------
        #                                       |   Name  | q_id  | j_id |
        #                                        ------------------------
        #                                       |  torso  |  19   |  12  |
        #                                        ------------------------
        #                                                  LEGS
        #         ---------------------------------------------------------------------------------------------------
        #        |   Name  | q_id  | j_id |   Name  | q_id  | j_id |   Name  | q_id  | j_id |   Name  | q_id  | j_id |
        #         ---------------------------------------------------------------------------------------------------
        #        | l_hip_y |   7   |  0   | l_hip_r |  8    |  1   | r_hip_y |  13   |  6   | r_hip_r |  14   |  7   |
        #         ---------------------------------------------------------------------------------------------------
        #        | l_hip_p |   9   |  2   | l_knee  |  10   |  3   | r_hip_p |  15   |  8   | r_knee  |  16   |  9   |
        #         ---------------------------------------------------------------------------------------------------
        #        |l_ankle_p|   11  |  4   |l_ankle_r|  12   |  5   |r_ankle_p|  17   |  10  |r_ankle_r|  18   |  11  |
        #         ---------------------------------------------------------------------------------------------------
        #                                                  ARMS
        #  ---------------------------------------------------------------------------------------------------------------
        # |    Name    | q_id  | j_id |     Name   | q_id  | j_id |    Name    | q_id  | j_id |     Name   | q_id  | j_id |
        #  ---------------------------------------------------------------------------------------------------------------
        # |l_shoulder_p|   20  |  13  |l_shoulder_r|  21   |  14  |r_shoulder_p|   24  |  17  |r_shoulder_r|  25   |  18  |
        #  ---------------------------------------------------------------------------------------------------------------
        # |l_shoulder_y|   22  |  15  |   l_elbow  |  23   |  16  |r_shoulder_y|   26  |  19  |   r_elbow  |  27   |  20  |
        #  ---------------------------------------------------------------------------------------------------------------
        ###############################PyBullet Manual Tuning(Joint ID table)##############################################
        #  ---------------------------
        # |  r_finger  |   28  |  21  |
        #  ---------------------------

        self.q0[2]+=1e-8 # increase height for pybullet simulation
        self.x0 = np.concatenate([self.q0, self.dq0])
        self.model.defaultState = np.concatenate([self.q0, np.zeros((self.model.nv, 1))])
        if display:
            self.show()
        # Calculate initial pose
        # Define IK task

        lf = self.data_o.oMf[self.lSoleId_o].copy()
        rf = self.data_o.oMf[self.rSoleId_o].copy()
        com = m2a(pinocchio.centerOfMass(self.model_o, self.data_o, self.q0))

        # Define feet translational task
        rf.translation[1] -= 0.07
        # rf.translation[0] -=0.07
        lf.translation[1] += 0.01

        # Define feet rotational task
        rf = rf*pinocchio.SE3(rotate('z', np.pi/180*10), zero(3))
        lf = lf*pinocchio.SE3(rotate('z', np.pi/180*35), zero(3))

        # Define com task
        com[1] -= 0.03
        com[0] -= 0.03
        com[2] -= 0.01

        # Solve IK
        #self.q0= self.solveIK(self.q0, lf,rf,com, display=self.display)

        self.q0[20] = 0./180.*np.pi
        self.q0[24] = -45./180.*np.pi

        self.q0[21] = 15./180.*np.pi
        self.q0[25]= 60./180.*np.pi

        self.q0[22] = -(np.pi/2-15./180.*np.pi)
        self.q0[26] = 5./180.*np.pi

        self.q0[23] = (np.pi-163./180.*np.pi)
        self.q0[27] = 5./180.*np.pi

        self.q0[19] = 5./180.*np.pi

        if display:
            # self.robot_o.display(q)
            self.visualizeConfig(self.q0)
        self.x0 = np.concatenate([self.q0, self.dq0])
        self.model.defaultState = np.concatenate([self.q0, np.zeros((self.model.nv, 1))])
        self.model_o.defaultState = np.concatenate([self.q0, np.zeros((self.model.nv, 1))])
        self.q0_legacy = self.q0.copy()
        rgbt = [1.0, 0.2, 0.2, 1.0]  # red, green, blue, transparency
        # self.robot_o.viewer.gui.addSphere("world/sphere", .01, rgbt)  # .1 is the radius
        time.sleep(0.1)
        # self.show()
    
    def clearPose(self):
        self.q0 = self.model_o.referenceConfigurations["standing"].copy()
        #arm
        self.q0[20] = 0./180.*np.pi
        self.q0[24] = -250./180.*np.pi # r_shoulder_p
        self.q0[25]= 45./180.*np.pi # r_shoulder_r
        self.q0[26] = 135./180.*np.pi # r_shoulder_y
        self.q0[27]= 5./180.*np.pi # r_elbow
        self.q0[21] = 15./180.*np.pi     
        self.q0[22] = -(np.pi/2-15./180.*np.pi)
        self.q0[23] = (np.pi-163./180.*np.pi)
        
        
        #leg
        self.q0[9] = self.q0[15]= np.pi/12.
        self.q0[10] = self.q0[16]= -np.pi/6.
        self.q0[11] = self.q0[17]= -np.pi/12.
        self.calibrateOrigin()
        # if self.display:
        #     self.robot_o.display(self.q0)
    
    def clearPose_no_ball(self):
        self.q0 = self.model.referenceConfigurations["standing"].copy()
        #arm
        self.q0[20] = 0./180.*np.pi
        self.q0[24] = -45./180.*np.pi
        self.q0[21] = 15./180.*np.pi
        self.q0[25]= 45./180.*np.pi
        self.q0[22] = -(np.pi/2-15./180.*np.pi)
        self.q0[26] = np.pi/2-15./180.*np.pi
        self.q0[23] = (np.pi-163./180.*np.pi)
        self.q0[27]= (np.pi-163./180.*np.pi)
        
        #leg
        self.q0[9] = self.q0[15]= np.pi/12.
        self.q0[10] = self.q0[16]= -np.pi/6.
        self.q0[11] = self.q0[17]= -np.pi/12.
        self.calibrateOrigin()
        if self.display:
            self.robot.display(self.q0)   
    
    def show(self):
            time.sleep(10)
            # self.viz.initViewer(loadModel=True)
            # self.robot_o.viewer.gui.addFloor('hpp-gui/floor')
            self.viz.display(self.q0)
            # cameraTF = [1.53, 0.22, 1.08, 0.35, 0.41, 0.63, 0.55]
            # # cameraTF = [1.2, 1.5, 0.5, 0.2, 0.62, 0.72, 0.22]
            # self.robot_o.viewer.gui.setCameraTransform(self.robot_o.viz.windowID, cameraTF)
    def setCamera(self, view='front'):
        if view=='front':
            cameraTF = [1.53, 0.22, 1.08, 0.35, 0.41, 0.63, 0.55]
        elif view == 'side':
            cameraTF = [0.20751619338989258, -1.578794240951538, 0.9112874269485474, 0.5566732883453369, 0.04601140320301056, 0.057916779071092606, 0.827431857585907]
        elif view == 'back':
            cameraTF = [-1.4865986108779907, -0.3901124894618988, 0.7630850672721863, 0.48666736483573914, -0.35458531975746155, -0.4900544583797455, 0.630294144153595]
        elif view == 'normal':
            cameraTF = [0.752, -0.86999797821, 1.185777187347412, 0.4218546450138092, 0.13384711742401123, 0.32200172543525696, 0.8369220495223999]
        # self.robot_o.viewer.gui.setCameraTransform(self.robot_o.viz.windowID, cameraTF)
    def getCamera(self):
        for i in range(10000):
            # cameraTF = self.robot_o.viewer.gui.getCameraTransform(self.robot_o.viz.windowID)
            # print(cameraTF)
            time.sleep(1)

    def calibrateOrigin(self):
        pinocchio.forwardKinematics(self.model_o, self.data_o, self.q0)
        pinocchio.forwardKinematics(self.model, self.data, self.q0)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        pinocchio.updateFramePlacements(self.model, self.data)

        lfPos_o = self.data_o.oMf[self.lSoleId_o].translation
        # lfPos = self.data.oMf[self.lSoleId].translation
        rfPos_o = self.data_o.oMf[self.rSoleId_o].translation
        # rfPos = self.data.oMf[self.rSoleId].translation
        self.q0[:3]-=(lfPos_o+rfPos_o)/2
        pinocchio.forwardKinematics(self.model_o, self.data_o, self.q0)
        pinocchio.forwardKinematics(self.model, self.data, self.q0)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        pinocchio.updateFramePlacements(self.model, self.data)
        lfPos_o = self.data_o.oMf[self.lSoleId_o].translation
        # lfPos = self.data.oMf[self.lSoleId].translation
        rfPos_o = self.data_o.oMf[self.rSoleId_o].translation
        # rfPos = self.data.oMf[self.rSoleId].translation
        originPos0 = (lfPos_o+rfPos_o)/2
        assert abs(originPos0[0])<1.e-6, "[Error]Origin is not zeroed in x direction!"
        assert abs(originPos0[1])<1.e-6, "[Error]Origin is not zeroed in y direction!"
        assert abs(originPos0[2])<1.e-6, "[Error]Origin is not zeroed in z direction!"
    def calibrateOriginZ(self, q):
        pinocchio.forwardKinematics(self.model_o, self.data_o, q)
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        pinocchio.updateFramePlacements(self.model, self.data)

        lfPos_o = self.data_o.oMf[self.lSoleId_o].translation
        # lfPos = self.data.oMf[self.lSoleId].translation
        rfPos_o = self.data_o.oMf[self.rSoleId_o].translation
        # rfPos = self.data.oMf[self.rSoleId].translation
        # print(q, q.shape, type(q), lfPos_o, lfPos_o.shape, type(lfPos_o))
        q[2, 0]-=(lfPos_o[2, 0]+rfPos_o[2, 0])/2
        pinocchio.forwardKinematics(self.model_o, self.data_o, q)
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        pinocchio.updateFramePlacements(self.model, self.data)
        lfPos_o = self.data_o.oMf[self.lSoleId_o].translation
        # lfPos = self.data.oMf[self.lSoleId].translation
        rfPos_o = self.data_o.oMf[self.rSoleId_o].translation
        # rfPos = self.data.oMf[self.rSoleId].translation
        originPos0 = (lfPos_o[2, 0]+rfPos_o[2, 0])/2
        assert abs(originPos0)<1.e-6, "[Error]Origin is not zeroed in z direction!"
    
    def visualizeConfig(self,q, align=False):
        # def showGepetto():
        #     subprocess.call(["gepetto-gui"])
        
        # try:
        #     thread = threading.Thread(target=showGepetto)
        #     thread.start()
        #     #thread.join()
        # except:
        #     print("Error: unable to start Gepetto-GUI thread")
        # time.sleep(5)
        # self.robot.initViewer(loadModel=True)
        # self.robot.viewer.gui.addFloor('hpp-gui/floor')
        # size_q = np.shape(q)[0]
        # if size_q == self.nq-7:
        #     torso_pose = np.array([0,0,0.5, 0,0,0,1])
        #     q = np.hstack((torso_pose,q))
        pinocchio.forwardKinematics(self.model_o, self.data_o, q)
        pinocchio.computeJointJacobians(self.model_o, self.data_o, q)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        com = m2a(pinocchio.centerOfMass(self.model_o, self.data_o, q))
        

        # ensure one foot must touch ground
        if align:
            lfPos_o = self.data_o.oMf[self.lSoleId_o].translation
            rfPos_o = self.data_o.oMf[self.rSoleId_o].translation
            lfPos_z = lfPos_o[2, 0]
            rfPos_z = rfPos_o[2, 0]
            lrPos_z_min = min(lfPos_z, rfPos_z)
            if abs(lrPos_z_min)>1.e-6:
                # print("one foot must touch ground! align foot")
                q[2,0]-=lrPos_z_min

        
        # self.robot_o.viewer.gui.applyConfiguration("world/sphere", (com[0], com[1], 0., 1.,0.,0.,0. ))
        # self.robot_o.viewer.gui.refresh()  # Refresh the window.
        self.viz.display(q)


    def generateSwData(self, x):
        '''
        This function is used to generate solidworks data type
        '''
        q = np.asarray(x).copy()
        quat = q[3:7].copy()
        # Convert quaternion to rpy
        vector = np.matrix([0, 0, 0, quat[0], quat[1], quat[2], quat[3]]).T
        se3 = pinocchio.XYZQUATToSE3(vector)
        rpy = matrixToRpy(se3.rotation)
        q[3] = rpy[0]
        q[4] = rpy[1]
        q[5] = rpy[2]
        q[6] = 0

        # Process unit, for distance convert m to mm
        # Process unit, for angle, convert radian to degree
        # print(q[0:3])
        q[0] += 0.0264
        q[1] += 0.
        q[2] += 0.50958
        for i in range(3):
            q[i] *=1000.
            q[i] = np.abs(q[i])
        
        for i in range(3, self.nq):
            q[i] *=180./np.pi
            if q[i] < 0:
                q[i] +=360.
        return q 

    def saveEquation(self, x, savePath):
        '''
        Legacy function
        This function is used to generate configuration file with Solidworks in "equation" interface
        '''

        q = self.generateSwData(x)

        f = open(savePath, "w+")
        for i in range(self.nq):
            f.write('"q%d" = %f\r\n'%(i, q[i]))
        f.write('\n')

        names = []
        names.append('"D1@q0_x"= "q0"\r\n')
        names.append('"D1@q1_y"= "q1"\r\n')
        names.append('"D1@q2_z"= "q2"\r\n')
        names.append('"D1@q3_x"= "q3"\r\n')
        names.append('"D1@q4_y"= "q4"\r\n')
        names.append('"D1@q5_z"= "q5"\r\n')
        names.append('"D1@q7_l_hip_y"= "q7"\r\n')
        names.append('"D1@q8_l_hip_r"= "q8"\r\n')
        names.append('"D1@q9_l_hip_p"= "q9"\r\n')
        names.append('"D1@q10_l_knee"= "q10"\r\n')
        names.append('"D1@q11_l_ankle_p"= "q11"\r\n')
        names.append('"D1@q12_l_ankle_r"= "q12"\r\n')
        names.append('"D1@q13_l_shoulder_p"= "q13"\r\n')
        names.append('"D1@q14_l_shoulder_r"= "q14"\r\n')
        names.append('"D1@q15_l_elbow"= "q15"\r\n')
        names.append('"D1@q16_r_hip_y"= "q16"\r\n')
        names.append('"D1@q17_r_hip_r"= "q17"\r\n')
        names.append('"D1@q18_r_hip_p"= "q18"\r\n')
        names.append('"D1@q19_r_knee"= "q19"\r\n')
        names.append('"D1@q20_r_ankle_p"= "q20"\r\n')
        names.append('"D1@q21_r_ankle_r"= "q21"\r\n')
        names.append('"D1@q22_r_shoulder_p"= "q22"\r\n')
        names.append('"D1@q23_r_shoulder_r"= "q23"\r\n')
        names.append('"D1@q24_r_elbow"= "q24"\r\n')
        for i in range(self.nq-1):
            f.write(names[i])

    def saveConfig(self, x, savePath):
        '''
        This function is used to generate configuration file with Solidworks in "configuration" interface
        '''
        q = self.generateSwData(x)
        
        import xlsxwriter

        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(savePath)
        worksheet = workbook.add_worksheet()

        # Some data we want to write to the worksheet.
        jointName = []
        jointName.append('D1@q0_x')
        jointName.append('D1@q1_y')
        jointName.append('D1@q2_z')
        jointName.append('D1@q3_x')
        jointName.append('D1@q4_y')
        jointName.append('D1@q5_z')
        jointName.append('null')

        jointName.append('D1@q7_l_hip_y')
        jointName.append('D1@q8_l_hip_r')
        jointName.append('D1@q9_l_hip_p')
        jointName.append('D1@q10_l_knee')
        jointName.append('D1@q11_l_ankle_p')
        jointName.append('D1@q12_l_ankle_r')

        jointName.append('D1@q13_l_shoulder_p')
        jointName.append('D1@q14_l_shoulder_r')
        jointName.append('D1@q15_l_shoulder_y')
        jointName.append('D1@q16_l_elbow')

        jointName.append('D1@q17_r_hip_y')
        jointName.append('D1@q18_r_hip_r')
        jointName.append('D1@q19_r_hip_p')
        jointName.append('D1@q20_r_knee')
        jointName.append('D1@q21_r_ankle_p')
        jointName.append('D1@q22_r_ankle_r')

        jointName.append('D1@q23_r_shoulder_p')
        jointName.append('D1@q24_r_shoulder_r')
        jointName.append('D1@q25_r_shoulder_y')
        jointName.append('D1@q26_r_elbow')

        q = q.tolist()

        # Start from the first cell. Rows and columns are zero indexed.
        row = 0
        col = 0
        from sets import Set
        inverse = Set(['D1@q20_r_knee',
                        'D1@q9_l_hip_p'])
        flip = Set(['D1@q10_l_knee',
                        'D1@q23_r_shoulder_p'])
        turn = Set(['D1@q15_l_shoulder_y'])                
        # Iterate over the data and write it out row by row.
        for item, value in zip(jointName, q):

            if item in inverse:
                value = 360 -value
            if item in flip:
                value = 180 + value
            if item in turn:
                value = 90+value
            if item =='null':
                pass
            else:
                worksheet.write(row, col, item)
                worksheet.write(row+1, col, value)
                col += 1

        workbook.close()

    def readConfig(self, filePath):
        '''
        This function is used to read configuration file from Solidworks in "configuration" interface
        '''
        import xlrd

        wb = xlrd.open_workbook(filePath) 
        sheet = wb.sheet_by_index(0)
        joint_values = []
        joint_names = []
        from sets import Set
        inverse = Set(['D1@q20_r_knee',
                        'D1@q9_l_hip_p'])
        flip = Set(['D1@q10_l_knee',
                        'D1@q23_r_shoulder_p'])
        turn = Set(['D1@q15_l_shoulder_y'])
        row =0
        col =0
        for i in range(self.nq):
            if i ==6:
                joint_values.append(0)
                joint_names.append('null')
            else:
                name = sheet.cell_value(row, col)
                value = sheet.cell_value(row+1, col)
                if name in inverse:
                    value = 360-value
                if name in flip:
                    value = 180 + value
                if name in turn:
                    value = 90 + value
                joint_names.append(name)
                joint_values.append(value)
                col +=1
        
        for i in range(3, self.nq):
            if joint_values[i] > 180:
                joint_values[i] -=360

            joint_values[i] *=np.pi/180.
        
        for i in range(3):
            joint_values[i] *=0.001

        joint_values[0] -= 0.0264
        joint_values[1] -= 0.
        joint_values[2] -= 0.50958
        
        # Convert quaternion to rpy
        rpy = np.asarray(joint_values[3:6])
        se3 = pinocchio.SE3.Identity()
        se3.translation = np.asarray(joint_values[0:3])
        se3.rotation = rpyToMatrix(rpy)
        xyzquaternion = pinocchio.SE3ToXYZQUAT(se3).T.tolist()[0]
        joint_values[0:7] = xyzquaternion
        # print(joint_values)

        joint_values = np.asarray(joint_values)
        return joint_values
    
    def getLimit(self):
        u = []
        l = []
        # l_hip_y
        u.append(np.pi/2)  # fore to inward
        l.append(-np.pi/6) # fore to outward
        # l_hip_r
        u.append(np.pi/4) #lift left
        l.append(-np.pi/4) #lift right
        # l_hip_p
        u.append(np.pi/2) #bend forward
        l.append(-np.pi/6) # bend backward
        # l_knee
        u.append(0) # bend forward impossible
        l.append(-np.pi/2-np.pi/10) # bend backward
        # l_ankle_p
        u.append(np.pi/6) # rear lift
        l.append(-np.pi/2)# fore lift
        # l_ankle_r
        u.append(np.pi/4) # lift right
        l.append(-np.pi/4)# lift left
    
        # r_hip_y
        u.append(np.pi/6)  # fore to outward
        l.append(-np.pi) # fore to inward
        # r_hip_r
        u.append(np.pi/4) #lift left
        l.append(-np.pi/4) #lift right
        # r_hip_p
        u.append(np.pi/2) #bend forward
        l.append(-np.pi/6) # bend backward
        # r_knee
        u.append(0) # bend forward impossible
        l.append(-np.pi/2-np.pi/10) # bend backward
        # r_ankle_p
        u.append(np.pi/6) # rear lift
        l.append(-np.pi/2)# fore lift
        # r_ankle_r
        u.append(np.pi/4) # lift right
        l.append(-np.pi/4)# lift left
        # waist
        u.append(np.pi/2)
        l.append(-np.pi/2)
        # l_shoulder_p
        u.append(np.pi*2) #  arm backward
        l.append(-np.pi*2)#  arm forward
        # l_shoulder_r
        u.append(np.pi) #  arm left
        l.append(-np.pi/18)#  arm right
        # l_shoulder_y
        u.append(np.pi*2) #  elbow left
        l.append(-np.pi*2)#  elbow right
        # l_elbow
        u.append(np.pi/2+np.pi/6) #  elbow bend inward
        l.append(0)#  elbow bend outward impossible

        # r_shoulder_p
        u.append(np.pi*2) #  arm backward
        l.append(-np.pi*2)#  arm forward
        # r_shoulder_r
        u.append(np.pi) #  arm right
        l.append(-np.pi/18)#  arm left
        # r_shoulder_y
        u.append(np.pi*2) #  elbow left
        l.append(-np.pi*2)#  elbow right
        # r_elbow
        u.append(np.pi/2+np.pi/6) #  elbow bend inward
        l.append(0)#  elbow bend outward impossible
        
        # r_finger
        u.append(2.093) #  elbow bend inward
        l.append(-1)#  elbow bend outward impossible

        return l, u
    
    def getVelLimit(self):
        u = []
        # Mx106:4.7
        # Mx64:6.6
        # Mx28:5.8
        vel_106 =4.7
        vel_64 = 6.6
        vel_28 =5.8
        # l_hip_y
        u.append(vel_64)  # fore to inward
        # l_hip_r
        u.append(vel_106) #lift left
        # l_hip_p
        u.append(vel_106) #bend forward
        # l_knee
        u.append(vel_106) # bend forward impossible
        # l_ankle_p
        u.append(vel_106) # rear lift
        # l_ankle_r
        u.append(vel_106) # lift right
        
        # r_hip_y
        u.append(vel_64)  # fore to outward
        # r_hip_r
        u.append(vel_106) #lift left
        # r_hip_p
        u.append(vel_106) #bend forward
        # r_knee
        u.append(vel_106) # bend forward impossible
        # r_ankle_p
        u.append(vel_106) # rear lift
        # r_ankle_r
        u.append(vel_106) # lift right

        # torso
        u.append(vel_106) # torso

        # l_shoulder_p
        u.append(vel_64) #  arm backward
        # l_shoulder_r
        u.append(vel_64) #  arm left
        # l_shoulder_y
        u.append(vel_28) #  elbow left
        # l_elbow
        u.append(vel_28) #  elbow bend inward

        # r_shoulder_p
        u.append(vel_64) #  arm backward
        # r_shoulder_r
        u.append(vel_64) #  arm right
        # r_shoulder_y
        u.append(vel_28) #  elbow left
        # r_elbow
        u.append(vel_28) #  elbow bend inward
        # r_finger
        u.append(vel_28) #  elbow bend inward
        return u
    
    def getTorqueLimit(self):
        torqueLower = []
        torqueUpper = []

        torqueLower.append(-6.0)
        torqueUpper.append(6.0)
        for i in range(5):
            torqueLower.append(-8.4)
            torqueUpper.append(8.4)

        torqueLower.append(-6.0)
        torqueUpper.append(6.0)
        for i in range(6):
            torqueLower.append(-8.4)
            torqueUpper.append(8.4) 

        for i in range(2):
            torqueLower.append(-6.0)
            torqueUpper.append(6.0) 
        for i in range(2):
            torqueLower.append(-2.5)
            torqueUpper.append(2.5)
            
        for i in range(2):
            torqueLower.append(-6.0)
            torqueUpper.append(6.0)
        for i in range(3):
            torqueLower.append(-2.5)
            torqueUpper.append(2.5)
        return torqueLower, torqueUpper

    def solveIK(self, q_init, refLf, refRf, refCom,refTorsoOri, display=False, debug=True, ignoreUpper=False):
        '''
        This function is used to solve inverse kinematics problem
        to satisfy the predefined feet pose and com
        lfPos \in SE3
        rfPos \in SE3
        com \in R^3
        '''
        def getProjector(J):
            I = np.eye(len(J[0]))

            N = I - np.linalg.pinv(J).dot(J)

            return N
        
        T_left_foot = self.data_o.oMf[self.lSoleId_o]
        T_right_foot = self.data_o.oMf[self.rSoleId_o]
        # print(T_left_foot, T_right_foot)
        com = m2a(pinocchio.centerOfMass(self.model_o, self.data_o, q_init))
        T_torso_ = self.data_o.oMf[self.torsoId_o].rotation
        # refTorso= T_torso_.copy()
        refTorso = pinocchio.rpy.rpyToMatrix(refTorsoOri)

        tmp_left = T_left_foot.inverse() * refLf
        Vs_left = m2a(pinocchio.log(tmp_left).vector)

        tmp_right = T_right_foot.inverse() * refRf
        Vs_right = m2a(pinocchio.log(tmp_right).vector)

        tmp_com = refCom - com
        residue_com = sum(tmp_com ** 2)
        # print(T_torso_, refTorso)
        Vs_torso = m2a(pinocchio.log3(np.linalg.inv(T_torso_).dot(refTorso)))

        eomg = 1e-4
        ev = 1e-2
        ecom = 0.5e-5
        etorso = 1e-4
        j = 0
        maxiterations = 10000
        epsilon = 0.01

        err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
        err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
        err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
        err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])
        err_torso = np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) # loose constraint of torso

        err = err_left_trans > eomg or err_left_orien > ev or err_right_trans > eomg or err_right_orien > ev or err_torso > ev or residue_com > ecom
        q = q_init.copy()
        if debug:
            print('**********************************Solve Inverse Kinematic Problem****************************************')
            print('         lf trans err     lf orien err    rf trans err    rf orien err     com err')
            print('tolerance  %.4f          %.4f           %.4f          %.4f         %.5f'\
                %(eomg, ev, eomg, ev, ecom))
            print
            print('iter     lf trans err     lf orien err    rf trans err    rf orien err     com err  torso orien err  done')
            print('%04d        %.4f          %.4f            %.4f         %.4f        %.5f      %.4f       %r'\
                %(j, err_left_trans, err_left_orien, err_right_trans, err_right_orien, residue_com, err_torso, not err))

        while err and j < maxiterations:

            # construct J matrix
            J = np.empty([15, self.nv])
            J_2 = pinocchio.jacobianCenterOfMass(self.model_o, self.data_o, q)
            J[3:9, :] = pinocchio.getFrameJacobian(self.model_o, self.data_o, self.rSoleId_o, pinocchio.ReferenceFrame.WORLD)
            J[9:15, :] = pinocchio.getFrameJacobian(self.model_o, self.data_o, self.lSoleId_o, pinocchio.ReferenceFrame.WORLD)
            J[0:3, :] = pinocchio.getFrameJacobian(self.model_o, self.data_o, self.torsoId_o, pinocchio.ReferenceFrame.WORLD)[3:6, :]
            Vs = np.zeros([15, 1])

            Vs[3:9, 0] = Vs_right
            Vs[9:15, 0] = Vs_left

            Vs_2 = np.zeros([3, 1])
            Vs_2[0:3, 0] = tmp_com
            Vs[0:3, 0] = Vs_torso

            # calculate dq
            tmp_item = np.linalg.pinv(J_2.dot(getProjector(J)))
            tmp_item_2 = Vs_2 - J_2.dot(np.linalg.pinv(J)).dot(Vs)
            dq = np.linalg.pinv(J).dot(Vs) + getProjector(J).dot(tmp_item.dot(tmp_item_2))
            if ignoreUpper:
                dq[18:] = 0.
            dq = dq * epsilon

            q = pinocchio.integrate(self.model_o, q, dq)

            j = j + 1

            pinocchio.forwardKinematics(self.model_o, self.data_o, q)
            pinocchio.computeJointJacobians(self.model_o, self.data_o, q)
            pinocchio.updateFramePlacements(self.model_o, self.data_o)

            T_left_foot = self.data_o.oMf[self.lSoleId_o]
            T_right_foot = self.data_o.oMf[self.rSoleId_o]
            com = m2a(pinocchio.centerOfMass(self.model_o, self.data_o, q))
            T_torso_ = self.data_o.oMf[self.torsoId_o].rotation

            tmp_left = T_left_foot.inverse() * refLf
            Vs_left = m2a(pinocchio.log(tmp_left).vector)

            tmp_right = T_right_foot.inverse() * refRf
            Vs_right = m2a(pinocchio.log(tmp_right).vector)

            tmp_com = refCom - com
            residue_com = sum(tmp_com ** 2)

            Vs_torso = m2a(pinocchio.log3(np.linalg.inv(T_torso_).dot(refTorso)))

            err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
            err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
            err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
            err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])
            err_torso = np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) # loose constraint of torso

            err = err_left_trans > eomg or err_left_orien > ev or err_right_trans > eomg or err_right_orien > ev or err_torso> ev or residue_com > ecom
            if j%100==0 or not err:
                if debug:
                    print('%04d        %.4f          %.4f            %.4f         %.4f        %.5f      %.4f       %r'\
                        %(j, err_left_trans, err_left_orien, err_right_trans, err_right_orien, residue_com, err_torso, not err))

            if display:
                if j%10==0 or not err:
                    self.visualizeConfig(q)
                    # self.robot_o.display(q)
                    
                    # cameraTF = self.robot_o.viewer.gui.getCameraTransform(self.robot_o.viz.windowID)
                    # print(cameraTF)

                    
        # restore the memory
        pinocchio.forwardKinematics(self.model_o, self.data_o, q_init)
        pinocchio.computeJointJacobians(self.model_o, self.data_o, q_init)
        pinocchio.updateFramePlacements(self.model_o, self.data_o)
        if not err:
            # self.q0 = q
            # torsoBase = pinocchio.XYZQUATToSE3(self.q0[:7])
            # torsoRot = torsoBase*pinocchio.SE3(rotate('z', -np.pi/4-np.pi/18), zero(3))
            # torsoQuat = pinocchio.se3ToXYZQUAT(torsoRot)
            # self.q0[:7] = torsoQuat[:7]
            return q.copy(), err
        else:
            print('IK failed!')
            return q_init.copy(), err
    
    def solveIK_no_ball(self, refLf, refRf, refCom, display=False):
        '''
        This function is used to solve inverse kinematics problem
        to satisfy the predefined feet pose and com
        lfPos \in SE3
        rfPos \in SE3
        com \in R^3
        '''
        def getProjector(J):
            I = np.eye(len(J[0]))

            N = I - np.linalg.pinv(J).dot(J)

            return N
        
        T_left_foot = self.data.oMf[self.lSoleId]
        T_right_foot = self.data.oMf[self.rSoleId]
        print(T_left_foot, T_right_foot)
        com = m2a(pinocchio.centerOfMass(self.model, self.data, self.q0))
        T_torso_ = self.data.oMf[self.torsoId].rotation
        refTorso= T_torso_.copy()

        tmp_left = T_left_foot.inverse() * refLf
        Vs_left = m2a(pinocchio.log(tmp_left).vector)

        tmp_right = T_right_foot.inverse() * refRf
        Vs_right = m2a(pinocchio.log(tmp_right).vector)

        tmp_com = refCom - com
        residue_com = sum(tmp_com ** 2)
        # print(T_torso_, refTorso)
        Vs_torso = m2a(pinocchio.log3(np.linalg.inv(T_torso_).dot(refTorso)))

        eomg = 1e-4
        ev = 1e-2
        ecom = 0.5e-5
        etorso = 1e-4
        j = 0
        maxiterations = 10000
        epsilon = 0.01

        err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
        err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
        err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
        err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])
        err_torso = np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) # loose constraint of torso

        err = err_left_trans > eomg or err_left_orien > ev or err_right_trans > eomg or err_right_orien > ev or residue_com > ecom
        q = self.q0.copy()

        print('         lf trans err     lf orien err    rf trans err    rf orien err     com err')
        print('tolerance  %.4f          %.4f           %.4f          %.4f         %.5f'\
            %(eomg, ev, eomg, ev, ecom))
        print
        print('iter     lf trans err     lf orien err    rf trans err    rf orien err     com err  torso orien err  done')
        print('%04d        %.4f          %.4f            %.4f         %.4f        %.5f      %.4f       %r'\
            %(j, err_left_trans, err_left_orien, err_right_trans, err_right_orien, residue_com, err_torso, not err))

        while err and j < maxiterations:

            # construct J matrix
            J = np.empty([15, self.nv])
            J_2 = pinocchio.jacobianCenterOfMass(self.model, self.data, q)
            J[3:9, :] = pinocchio.getFrameJacobian(self.model, self.data, self.rSoleId, pinocchio.ReferenceFrame.WORLD)
            J[9:15, :] = pinocchio.getFrameJacobian(self.model, self.data, self.lSoleId, pinocchio.ReferenceFrame.WORLD)
            J[0:3, :] = pinocchio.getFrameJacobian(self.model, self.data, self.torsoId, pinocchio.ReferenceFrame.WORLD)[3:6, :]
            Vs = np.zeros([15, 1])

            Vs[3:9, 0] = Vs_right
            Vs[9:15, 0] = Vs_left

            Vs_2 = np.zeros([3, 1])
            Vs_2[0:3, 0] = tmp_com
            Vs[0:3, 0] = Vs_torso

            # calculate dq
            tmp_item = np.linalg.pinv(J_2.dot(getProjector(J)))
            tmp_item_2 = Vs_2 - J_2.dot(np.linalg.pinv(J)).dot(Vs)
            dq = np.linalg.pinv(J).dot(Vs) + getProjector(J).dot(tmp_item.dot(tmp_item_2))
            dq = dq * epsilon

            q = pinocchio.integrate(self.model, q, dq)

            j = j + 1

            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.computeJointJacobians(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)

            T_left_foot = self.data.oMf[self.lSoleId]
            T_right_foot = self.data.oMf[self.rSoleId]
            com = m2a(pinocchio.centerOfMass(self.model, self.data, q))
            T_torso_ = self.data.oMf[self.torsoId].rotation

            tmp_left = T_left_foot.inverse() * refLf
            Vs_left = m2a(pinocchio.log(tmp_left).vector)

            tmp_right = T_right_foot.inverse() * refRf
            Vs_right = m2a(pinocchio.log(tmp_right).vector)

            tmp_com = refCom - com
            residue_com = sum(tmp_com ** 2)

            Vs_torso = m2a(pinocchio.log3(np.linalg.inv(T_torso_).dot(refTorso)))

            err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
            err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
            err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
            err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])
            err_torso = np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) # loose constraint of torso

            err = err_left_trans > eomg or err_left_orien > ev or err_right_trans > eomg or err_right_orien > ev or residue_com > ecom
            if j%100==0 or not err:
                print('%04d        %.4f          %.4f            %.4f         %.4f        %.5f      %.4f       %r'\
                    %(j, err_left_trans, err_left_orien, err_right_trans, err_right_orien, residue_com, err_torso, not err))

            if display:
                if j%10==0 or not err:
                    self.robot.display(q)
                    time.sleep(0.1)
        # restore the memory
        pinocchio.forwardKinematics(self.model, self.data, self.q0)
        pinocchio.computeJointJacobians(self.model, self.data, self.q0)
        pinocchio.updateFramePlacements(self.model, self.data)
        if not err:
            self.q0 = q
            self.x0 = np.concatenate([self.q0, self.dq0])
            self.model.defaultState = np.concatenate([self.q0, np.zeros((self.model.nv, 1))])
        else:
            print('IK failed!')



'''
# Visualize the robot model

import meshcat.geometry as g
import meshcat.transformations as tf

def createDisplay_o(targets):
    display = crocoddyl.MeshcatDisplay(robot_o, 4, 4, False)
    for i, target in enumerate(targets):
        display.robot.viewer["target_" + str(i)].set_object(g.Sphere(0.025))
        Href = np.array([[1., 0., 0., target[0]],
                         [0., 1., 0., target[1]],
                         [0., 0., 1., target[2]],
                         [0., 0., 0., 1.]])
        display.robot.viewer["target_" + str(i)].set_transform(np.array([[1., 0., 0., target[0]],
                         [0., 1., 0., target[1]],
                         [0., 0., 1., target[2]],
                         [0., 0., 0., 1.]]))
    return display
def createDisplay(targets):
    display = crocoddyl.MeshcatDisplay(robot, 4, 4, False)
    for i, target in enumerate(targets):
        display.robot.viewer["target_" + str(i)].set_object(g.Sphere(0.025))
        Href = np.array([[1., 0., 0., target[0]],
                         [0., 1., 0., target[1]],
                         [0., 0., 1., target[2]],
                         [0., 0., 0., 1.]])
        display.robot.viewer["target_" + str(i)].set_transform(np.array([[1., 0., 0., target[0]],
                         [0., 1., 0., target[1]],
                         [0., 0., 1., target[2]],
                         [0., 0., 0., 1.]]))
    return display

# Display target pose in jupyter notebook cell
# target = np.array([0.4, 0, 1.2])
# display = createDisplay(m.robot, [target])
# display.robot.viewer.jupyter_cell()
# m.robot.display(m.q0)

'''
class ReferenceState():
    def __init__(self, state):
        self.state_names = []
        
        self.state_names.append('x')
        self.state_names.append('y')
        self.state_names.append('z')
        self.state_names.append('rx')
        self.state_names.append('ry')
        self.state_names.append('rz')

        self.state_names.append('l_hip_y')
        self.state_names.append('l_hip_r')
        self.state_names.append('l_hip_p')
        self.state_names.append('l_knee')
        self.state_names.append('l_ankle_p')
        self.state_names.append('l_ankle_r')

        self.state_names.append('r_hip_y')
        self.state_names.append('r_hip_r')
        self.state_names.append('r_hip_p')
        self.state_names.append('r_knee')
        self.state_names.append('r_ankle_p')
        self.state_names.append('r_ankle_r')

        self.state_names.append('torso')

        self.state_names.append('l_shoulder_p')
        self.state_names.append('l_shoulder_r')
        self.state_names.append('l_shoulder_y')
        self.state_names.append('l_elbow')

        self.state_names.append('r_shoulder_p')
        self.state_names.append('r_shoulder_r')
        self.state_names.append('r_shoulder_y')
        self.state_names.append('r_elbow')

        self.state_names.append('v_x')
        self.state_names.append('v_y')
        self.state_names.append('v_z')
        self.state_names.append('v_rx')
        self.state_names.append('v_ry')
        self.state_names.append('v_rz')

        self.state_names.append('v_l_hip_y')
        self.state_names.append('v_l_hip_r')
        self.state_names.append('v_l_hip_p')
        self.state_names.append('v_l_knee')
        self.state_names.append('v_l_ankle_p')
        self.state_names.append('v_l_ankle_r')

        self.state_names.append('v_r_hip_y')
        self.state_names.append('v_r_hip_r')
        self.state_names.append('v_r_hip_p')
        self.state_names.append('v_r_knee')
        self.state_names.append('v_r_ankle_p')
        self.state_names.append('v_r_ankle_r')

        self.state_names.append('v_torso')

        self.state_names.append('v_l_shoulder_p')
        self.state_names.append('v_l_shoulder_r')
        self.state_names.append('v_l_shoulder_y')
        self.state_names.append('v_l_elbow')

        self.state_names.append('v_r_shoulder_p')
        self.state_names.append('v_r_shoulder_r')
        self.state_names.append('v_r_shoulder_y')
        self.state_names.append('v_r_elbow')

        self.reference_state = state.copy()
        self.stateWeights = np.array([0.] * 3 + [10.] * 3 
                                + [0.01]*6
                                + [0.01]*6
                                + [0.01]
                                + [0.01]*4
                                + [0.01]*4 
                                + [10.] * 27)
        self.config_id = {}
        self.weight_id = {}
        self._value = {}
        for i, (name) in enumerate(self.state_names):
            
            if i>5:
                j = i +1
            else:
                j = i
            self.config_id[name] = j
            self.weight_id[name] = i
            self._value[name] = [np.asscalar(self.reference_state[j]), np.asscalar(self.stateWeights[i])]
    @property
    def value(self):
        return self._value

    # @property
    # def weight(self):
    #     return self._weight

    def update(self):
        for name in self.state_names:
            config_id = self.config_id[name]
            weight_id = self.weight_id[name]
            self.reference_state[config_id] = self._value[name][0]
            self.stateWeights[weight_id] = self._value[name][1]
            

def main():
    VisualModel(display=True)

if __name__ == "__main__":
    main()