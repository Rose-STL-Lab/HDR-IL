import time
import pybullet as p
import numpy as np
import pandas as pd

def getJointNames(botId):

    for i in range(p.getNumJoints(botId)):
        print(p.getJointInfo(botId, i)[0:2])

def getRevoluteJoints(botId):

    revoluteJoints = []

    for i in range(p.getNumJoints(botId)):

        if p.getJointInfo(botId, i)[2] == 0:
            revoluteJoints.append(i)

    return revoluteJoints



def getJointStates(botId):

    jointStates = []
    revoluteJoints = getRevoluteJoints(botId)

    for joint in revoluteJoints:
        jointStates.append(p.getJointState(botId, joint)[0])

    return jointStates



def visualize(data):

    print(data.shape)
    print("test", data.iloc[0, 2])
    p.connect(p.GUI)

    p.resetDebugVisualizerCamera( cameraDistance=3, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,0])

    p.resetSimulation()
    p.setGravity(0,0,-9.81)
    p.setTimeStep(0.01)
    planeId = p.loadURDF("plane.xml")
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    baxterStartOrientation = p.getQuaternionFromEuler([0,0,0])

    botId = p.loadURDF("baxter.xml",
                    [0, 0, 0],
                    baxterStartOrientation, useFixedBase=1)

    p.setJointMotorControlArray(botId,
                                jointIndices=[13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 39, 40, 42, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75, -0.9, 0, 1.8, 0, -0.9, 0, -0.75, -0.9, 0, 1.8, 0, -0.9, 0])

    for i in range(10):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    p.setJointMotorControlArray(botId,
                                jointIndices=[20, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[1.8, 1.8])

    for i in range(10):
        p.stepSimulation()

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(10):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)

        for j in range(10):
            p.stepSimulation()

        count += 1

    for i in range(100):
        p.stepSimulation()

    tableId1 = p.loadURDF("table_dock.xml",
                      [1 + data.iloc[0, 14], 0.1 + data.iloc[0, 15], 0],
                      cubeStartOrientation,
                      globalScaling=0.75)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.55]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    for i in range(10):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)

        for i in range(10):
            p.stepSimulation()

        count += 1

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    inertpos, inertorientation = p.invertTransform(p.getDynamicsInfo(tableId1, -1)[3], p.getDynamicsInfo(tableId1, -1)[4])
    table1pos, table1orientation = p.multiplyTransforms(p.getBasePositionAndOrientation(tableId1)[0], p.getBasePositionAndOrientation(tableId1)[1], inertpos, inertorientation)

    for i in range(100):
        p.stepSimulation()

    for i in range(40):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)

        for j in range(10):
            p.stepSimulation()

        count += 1


    for i in range(100):
        p.stepSimulation()

    p.removeBody(tableId1)
    tableId1 = p.loadURDF("table_dock.xml",
                      table1pos,
                      table1orientation,
                      globalScaling=0.75)

    tableId2 = p.loadURDF("table_peg.xml",
                      [1 + data.iloc[0, 16], -0.1 + data.iloc[0, 17], 0],
                      cubeStartOrientation,
                      globalScaling=0.75)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.75]*4,
                                forces = [10000]*4)



    for i in range(1000):
        p.stepSimulation()

    for i in range(10):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)

        p.setJointMotorControlArray(botId,
                                    jointIndices=[29, 31, 52, 54],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.55]*4,
                                    forces = [10000]*4)

        for i in range(10):
            p.stepSimulation()

        count += 1

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[1.2]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    inertpos, inertorientation = p.invertTransform(p.getDynamicsInfo(tableId2, -1)[3], p.getDynamicsInfo(tableId2, -1)[4])
    table2pos, table2orientation = p.multiplyTransforms(p.getBasePositionAndOrientation(tableId2)[0], p.getBasePositionAndOrientation(tableId2)[1], inertpos, inertorientation)

    for i in range(100):
        p.stepSimulation()

    for i in range(40):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)

        for i in range(10):
            p.stepSimulation()

        count += 1

    for i in range(100):
        p.stepSimulation()

    p.removeBody(tableId2)
    tableId2 = p.loadURDF("table_peg.xml",
                      table2pos,
                      table2orientation,
                      globalScaling=0.75)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    for i in range(20):

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[100]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[100]*9)
        p.setJointMotorControlArray(botId,
                                    jointIndices=[29, 31, 52, 54],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.75]*4,
                                    forces = [10000]*4)
        count += 1

        for i in range(10):
            p.stepSimulation()

    for i in range(100):
        p.stepSimulation()

    if p.getBasePositionAndOrientation(tableId1)[0][2] > 0.4 and p.getBasePositionAndOrientation(tableId2)[0][2] > 0.4:

        label = 1

    else:

        label = 0

    if abs(table1orientation[2] - table2orientation[2]) < 0.05:

        orientation = 1

    else:

        orientation = 0

    p.disconnect()

    return label, orientation


#The following script visualizes the output predictions of our model on the peg in hole task.
def main():

    # data = pd.read_csv('projection a box.csv')
    data = pd.read_csv("lift_primitive_data.csv")

    # data = data[["right_gripper_pole_x_1", "right_gripper_pole_y_1", "right_gripper_pole_z_1",
    #         "right_gripper_pole_q_11", "right_gripper_pole_q_12", "right_gripper_pole_q_13", "right_gripper_pole_q_14",
    #         "left_gripper_pole_x_1", "left_gripper_pole_y_1", "left_gripper_pole_z_1",
    #         "left_gripper_pole_q_11", "left_gripper_pole_q_12", "left_gripper_pole_q_13", "left_gripper_pole_q_14",
    #         "right_gripper_pole_x_2", "right_gripper_pole_y_2", "right_gripper_pole_z_2",
    #         "right_gripper_pole_q_21", "right_gripper_pole_q_22", "right_gripper_pole_q_23", "right_gripper_pole_q_24",
    #         "left_gripper_pole_x_2", "left_gripper_pole_y_2", "left_gripper_pole_z_2",
    #         "left_gripper_pole_q_21", "left_gripper_pole_q_22", "left_gripper_pole_q_23", "left_gripper_pole_q_24",
    #         "x_displacement1", "y_displacement1", "x_displacement2", "y_displacement2"]]

    data = data[["right_gripper_pole_x_2", "right_gripper_pole_y_2", "right_gripper_pole_z_2",
            "right_gripper_pole_q_21", "right_gripper_pole_q_22", "right_gripper_pole_q_23", "right_gripper_pole_q_24",
            "left_gripper_pole_x_2", "left_gripper_pole_y_2", "left_gripper_pole_z_2",
            "left_gripper_pole_q_21", "left_gripper_pole_q_22", "left_gripper_pole_q_23", "left_gripper_pole_q_24",
            "x_displacement1", "y_displacement1", "x_displacement2", "y_displacement2",
            "table1_x_2", "table1_y_2", "table1_z_2", "table1_quat1_2", "table1_quat2_2", "table1_quat3_2", "table1_quat4_2",
            "table2_x_2", "table2_y_2", "table2_z_2", "table2_quat1_2", "table2_quat2_2", "table2_quat3_2", "table2_quat4_2"]]
    # data = data[["right_gripper_pole_x_1", "right_gripper_pole_y_1", "right_gripper_pole_z_1",
    #         "right_gripper_pole_q_11", "right_gripper_pole_q_12", "right_gripper_pole_q_13", "right_gripper_pole_q_14",
    #         "left_gripper_pole_x_1", "left_gripper_pole_y_1", "left_gripper_pole_z_1",
    #         "left_gripper_pole_q_11", "left_gripper_pole_q_12", "left_gripper_pole_q_13", "left_gripper_pole_q_14",
    #         "x_displacement1", "y_displacement1", "x_displacement2", "y_displacement2"]]
    # print(data["y_displacement2"])

    print(data.shape)

    label = 0
    orientation = 0

    for i in range(0, data.shape[0], 130):

        sub_dataset = data.iloc[i:i+130, :]
        temp_label, temp_orientation = visualize(sub_dataset)
        label += temp_label
        orientation += temp_orientation

    accuracy = label / (data.shape[0]/130)
    orientation_acc = orientation / (data.shape[0]/130)

    print("Accuracy: ", accuracy)
    print("Orientation Accuracy: ", orientation_acc)


if __name__=="__main__":
    main()
