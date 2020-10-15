import time

import pybullet as p
import numpy as np
import pandas as pd

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




def getPredictedGripperPositions():

    data = pd.read_csv("lift_primitive_data.csv")

    jointStates = []

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

    for i in range(100000000000000000):
        p.stepSimulation()

    revoluteJoints = getRevoluteJoints(botId)

    for i in range(70):

        gripperPosition1 = p.calculateInverseKinematics(botId, 29, [data.iloc[i, 2], data.iloc[i, 3], data.iloc[i, 4]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 52, [data.iloc[i, 5], data.iloc[i, 6], data.iloc[i, 7]], maxNumIterations=1000)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for j in range(10):
            p.stepSimulation()

        jointStates.append(getJointStates(botId))

    jointStates = pd.DataFrame(jointStates)

    jointStates.to_csv("joint_states.csv")









def visualize(data):

    # data = pd.read_csv("lift_primitive_data.csv")
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

    for i in range(10):
        p.stepSimulation()

    boxId = p.loadURDF("table.xml",
                      [0.9 + data.iloc[1, 21], data.iloc[1, 22], 0],
                      cubeStartOrientation)

    for i in range(10):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.55]*4,
                                forces = [10000]*4)

    for i in range(10):
        p.stepSimulation()

    for i in range(12):

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

        for j in range(10):
            p.stepSimulation()

        count += 1

    blockId = p.loadURDF("block.xml",
                      [1.6, 0, 0],
                      cubeStartOrientation,
                      useFixedBase=1)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.8]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    revoluteJoints = getRevoluteJoints(botId)
    right = list(p.getLinkState(botId, 51)[0])
    right[0] += 0.03
    left = list(p.getLinkState(botId, 28)[0])
    left[0] += 0.03

    gripperPosition1 = p.calculateInverseKinematics(botId, 28, left, maxNumIterations=1000)
    gripperPosition2 = p.calculateInverseKinematics(botId, 51, right, maxNumIterations=1000)

    p.setJointMotorControlArray(botId,
                                jointIndices=revoluteJoints[0:10],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=gripperPosition1[0:10])

    p.setJointMotorControlArray(botId,
                                jointIndices=revoluteJoints[10:],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=gripperPosition2[10:])

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.55]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    for i in range(48):

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

        if count < 40 and count > 30:

            gripperPosition1 = p.calculateInverseKinematics(botId, 28, [data.iloc[count, 0], data.iloc[count, 1], data.iloc[count, 2]], [data.iloc[count, 3], data.iloc[count, 4], data.iloc[count, 5], data.iloc[count, 6]], maxNumIterations=1000)
            gripperPosition2 = p.calculateInverseKinematics(botId, 51, [data.iloc[count, 7], data.iloc[count, 8], data.iloc[count, 9]], [data.iloc[count, 10], data.iloc[count, 11], data.iloc[count, 12], data.iloc[count, 13]], maxNumIterations=1000)

            p.setJointMotorControlArray(botId,
                                        jointIndices=revoluteJoints[0:10],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=gripperPosition1[0:10],
                                        forces=[50]*10)
            p.setJointMotorControlArray(botId,
                                        jointIndices=revoluteJoints[10:],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=gripperPosition2[10:],
                                        forces=[50]*9)

            p.setJointMotorControlArray(botId,
                                        jointIndices=[29, 31, 52, 54],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[-0.55]*4,
                                        forces = [10000]*4)
            for j in range(20):
                p.stepSimulation()

        elif count < 58 and count > 40:
            p.setJointMotorControlArray(botId,
                                        jointIndices=[29, 31, 52, 54],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0.8]*4,
                                        forces = [10000]*4)

        for j in range(10):
            p.stepSimulation()

        count += 1

    height = 0.65

    if p.getLinkState(boxId, 0)[0][2] >= height or p.getLinkState(boxId, 1)[0][2] >= height or p.getLinkState(boxId, 1)[0][2] >= height or p.getLinkState(boxId, 2)[0][2] >= height:
        label = 1

    else:

        label = 0

    p.disconnect()

    return label


#The following script visualizes the output predictions of our model on the table lift and place task.
def main():

    data = pd.read_csv('projection a success v5 ODE.csv')

    label = 0

    for i in range(0, data.shape[0], 70):

        sub_dataset = data.iloc[i:i+70, :]
        label += visualize(sub_dataset)

    accuracy = label / (data.shape[0]/70)

    print("Accuracy: ", accuracy)

    # getPredictedGripperPositions()

if __name__=="__main__":
    main()
