import pybullet as p
import numpy as np
import pandas as pd


#Thw following function returns all of the revolute joints associated with the URDF object that is passed to it
def getRevoluteJoints(botId):

    revoluteJoints = []

    for i in range(p.getNumJoints(botId)):

        if p.getJointInfo(botId, i)[2] == 0:
            revoluteJoints.append(i)

    return revoluteJoints


#The following function returns the state of all revolute joints associated with the URDF object passed to it
def getJointStates(botId):

    jointStates = []
    revoluteJoints = getRevoluteJoints(botId)

    for joint in revoluteJoints:
        jointStates.append(p.getJointState(botId, joint)[0])

    return jointStates



#Primitive to grasp objects from the front
def front_grasp(botId, rightCoords, leftCoords, left_orientation, right_orientation, steps=1):

    revoluteJoints = getRevoluteJoints(botId)

    x_prev_left = p.getLinkState(botId, 28)[0][0]

    iter_left_x = (rightCoords[0]-x_prev_left)/steps

    x_prev_right = p.getLinkState(botId, 51)[0][0]

    iter_right_x = (rightCoords[0]-x_prev_right)/steps

    y_prev_left = p.getLinkState(botId, 28)[0][1]

    iter_left_y = (leftCoords[1]-y_prev_left)/steps

    y_prev_right = p.getLinkState(botId, 51)[0][1]

    iter_right_y = (rightCoords[1]-y_prev_right)/steps

    i = 0

    for i in range(steps):

        x_prev_left += iter_left_x
        x_prev_right += iter_right_x
        y_prev_left += iter_left_y
        y_prev_right += iter_right_y
        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [x_prev_left, y_prev_left, leftCoords[2]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [x_prev_right, y_prev_right, rightCoords[2]], maxNumIterations=1000)
        jointStates = getJointStates(botId)

        data["Primitive"].append("Front_Grasp")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for i in range(10):
            p.stepSimulation()

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        data["label"].append(None)

    return gripperPosition1, gripperPosition2




#Primitive method to move a grasped object along the x and y axis to a new location
def move_sideways(botId, rightCoords, leftCoords, left_orientation, right_orientation, steps=1):

    revoluteJoints = getRevoluteJoints(botId)

    y_prev_left = p.getLinkState(botId, 28)[0][1]

    iter_left = (leftCoords[1]-y_prev_left)/steps

    y_prev_right = p.getLinkState(botId, 51)[0][1]

    iter_right = (rightCoords[1]-y_prev_right)/steps

    i = 0

    for i in range(steps):

        y_prev_left += iter_left
        y_prev_right += iter_right
        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [leftCoords[0], y_prev_left, leftCoords[2]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [rightCoords[0], y_prev_right, rightCoords[2]], maxNumIterations=1000)
        jointStates = getJointStates(botId)
        data["Primitive"].append("MoveSideways")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])

        for i in range(10):
            p.stepSimulation()

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        data["label"].append(None)

    return gripperPosition1, gripperPosition2






#Primitive to lift objects along the z axis
def lift(botId, z, boxId, steps=1):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(steps):

        rightGripperPosition[2] += 0.04
        leftGripperPosition[2] += 0.04

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        data["Primitive"].append("Lift")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])

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

        p.setJointMotorControlArray(botId,
                                    jointIndices=[19, 42],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.35]*2,
                                    forces=[10]*2)

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        data["label"].append(None)


    return gripperPosition1, gripperPosition2






#Primitive to extend a grasped object out along the x axis
def extend(botId, x, boxId, steps=1):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    # iter = (x-rightGripperOrientation[0])/steps
    iter = 0.08

    count = 0

    for i in range(steps):

        rightGripperPosition[0] += iter
        leftGripperPosition[0] += iter

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        data["Primitive"].append("extend")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])

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
                                    forces = [1000]*4)
        for i in range(10):
            p.stepSimulation()

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        data["label"].append(None)

    return gripperPosition1, gripperPosition2





#Primitive to retract a robot's arms once it has placed an object down
def retract(botId, x, boxId, steps=1):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    iter = 0.02

    count = 0

    for i in range(steps):

        rightGripperPosition[0] -= iter
        leftGripperPosition[0] -= iter
        rightGripperPosition[1] += 0.02
        leftGripperPosition[1] += 0.02

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        data["Primitive"].append("Retract")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])

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

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        if (count < steps - 1):
            data["label"].append(None)

        count += 1

    return gripperPosition1, gripperPosition2



#Primitive to place objects that have been lifted at a specified location
def place(botId, z, boxId, steps=1):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(steps):

        rightGripperPosition[2] -= 0.02
        leftGripperPosition[2] -= 0.02

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        data["Primitive"].append("Place")

        data["right_gripper_pole_x_1"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_1"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_1"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_1"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_1"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_1"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_11"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_12"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_13"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_11"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_12"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_13"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_14"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_1"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_1"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_1"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_1"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_1"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_1"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_1"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_1"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_1"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_1"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_1"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_1"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_1"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_1"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_1"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_1"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_1"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_1"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_1"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_1"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_1"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_1"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_1"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_1"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_1"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_1"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_1"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_1"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_1"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_1"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_1"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_1"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_1"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_1"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_1"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_1"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_1"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_1"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_1"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_1"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_1"].append(p.getLinkState(botId, 20)[1][3])

        data["right_action1"].append(gripperPosition1[0])
        data["right_action2"].append(gripperPosition1[1])
        data["right_action3"].append(gripperPosition1[2])
        data["right_action4"].append(gripperPosition1[3])
        data["right_action5"].append(gripperPosition1[4])
        data["right_action6"].append(gripperPosition1[5])
        data["right_action7"].append(gripperPosition1[6])
        data["right_action8"].append(gripperPosition1[7])
        data["right_action9"].append(gripperPosition1[8])
        data["right_action10"].append(gripperPosition1[9])
        data["right_action11"].append(gripperPosition1[10])
        data["right_action12"].append(gripperPosition1[11])
        data["right_action13"].append(gripperPosition1[12])
        data["right_action14"].append(gripperPosition1[13])
        data["right_action15"].append(gripperPosition1[14])
        data["right_action16"].append(gripperPosition1[15])
        data["right_action17"].append(gripperPosition1[16])
        data["right_action18"].append(gripperPosition1[17])
        data["right_action19"].append(gripperPosition1[18])

        data["left_action1"].append(gripperPosition2[0])
        data["left_action2"].append(gripperPosition2[1])
        data["left_action3"].append(gripperPosition2[2])
        data["left_action4"].append(gripperPosition2[3])
        data["left_action5"].append(gripperPosition2[4])
        data["left_action6"].append(gripperPosition2[5])
        data["left_action7"].append(gripperPosition2[6])
        data["left_action8"].append(gripperPosition2[7])
        data["left_action9"].append(gripperPosition2[8])
        data["left_action10"].append(gripperPosition2[9])
        data["left_action11"].append(gripperPosition2[10])
        data["left_action12"].append(gripperPosition2[11])
        data["left_action13"].append(gripperPosition2[12])
        data["left_action14"].append(gripperPosition2[13])
        data["left_action15"].append(gripperPosition2[14])
        data["left_action16"].append(gripperPosition2[15])
        data["left_action17"].append(gripperPosition2[16])
        data["left_action18"].append(gripperPosition2[17])
        data["left_action19"].append(gripperPosition2[18])

        data["Joint1_1"].append(jointStates[0])
        data["Joint1_2"].append(jointStates[1])
        data["Joint1_3"].append(jointStates[2])
        data["Joint1_4"].append(jointStates[3])
        data["Joint1_5"].append(jointStates[4])
        data["Joint1_6"].append(jointStates[5])
        data["Joint1_7"].append(jointStates[6])
        data["Joint1_8"].append(jointStates[7])
        data["Joint1_9"].append(jointStates[8])
        data["Joint1_10"].append(jointStates[9])
        data["Joint1_11"].append(jointStates[10])
        data["Joint1_12"].append(jointStates[11])
        data["Joint1_13"].append(jointStates[12])
        data["Joint1_14"].append(jointStates[13])
        data["Joint1_15"].append(jointStates[14])
        data["Joint1_16"].append(jointStates[15])
        data["Joint1_17"].append(jointStates[16])
        data["Joint1_18"].append(jointStates[17])
        data["Joint1_19"].append(jointStates[18])

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])


        if i > 5:

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

        else:

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

        jointStates = getJointStates(botId)

        data["right_gripper_pole_x_2"].append(p.getLinkState(botId, 28)[0][0])
        data["right_gripper_pole_y_2"].append(p.getLinkState(botId, 28)[0][1])
        data["right_gripper_pole_z_2"].append(p.getLinkState(botId, 28)[0][2])
        data["left_gripper_pole_x_2"].append(p.getLinkState(botId, 51)[0][0])
        data["left_gripper_pole_y_2"].append(p.getLinkState(botId, 51)[0][1])
        data["left_gripper_pole_z_2"].append(p.getLinkState(botId, 51)[0][2])

        data["right_gripper_pole_q_21"].append(p.getLinkState(botId, 28)[1][0])
        data["right_gripper_pole_q_22"].append(p.getLinkState(botId, 28)[1][1])
        data["right_gripper_pole_q_23"].append(p.getLinkState(botId, 28)[1][2])
        data["right_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])
        data["left_gripper_pole_q_21"].append(p.getLinkState(botId, 51)[1][0])
        data["left_gripper_pole_q_22"].append(p.getLinkState(botId, 51)[1][1])
        data["left_gripper_pole_q_23"].append(p.getLinkState(botId, 51)[1][2])
        data["left_gripper_pole_q_24"].append(p.getLinkState(botId, 28)[1][3])

        data["right_gripper_x_2"].append(p.getLinkState(botId, 29)[0][0])
        data["right_gripper_y_2"].append(p.getLinkState(botId, 29)[0][1])
        data["right_gripper_z_2"].append(p.getLinkState(botId, 29)[0][2])
        data["left_gripper_x_2"].append(p.getLinkState(botId, 52)[0][0])
        data["left_gripper_y_2"].append(p.getLinkState(botId, 52)[0][1])
        data["left_gripper_z_2"].append(p.getLinkState(botId, 52)[0][2])

        data["right_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["right_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["right_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["right_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["right_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["right_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["right_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["right_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["right_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["right_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["right_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["right_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["right_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["right_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["right_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["right_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["right_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["right_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["right_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["right_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["right_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["right_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["right_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["right_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["right_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["right_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["right_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["right_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["right_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["right_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["right_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["right_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["right_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["right_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["right_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["left_upper_shoulder_x_2"].append(p.getLinkState(botId, 13)[0][0])
        data["left_upper_shoulder_y_2"].append(p.getLinkState(botId, 13)[0][1])
        data["left_upper_shoulder_z_2"].append(p.getLinkState(botId, 13)[0][2])
        data["left_upper_shoulder_quat1_2"].append(p.getLinkState(botId, 13)[1][0])
        data["left_upper_shoulder_quat2_2"].append(p.getLinkState(botId, 13)[1][1])
        data["left_upper_shoulder_quat3_2"].append(p.getLinkState(botId, 13)[1][2])
        data["left_upper_shoulder_quat4_2"].append(p.getLinkState(botId, 13)[1][3])

        data["left_lower_shoulder_x_2"].append(p.getLinkState(botId, 14)[0][0])
        data["left_lower_shoulder_y_2"].append(p.getLinkState(botId, 14)[0][1])
        data["left_lower_shoulder_z_2"].append(p.getLinkState(botId, 14)[0][2])
        data["left_lower_shoulder_quat1_2"].append(p.getLinkState(botId, 14)[1][0])
        data["left_lower_shoulder_quat2_2"].append(p.getLinkState(botId, 14)[1][1])
        data["left_lower_shoulder_quat3_2"].append(p.getLinkState(botId, 14)[1][2])
        data["left_lower_shoulder_quat4_2"].append(p.getLinkState(botId, 14)[1][3])

        data["left_upper_forearm_x_2"].append(p.getLinkState(botId, 17)[0][0])
        data["left_upper_forearm_y_2"].append(p.getLinkState(botId, 17)[0][1])
        data["left_upper_forearm_z_2"].append(p.getLinkState(botId, 17)[0][2])
        data["left_upper_forearm_quat1_2"].append(p.getLinkState(botId, 17)[1][0])
        data["left_upper_forearm_quat2_2"].append(p.getLinkState(botId, 17)[1][1])
        data["left_upper_forearm_quat3_2"].append(p.getLinkState(botId, 17)[1][2])
        data["left_upper_forearm_quat4_2"].append(p.getLinkState(botId, 17)[1][3])

        data["left_lower_forearm_x_2"].append(p.getLinkState(botId, 19)[0][0])
        data["left_lower_forearm_y_2"].append(p.getLinkState(botId, 19)[0][1])
        data["left_lower_forearm_z_2"].append(p.getLinkState(botId, 19)[0][2])
        data["left_lower_forearm_quat1_2"].append(p.getLinkState(botId, 19)[1][0])
        data["left_lower_forearm_quat2_2"].append(p.getLinkState(botId, 19)[1][1])
        data["left_lower_forearm_quat3_2"].append(p.getLinkState(botId, 19)[1][2])
        data["left_lower_forearm_quat4_2"].append(p.getLinkState(botId, 19)[1][3])

        data["left_wrist_x_2"].append(p.getLinkState(botId, 20)[0][0])
        data["left_wrist_y_2"].append(p.getLinkState(botId, 20)[0][1])
        data["left_wrist_z_2"].append(p.getLinkState(botId, 20)[0][2])
        data["left_wrist_quat1_2"].append(p.getLinkState(botId, 20)[1][0])
        data["left_wrist_quat2_2"].append(p.getLinkState(botId, 20)[1][1])
        data["left_wrist_quat3_2"].append(p.getLinkState(botId, 20)[1][2])
        data["left_wrist_quat4_2"].append(p.getLinkState(botId, 20)[1][3])

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

        data["Joint2_1"].append(jointStates[0])
        data["Joint2_2"].append(jointStates[1])
        data["Joint2_3"].append(jointStates[2])
        data["Joint2_4"].append(jointStates[3])
        data["Joint2_5"].append(jointStates[4])
        data["Joint2_6"].append(jointStates[5])
        data["Joint2_7"].append(jointStates[6])
        data["Joint2_8"].append(jointStates[7])
        data["Joint2_9"].append(jointStates[8])
        data["Joint2_10"].append(jointStates[9])
        data["Joint2_11"].append(jointStates[10])
        data["Joint2_12"].append(jointStates[11])
        data["Joint2_13"].append(jointStates[12])
        data["Joint2_14"].append(jointStates[13])
        data["Joint2_15"].append(jointStates[14])
        data["Joint2_16"].append(jointStates[15])
        data["Joint2_17"].append(jointStates[16])
        data["Joint2_18"].append(jointStates[17])
        data["Joint2_19"].append(jointStates[18])

        data["label"].append(None)


    return gripperPosition1, gripperPosition2



#The folowing script generates simulation data for the bimanual manipulation task of grasping a table, lifting it into the air, and placing it on top of a block.
#This was the first simulation that we designed for our project, and so the emphasis on making sure that we got quality demonstration data, which we yielded
#through an iterative process of generating data, using it to train our models, evaluating them, using those evaluations to make improvements to the simulations,
#and then repeating this process. Because this was the first simulation we designed, we had not yet developed a framework and all of the code is condensed into this
#one script. For our second, more complex simulation, the peg-in-hole task, we designed it in a much more streamlined manner. All of the code for the primitives used
#for that simulation is condensed into a file called primitives.py. This file can easily be imported and used to design other tasks, with the goal of allowing other
#researchers to easily design their own simulations to obtain demonstration data for bimanual manipulation tasks.
p.connect(p.GUI)

data = {"Primitive": [], "right_gripper_pole_x_1": [], "right_gripper_pole_y_1": [], "right_gripper_pole_z_1": [],
        "left_gripper_pole_x_1": [], "left_gripper_pole_y_1": [], "left_gripper_pole_z_1": [],
        "right_gripper_pole_q_11": [], "right_gripper_pole_q_12": [], "right_gripper_pole_q_13": [], "right_gripper_pole_q_14": [],
        "left_gripper_pole_q_11": [], "left_gripper_pole_q_12": [], "left_gripper_pole_q_13": [], "left_gripper_pole_q_14": [],
        "right_gripper_x_1": [], "right_gripper_y_1": [], "right_gripper_z_1": [],
        "left_gripper_x_1": [], "left_gripper_y_1": [], "left_gripper_z_1": [],
        "right_upper_shoulder_x_1": [], "right_lower_shoulder_x_1": [], "right_upper_forearm_x_1": [], "right_lower_forearm_x_1": [], "right_wrist_x_1": [],
        "left_upper_shoulder_x_1": [], "left_lower_shoulder_x_1": [], "left_upper_forearm_x_1": [], "left_lower_forearm_x_1": [], "left_wrist_x_1": [],
        "right_upper_shoulder_y_1": [], "right_lower_shoulder_y_1": [], "right_upper_forearm_y_1": [], "right_lower_forearm_y_1": [], "right_wrist_y_1": [],
        "left_upper_shoulder_y_1": [], "left_lower_shoulder_y_1": [], "left_upper_forearm_y_1": [], "left_lower_forearm_y_1": [], "left_wrist_y_1": [],
        "right_upper_shoulder_z_1": [], "right_lower_shoulder_z_1": [], "right_upper_forearm_z_1": [], "right_lower_forearm_z_1": [], "right_wrist_z_1": [],
        "left_upper_shoulder_z_1": [], "left_lower_shoulder_z_1": [], "left_upper_forearm_z_1": [], "left_lower_forearm_z_1": [], "left_wrist_z_1": [],
        "right_upper_shoulder_quat1_1": [], "right_lower_shoulder_quat1_1": [], "right_upper_forearm_quat1_1": [], "right_lower_forearm_quat1_1": [], "right_wrist_quat1_1": [],
        "left_upper_shoulder_quat1_1": [], "left_lower_shoulder_quat1_1": [], "left_upper_forearm_quat1_1": [], "left_lower_forearm_quat1_1": [], "left_wrist_quat1_1": [],
        "right_upper_shoulder_quat2_1": [], "right_lower_shoulder_quat2_1": [], "right_upper_forearm_quat2_1": [], "right_lower_forearm_quat2_1": [], "right_wrist_quat2_1": [],
        "left_upper_shoulder_quat2_1": [], "left_lower_shoulder_quat2_1": [], "left_upper_forearm_quat2_1": [], "left_lower_forearm_quat2_1": [], "left_wrist_quat2_1": [],
        "right_upper_shoulder_quat3_1": [], "right_lower_shoulder_quat3_1": [], "right_upper_forearm_quat3_1": [], "right_lower_forearm_quat3_1": [], "right_wrist_quat3_1": [],
        "left_upper_shoulder_quat3_1": [], "left_lower_shoulder_quat3_1": [], "left_upper_forearm_quat3_1": [], "left_lower_forearm_quat3_1": [], "left_wrist_quat3_1": [],
        "right_upper_shoulder_quat4_1": [], "right_lower_shoulder_quat4_1": [], "right_upper_forearm_quat4_1": [], "right_lower_forearm_quat4_1": [], "right_wrist_quat4_1": [],
        "left_upper_shoulder_quat4_1": [], "left_lower_shoulder_quat4_1": [], "left_upper_forearm_quat4_1": [], "left_lower_forearm_quat4_1": [], "left_wrist_quat4_1": [],
        "x_1": [], "y_1": [], "z_1": [], "quat1_1": [], "quat2_1": [], "quat3_1": [],
        "quat4_1": [], "left_back_corner_x_1": [], "left_back_corner_y_1": [], "left_back_corner_z_1": [],
        "left_front_corner_x_1": [], "left_front_corner_y_1": [], "left_front_corner_z_1": [],
        "right_back_corner_x_1": [], "right_back_corner_y_1": [], "right_back_corner_z_1": [],
        "right_front_corner_x_1": [], "right_front_corner_y_1": [], "right_front_corner_z_1": [],
        "right_action1": [], "right_action2": [], "right_action3": [], "right_action4": [], "right_action5": [], "right_action6": [],
        "right_action7": [], "right_action8": [], "right_action9": [], "right_action10": [], "right_action11": [], "right_action12": [],
        "right_action13": [], "right_action14": [], "right_action15": [], "right_action16": [], "right_action17": [], "right_action18": [],
        "right_action19": [], "left_action1": [], "left_action2": [], "left_action3": [], "left_action4": [], "left_action5": [], "left_action6": [],
        "left_action7": [], "left_action8": [], "left_action9": [], "left_action10": [], "left_action11": [], "left_action12": [],
        "left_action13": [], "left_action14": [], "left_action15": [], "left_action16": [], "left_action17": [], "left_action18": [], "left_action19": [],
        "right_gripper_pole_x_2": [], "right_gripper_pole_y_2": [], "right_gripper_pole_z_2": [],
        "left_gripper_pole_x_2": [], "left_gripper_pole_y_2": [], "left_gripper_pole_z_2": [],
        "right_gripper_pole_q_21": [], "right_gripper_pole_q_22": [], "right_gripper_pole_q_23": [], "right_gripper_pole_q_24": [],
        "left_gripper_pole_q_21": [], "left_gripper_pole_q_22": [], "left_gripper_pole_q_23": [], "left_gripper_pole_q_24": [],
        "right_gripper_x_2": [], "right_gripper_y_2": [], "right_gripper_z_2": [], "left_gripper_x_2": [], "left_gripper_y_2": [], "left_gripper_z_2": [],
        "right_upper_shoulder_x_2": [], "right_lower_shoulder_x_2": [], "right_upper_forearm_x_2": [], "right_lower_forearm_x_2": [], "right_wrist_x_2": [],
        "left_upper_shoulder_x_2": [], "left_lower_shoulder_x_2": [], "left_upper_forearm_x_2": [], "left_lower_forearm_x_2": [], "left_wrist_x_2": [],
        "right_upper_shoulder_y_2": [], "right_lower_shoulder_y_2": [], "right_upper_forearm_y_2": [], "right_lower_forearm_y_2": [], "right_wrist_y_2": [],
        "left_upper_shoulder_y_2": [], "left_lower_shoulder_y_2": [], "left_upper_forearm_y_2": [], "left_lower_forearm_y_2": [], "left_wrist_y_2": [],
        "right_upper_shoulder_z_2": [], "right_lower_shoulder_z_2": [], "right_upper_forearm_z_2": [], "right_lower_forearm_z_2": [], "right_wrist_z_2": [],
        "left_upper_shoulder_z_2": [], "left_lower_shoulder_z_2": [], "left_upper_forearm_z_2": [], "left_lower_forearm_z_2": [], "left_wrist_z_2": [],
        "right_upper_shoulder_quat1_2": [], "right_lower_shoulder_quat1_2": [], "right_upper_forearm_quat1_2": [], "right_lower_forearm_quat1_2": [], "right_wrist_quat1_2": [],
        "left_upper_shoulder_quat1_2": [], "left_lower_shoulder_quat1_2": [], "left_upper_forearm_quat1_2": [], "left_lower_forearm_quat1_2": [], "left_wrist_quat1_2": [],
        "right_upper_shoulder_quat2_2": [], "right_lower_shoulder_quat2_2": [], "right_upper_forearm_quat2_2": [], "right_lower_forearm_quat2_2": [], "right_wrist_quat2_2": [],
        "left_upper_shoulder_quat2_2": [], "left_lower_shoulder_quat2_2": [], "left_upper_forearm_quat2_2": [], "left_lower_forearm_quat2_2": [], "left_wrist_quat2_2": [],
        "right_upper_shoulder_quat3_2": [], "right_lower_shoulder_quat3_2": [], "right_upper_forearm_quat3_2": [], "right_lower_forearm_quat3_2": [], "right_wrist_quat3_2": [],
        "left_upper_shoulder_quat3_2": [], "left_lower_shoulder_quat3_2": [], "left_upper_forearm_quat3_2": [], "left_lower_forearm_quat3_2": [], "left_wrist_quat3_2": [],
        "right_upper_shoulder_quat4_2": [], "right_lower_shoulder_quat4_2": [], "right_upper_forearm_quat4_2": [], "right_lower_forearm_quat4_2": [], "right_wrist_quat4_2": [],
        "left_upper_shoulder_quat4_2": [], "left_lower_shoulder_quat4_2": [], "left_upper_forearm_quat4_2": [], "left_lower_forearm_quat4_2": [], "left_wrist_quat4_2": [],
        "x_2": [], "y_2": [], "z_2": [], "quat1_2": [], "quat2_2": [], "quat3_2": [],
        "quat4_2": [], "left_back_corner_x_2": [], "left_back_corner_y_2": [], "left_back_corner_z_2": [],
        "left_front_corner_x_2": [], "left_front_corner_y_2": [], "left_front_corner_z_2": [],
        "right_back_corner_x_2": [], "right_back_corner_y_2": [], "right_back_corner_z_2": [],
        "right_front_corner_x_2": [], "right_front_corner_y_2": [], "right_front_corner_z_2": [], "gripper_open": [], "label": [],
        "x_displacement": [], "y_displacement": [], "Joint1_1": [], "Joint1_2": [], "Joint1_3": [], "Joint1_4": [], "Joint1_5": [],
        "Joint1_6": [], "Joint1_7": [], "Joint1_8": [], "Joint1_9": [], "Joint1_10": [],
        "Joint1_11": [], "Joint1_12": [], "Joint1_13": [], "Joint1_14": [], "Joint1_15": [],
        "Joint1_16": [], "Joint1_17": [], "Joint1_18": [], "Joint1_19": [], "Joint2_1": [],
        "Joint2_2": [], "Joint2_3": [], "Joint2_4": [], "Joint2_5": [], "Joint2_6": [],
        "Joint2_7": [], "Joint2_8": [], "Joint2_9": [], "Joint2_10": [], "Joint2_11": [],
        "Joint2_12": [], "Joint2_13": [], "Joint2_14": [], "Joint2_15": [], "Joint2_16": [],
        "Joint2_17": [], "Joint2_18": [], "Joint2_19": []}






#The following function gets the names for all of the links in Baxter's XML
# file as well as their associated IDs
def getLinkNames(model_id):
    _link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}

    for _id in range(p.getNumJoints(model_id)):
    	_name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
    	_link_name_to_index[_name] = _id

    return _link_name_to_index



def getGripperLocations(move_sideways=False):

    x_left = 0.74
    x_right = 0.74
    y_left = -0.35
    y_right = 0.35
    z = np.random.uniform(0.2, 0.3)

    x_displacement = np.random.uniform(0, 0.3)
    y_displacement = np.random.uniform(0, 0.6)-0.3

    if move_sideways:
        x_left += x_displacement
        y_left += y_displacement
        x_right += x_displacement
        y_right += y_displacement

    link_right = 1
    link_left = 1

    return x_displacement, y_displacement, link_right, link_left, [x_right, y_right, z], [x_left, y_left, z]


p.resetSimulation()
p.setGravity(0,0,-9.81)
p.setTimeStep(0.01)
planeId = p.loadURDF("plane.xml")
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
baxterStartOrientation = p.getQuaternionFromEuler([0,0,0])

botId = p.loadURDF("baxter.xml",
                [0, 0, 0],
                baxterStartOrientation, useFixedBase=1)

for iter in range(5000):

    print(iter)

    x_displacement, y_displacement, rightLink, leftLink, rightCoords, leftCoords = getGripperLocations(move_sideways=True)

    p.setJointMotorControlArray(botId,
                                jointIndices=[13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 39, 40, 42, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75, -0.9, 0, 1.8, 0, -0.9, 0, -0.75, -0.9, 0, 1.8, 0, -0.9, 0])

    revoluteJoints = getRevoluteJoints(botId)

    z = np.random.uniform(0.2, 0.6)

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
                                targetPositions=[0, 0])

    orientation = p.getQuaternionFromEuler([0, 0, 0])

    for i in range(10):
        p.stepSimulation()

    if leftLink >= 1/3:
        p.setJointMotorControlArray(botId,
                                    jointIndices=[20],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[1.8])
    if rightLink >= 1/3:
        p.setJointMotorControlArray(botId,
                                    jointIndices=[43],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[1.8])

    for i in range(100):
        p.stepSimulation()
    left_orientation = p.getLinkState(botId, 28)[1]
    right_orientation = p.getLinkState(botId, 51)[1]
    gripperPosition1, gripperPosition2 = front_grasp(botId, rightCoords, leftCoords, left_orientation, right_orientation, steps=10)

    for i in range(10):
        data["gripper_open"].append(1)

    for i in range(100):
        p.stepSimulation()

    boxId = p.loadURDF("table.xml",
                      [0.9 + x_displacement, y_displacement, 0],
                      cubeStartOrientation)

    for i in range(58):
        data["x_displacement"].append(x_displacement)
        data["y_displacement"].append(y_displacement)

    for i in range(100):
        p.stepSimulation()

    for i in range(10):

        data["x_1"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_1"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_1"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_1"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_1"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_1"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_1"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_1"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_1"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_1"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_1"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_1"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_1"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_1"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_1"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_1"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_1"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_1"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_1"].append(p.getLinkState(boxId, 3)[0][2])

    for i in range(10):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.55]*4,
                                forces = [10000]*4)

    for i in range(10):

        data["x_2"].append(p.getBasePositionAndOrientation(boxId)[0][0])
        data["y_2"].append(p.getBasePositionAndOrientation(boxId)[0][1])
        data["z_2"].append(p.getBasePositionAndOrientation(boxId)[0][2])
        data["quat1_2"].append(p.getBasePositionAndOrientation(boxId)[1][0])
        data["quat2_2"].append(p.getBasePositionAndOrientation(boxId)[1][1])
        data["quat3_2"].append(p.getBasePositionAndOrientation(boxId)[1][2])
        data["quat4_2"].append(p.getBasePositionAndOrientation(boxId)[1][3])
        data["right_back_corner_x_2"].append(p.getLinkState(boxId, 0)[0][0])
        data["right_back_corner_y_2"].append(p.getLinkState(boxId, 0)[0][1])
        data["right_back_corner_z_2"].append(p.getLinkState(boxId, 0)[0][2])
        data["left_front_corner_x_2"].append(p.getLinkState(boxId, 1)[0][0])
        data["left_front_corner_y_2"].append(p.getLinkState(boxId, 1)[0][1])
        data["left_front_corner_z_2"].append(p.getLinkState(boxId, 1)[0][2])
        data["right_front_corner_x_2"].append(p.getLinkState(boxId, 2)[0][0])
        data["right_front_corner_y_2"].append(p.getLinkState(boxId, 2)[0][1])
        data["right_front_corner_z_2"].append(p.getLinkState(boxId, 2)[0][2])
        data["left_back_corner_x_2"].append(p.getLinkState(boxId, 3)[0][0])
        data["left_back_corner_y_2"].append(p.getLinkState(boxId, 3)[0][1])
        data["left_back_corner_z_2"].append(p.getLinkState(boxId, 3)[0][2])

    _, _, rightLink, leftLink, rightCoords, leftCoords = getGripperLocations()

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = move_sideways(botId, rightCoords, leftCoords, left_orientation, right_orientation, steps=12)

    for i in range(100):
        p.stepSimulation()

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

    gripperPosition1, gripperPosition2 = lift(botId, 0.5, boxId, steps=12)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = extend(botId, 0.5, boxId, steps=12)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = place(botId, 0.5, boxId, steps=12)

    for i in range(100):
        p.stepSimulation()

    for i in range(36):
        data["gripper_open"].append(0)

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.8]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = retract(botId, 0.5, boxId, steps=12)

    for i in range(12):
        data["gripper_open"].append(1)

    for i in range(100):
        p.stepSimulation()

    height = 0.65

    if p.getLinkState(boxId, 0)[0][2] >= height and p.getLinkState(boxId, 1)[0][2] >= height and p.getLinkState(boxId, 1)[0][2] >= height and p.getLinkState(boxId, 2)[0][2] >= height:
        label = 1

    else:

        label = 0

    data["label"].append(label)

    p.removeBody(boxId)
    p.removeBody(blockId)

df = pd.DataFrame(data)
df.to_csv("lift_primitive_data.csv")

p.disconnect()
