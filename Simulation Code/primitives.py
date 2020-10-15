import pybullet as p



#The following function returns a list of the link names for a URDF object as well as their indices
def getLinkNames(model_id):
    _link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}

    for _id in range(p.getNumJoints(model_id)):
    	_name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
    	_link_name_to_index[_name] = _id

    return _link_name_to_index




#The following function prints the names of all of the joints for a URDF object
def getJointNames(botId):

    for i in range(p.getNumJoints(botId)):
        print(p.getJointInfo(botId, i)[0:2])



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


#The following function appends information about the cartesian coordinates and quaternion values for the center of
#mass values for different objects in the simulation. These values are used as features to train models from the
#demonstration data.
def get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=None, tableId2=None, front_grasp=True, both=True):

    if sequence == 1:

        data["Primitive"].append(primitive_name)

    if front_grasp:

        if sequence == 1:

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

        else:

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



    else:

        if both:

            if sequence == 1:

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

                data["table1_x_1"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
                data["table1_y_1"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
                data["table1_z_1"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
                data["table1_quat1_1"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
                data["table1_quat2_1"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
                data["table1_quat3_1"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
                data["table1_quat4_1"].append(p.getBasePositionAndOrientation(tableId1)[1][3])
                data["table2_x_1"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
                data["table2_y_1"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
                data["table2_z_1"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
                data["table2_quat1_1"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
                data["table2_quat2_1"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
                data["table2_quat3_1"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
                data["table2_quat4_1"].append(p.getBasePositionAndOrientation(tableId2)[1][3])

            else:

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

                data["table1_x_2"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
                data["table1_y_2"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
                data["table1_z_2"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
                data["table1_quat1_2"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
                data["table1_quat2_2"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
                data["table1_quat3_2"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
                data["table1_quat4_2"].append(p.getBasePositionAndOrientation(tableId1)[1][3])
                data["table2_x_2"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
                data["table2_y_2"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
                data["table2_z_2"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
                data["table2_quat1_2"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
                data["table2_quat2_2"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
                data["table2_quat3_2"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
                data["table2_quat4_2"].append(p.getBasePositionAndOrientation(tableId2)[1][3])

        elif tableId1:

            if sequence == 1:

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

                data["table1_x_1"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
                data["table1_y_1"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
                data["table1_z_1"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
                data["table1_quat1_1"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
                data["table1_quat2_1"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
                data["table1_quat3_1"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
                data["table1_quat4_1"].append(p.getBasePositionAndOrientation(tableId1)[1][3])

            else:

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

                data["table1_x_2"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
                data["table1_y_2"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
                data["table1_z_2"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
                data["table1_quat1_2"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
                data["table1_quat2_2"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
                data["table1_quat3_2"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
                data["table1_quat4_2"].append(p.getBasePositionAndOrientation(tableId1)[1][3])

        elif tableId2:

            if sequence == 1:

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

                data["table2_x_1"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
                data["table2_y_1"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
                data["table2_z_1"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
                data["table2_quat1_1"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
                data["table2_quat2_1"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
                data["table2_quat3_1"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
                data["table2_quat4_1"].append(p.getBasePositionAndOrientation(tableId2)[1][3])

            else:

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

                data["table2_x_2"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
                data["table2_y_2"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
                data["table2_z_2"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
                data["table2_quat1_2"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
                data["table2_quat2_2"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
                data["table2_quat3_2"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
                data["table2_quat4_2"].append(p.getBasePositionAndOrientation(tableId2)[1][3])


#Primitive to grasp objects from the front
def front_grasp(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

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

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

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

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

        jointStates = getJointStates(botId)

    return gripperPosition1, gripperPosition2


#Primitive to grasp objects from the front
def front_grasp2(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

    revoluteJoints = getRevoluteJoints(botId)

    x_prev = p.getLinkState(botId, 28)[0][0]

    iter = (rightCoords[0]-x_prev)/steps

    i = 0

    for i in range(steps):

        x_prev += iter
        gripperPosition1 = p.calculateInverseKinematics(botId, 28, [x_prev, leftCoords[1], leftCoords[2]], maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, [x_prev, rightCoords[1], rightCoords[2]], maxNumIterations=1000)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

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

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    return gripperPosition1, gripperPosition2


#Primitive to connect two different objects once they've been alligned
def connect(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(steps):

        rightGripperPosition[1] += 0.01
        leftGripperPosition[1] -= 0.01

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)
        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

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

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    return gripperPosition1, gripperPosition2



#Primitive to lift objects along the z axis
def lift(botId, z, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

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

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10],
                                    forces=[1000]*10)
        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:],
                                    forces=[1000]*9)
        p.setJointMotorControlArray(botId,
                                    jointIndices=[29, 31, 52, 54],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.75]*4,
                                    forces = [10000]*4)

        for i in range(10):
            p.stepSimulation()

        p.setJointMotorControlArray(botId,
                                    jointIndices=[19, 42],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.45]*2,
                                    forces=[100]*2)

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    return gripperPosition1, gripperPosition2



def move_sideways(botId, rightCoords, leftCoords, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

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

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[0:10],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition1[0:10])

        p.setJointMotorControlArray(botId,
                                    jointIndices=revoluteJoints[10:],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=gripperPosition2[10:])
        p.setJointMotorControlArray(botId,
                                    jointIndices=[29, 31, 52, 54],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.75]*4,
                                    forces = [10000]*4)
        for i in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    if tableId1 and not tableId2:
        return gripperPosition1, gripperPosition2, p.getBasePositionAndOrientation(tableId1)[0], p.getBasePositionAndOrientation(tableId1)[1], p.getBaseVelocity(tableId1)[0], p.getBaseVelocity(tableId1)[1]

    if tableId2 and tableId1:
        return gripperPosition1, gripperPosition2, p.getBasePositionAndOrientation(tableId2)[0], p.getBasePositionAndOrientation(tableId2)[1], p.getBaseVelocity(tableId2)[0], p.getBaseVelocity(tableId2)[1]

    return gripperPosition1, gripperPosition2


#Primitive to retract a robot's arms once it has placed an object down
def retract(botId, iter1, iter2, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(steps):

        rightGripperPosition[0] -= iter1
        leftGripperPosition[0] -= iter1
        rightGripperPosition[1] += iter2
        leftGripperPosition[1] += iter2
        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

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

        for i in range(10):
            p.stepSimulation()

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    return gripperPosition1, gripperPosition2




def lower(botId, iter1, data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name=None, front_grasp=False):

    rightGripperPosition, rightGripperOrientation = p.getLinkState(botId, 28)[0:2]
    rightGripperPosition = list(rightGripperPosition)
    leftGripperPosition, leftGripperOrientation = p.getLinkState(botId, 51)[0:2]
    leftGripperPosition = list(leftGripperPosition)

    revoluteJoints = getRevoluteJoints(botId)

    count = 0

    for i in range(steps):

        rightGripperPosition[2] -= iter1
        leftGripperPosition[2] -= iter1

        gripperPosition1 = p.calculateInverseKinematics(botId, 28, rightGripperPosition, rightGripperOrientation, maxNumIterations=1000)
        gripperPosition2 = p.calculateInverseKinematics(botId, 51, leftGripperPosition, leftGripperOrientation, maxNumIterations=1000)

        jointStates = getJointStates(botId)

        get_primitive_data(botId, data, primitive_name, sequence=1, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

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

        get_primitive_data(botId, data, primitive_name, sequence=2, tableId1=tableId1, tableId2=tableId2, front_grasp=front_grasp, both=both)

    return gripperPosition1, gripperPosition2
