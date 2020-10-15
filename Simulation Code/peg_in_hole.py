import pybullet as p
import numpy as np
import pandas as pd
import primitives as pm

#The folowing script generates simulation data for the bimanual manipulation task of alligning two seperate halves of a table, connecting them, and lifting the
#fully connected object into the air.
p.connect(p.GUI)

data = {"Primitive": [], "right_gripper_pole_x_1": [], "right_gripper_pole_y_1": [], "right_gripper_pole_z_1": [],
        "left_gripper_pole_x_1": [], "left_gripper_pole_y_1": [], "left_gripper_pole_z_1": [],
        "right_gripper_pole_q_11": [], "right_gripper_pole_q_12": [], "right_gripper_pole_q_13": [], "right_gripper_pole_q_14": [],
        "left_gripper_pole_q_11": [], "left_gripper_pole_q_12": [], "left_gripper_pole_q_13": [], "left_gripper_pole_q_14": [],
        "table1_x_1": [], "table1_y_1": [], "table1_z_1": [], "table1_quat1_1": [], "table1_quat2_1": [], "table1_quat3_1": [], "table1_quat4_1": [],
        "table2_x_1": [], "table2_y_1": [], "table2_z_1": [], "table2_quat1_1": [], "table2_quat2_1": [], "table2_quat3_1": [], "table2_quat4_1": [],
        "right_gripper_pole_x_2": [], "right_gripper_pole_y_2": [], "right_gripper_pole_z_2": [],
        "left_gripper_pole_x_2": [], "left_gripper_pole_y_2": [], "left_gripper_pole_z_2": [],
        "right_gripper_pole_q_21": [], "right_gripper_pole_q_22": [], "right_gripper_pole_q_23": [], "right_gripper_pole_q_24": [],
        "left_gripper_pole_q_21": [], "left_gripper_pole_q_22": [], "left_gripper_pole_q_23": [], "left_gripper_pole_q_24": [],
        "table1_x_2": [], "table1_y_2": [], "table1_z_2": [], "table1_quat1_2": [], "table1_quat2_2": [], "table1_quat3_2": [], "table1_quat4_2": [],
        "table2_x_2": [], "table2_y_2": [], "table2_z_2": [], "table2_quat1_2": [], "table2_quat2_2": [], "table2_quat3_2": [], "table2_quat4_2": [],
        "x_displacement1": [], "y_displacement1": [], "x_displacement2": [], "y_displacement2": [], "grippers_open": [], "label": []}

# p.resetDebugVisualizerCamera( cameraDistance=3, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,0])
# p.resetDebugVisualizerCamera( cameraDistance=3, cameraYaw=90, cameraPitch=-52, cameraTargetPosition=[0,0,0])
# p.resetDebugVisualizerCamera( cameraDistance=3, cameraYaw=150, cameraPitch=-52, cameraTargetPosition=[0,0,0])

p.resetSimulation()
p.setGravity(0,0,-9.81)
p.setTimeStep(0.01)
planeId = p.loadURDF("plane.xml")
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
baxterStartOrientation = p.getQuaternionFromEuler([0,0,0])

botId = p.loadURDF("baxter.xml",
                [0, 0, 0],
                baxterStartOrientation, useFixedBase=1)

count = 0

for iter in range(5000):

    count += 1

    print(iter)

    p.setJointMotorControlArray(botId,
                                jointIndices=[13, 14, 15, 16, 17, 19, 20, 36, 37, 38, 39, 40, 42, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75, -0.9, 0, 1.8, 0, -0.9, 0, -0.75, -0.9, 0, 1.8, 0, -0.9, 0])

    revoluteJoints = pm.getRevoluteJoints(botId)

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

    for i in range(100):
        p.stepSimulation()

    for i in range(10):
        data["grippers_open"].append(1)

    orientation = p.getQuaternionFromEuler([0, 0, 0])

    for i in range(100):
        p.stepSimulation()

    x_displacement1 = np.random.uniform(0, 0.1)
    y_displacement1 = np.random.uniform(0, 0.1)
    x_displacement2 = np.random.uniform(0, 0.1)
    y_displacement2 = np.random.uniform(0, 0.1)*-1

    gripperPosition1, gripperPosition2 = pm.front_grasp(botId, [0.88+x_displacement1, 0.365+y_displacement1, 0.15000000000000002], [0.87+x_displacement1, 0.145+y_displacement1, 0.15000000000000002], data, steps=10, tableId1=None, tableId2=None, both=None, primitive_name="Front_Grasp", front_grasp=True)

    for i in range(100):
        p.stepSimulation()

    tableId1 = p.loadURDF("table_dock.xml",
                      [1 + x_displacement1, 0.1 + y_displacement1, 0],
                      cubeStartOrientation,
                      globalScaling=0.75)

    for i in range(10):

        data["table1_x_1"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
        data["table1_y_1"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
        data["table1_z_1"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
        data["table1_quat1_1"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
        data["table1_quat2_1"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
        data["table1_quat3_1"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
        data["table1_quat4_1"].append(p.getBasePositionAndOrientation(tableId1)[1][3])
        data["table1_x_2"].append(p.getBasePositionAndOrientation(tableId1)[0][0])
        data["table1_y_2"].append(p.getBasePositionAndOrientation(tableId1)[0][1])
        data["table1_z_2"].append(p.getBasePositionAndOrientation(tableId1)[0][2])
        data["table1_quat1_2"].append(p.getBasePositionAndOrientation(tableId1)[1][0])
        data["table1_quat2_2"].append(p.getBasePositionAndOrientation(tableId1)[1][1])
        data["table1_quat3_2"].append(p.getBasePositionAndOrientation(tableId1)[1][2])
        data["table1_quat4_2"].append(p.getBasePositionAndOrientation(tableId1)[1][3])

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0]*4,
                                forces = [10]*4)

    for i in range(10):
        data["grippers_open"].append(0)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2, table1pos, table1orientation, table1vel, table1ang = pm.move_sideways(botId, [0.88, 0.3625, 0.15000000000000002], [0.88, 0.15625000000000003, 0.15000000000000002], data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Move_Sideways", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    for i in range(40):
        data["grippers_open"].append(1)

    gripperPosition1, gripperPosition2 = pm.lower(botId, 0.01, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Lower", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    inertpos, inertorientation = p.invertTransform(p.getDynamicsInfo(tableId1, -1)[3], p.getDynamicsInfo(tableId1, -1)[4])
    table1pos, table1orientation = p.multiplyTransforms(p.getBasePositionAndOrientation(tableId1)[0], p.getBasePositionAndOrientation(tableId1)[1], inertpos, inertorientation)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.retract(botId, 0.02, 0.02, data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Retract", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    p.removeBody(tableId1)
    tableId1 = p.loadURDF("table_dock.xml",
                      table1pos,
                      table1orientation,
                      globalScaling=0.75)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2, _, _, _, _ = pm.move_sideways(botId, [0.58, -0.15625000000000003, 0.15000000000000002], [0.58, -0.3625, 0.15000000000000002], data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Move_Sideways_2", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.front_grasp(botId, [0.87+x_displacement2, -0.14+y_displacement2, 0.15000000000000002], [0.88+x_displacement2, -0.33+y_displacement2, 0.15000000000000002], data, steps=10, tableId1=tableId1, tableId2=None, both=None, primitive_name="Front_Grasp_2", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    tableId2 = p.loadURDF("table_peg.xml",
                      [1 + x_displacement2, -0.1 + y_displacement2, 0],
                      cubeStartOrientation,
                      globalScaling=0.75)

    for i in range(60):

        data["table2_x_1"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
        data["table2_y_1"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
        data["table2_z_1"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
        data["table2_quat1_1"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
        data["table2_quat2_1"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
        data["table2_quat3_1"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
        data["table2_quat4_1"].append(p.getBasePositionAndOrientation(tableId2)[1][3])
        data["table2_x_2"].append(p.getBasePositionAndOrientation(tableId2)[0][0])
        data["table2_y_2"].append(p.getBasePositionAndOrientation(tableId2)[0][1])
        data["table2_z_2"].append(p.getBasePositionAndOrientation(tableId2)[0][2])
        data["table2_quat1_2"].append(p.getBasePositionAndOrientation(tableId2)[1][0])
        data["table2_quat2_2"].append(p.getBasePositionAndOrientation(tableId2)[1][1])
        data["table2_quat3_2"].append(p.getBasePositionAndOrientation(tableId2)[1][2])
        data["table2_quat4_2"].append(p.getBasePositionAndOrientation(tableId2)[1][3])

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    for i in range(10):
        data["grippers_open"].append(0)

    gripperPosition1, gripperPosition2, table2pos, table2orientation, table2vel, table2ang = pm.move_sideways(botId, [0.88, -0.15625000000000003, 0.15000000000000002], [0.88, -0.3625, 0.15000000000000002], data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Move_Sideways_3", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    for i in range(40):
        data["grippers_open"].append(1)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.lower(botId, 0.01, data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Lower_2", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    inertpos, inertorientation = p.invertTransform(p.getDynamicsInfo(tableId2, -1)[3], p.getDynamicsInfo(tableId2, -1)[4])
    table2pos, table2orientation = p.multiplyTransforms(p.getBasePositionAndOrientation(tableId2)[0], p.getBasePositionAndOrientation(tableId2)[1], inertpos, inertorientation)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.retract(botId, 0.02, -0.015, data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Retract_2", front_grasp=False)

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
                                targetPositions=[0.75]*4,
                                forces = [10000]*4)

    p.setJointMotorControlArray(botId,
                                jointIndices=[20, 43],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[0, 0])

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.front_grasp(botId, [0.6, 0.6, 0.30000000000000004], [0.6, -0.6, 0.30000000000000004], data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Front_Grasp_3", front_grasp=False)

    for i in range(100):
        p.stepSimulation()


    right_coords = list(p.getLinkState(tableId1, 9)[0])
    left_coords = list(p.getLinkState(tableId2, 5)[0])
    right_coords[0] -= 0.035
    left_coords[0] -= 0.035
    gripperPosition1, gripperPosition2 = pm.front_grasp2(botId, right_coords, left_coords, data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Front_Grasp_4", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    for i in range(20):
        data["grippers_open"].append(0)

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.connect(botId, [0.95, 0.45, 0.30000000000000004], [0.95, -0.45, 0.30000000000000004], data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Connect", front_grasp=False)

    for i in range(100):
        p.stepSimulation()

    p.setJointMotorControlArray(botId,
                                jointIndices=[29, 31, 52, 54],
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=[-0.75]*4,
                                forces = [10000]*4)

    for i in range(100):
        p.stepSimulation()

    gripperPosition1, gripperPosition2 = pm.lift(botId, 0.5, data, steps=10, tableId1=tableId1, tableId2=tableId2, both=True, primitive_name="Lift", front_grasp=False)

    for k in range(100):

        p.setJointMotorControlArray(botId,
                                    jointIndices=[29, 31, 52, 54],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[-0.75]*4,
                                    forces = [10000]*4)

        p.stepSimulation()

    if p.getBasePositionAndOrientation(tableId1)[0][2] > 0.4 and p.getBasePositionAndOrientation(tableId2)[0][2] > 0.4:
        label = 1

    else:

        label = 0

    for i in range(130):
        data["x_displacement1"].append(x_displacement1)
        data["y_displacement1"].append(y_displacement1)
        data["x_displacement2"].append(x_displacement2)
        data["y_displacement2"].append(y_displacement2)
        data["label"].append(label)

    p.removeBody(tableId1)
    p.removeBody(tableId2)

    # if count % 1000 == 0:
    df = pd.DataFrame(data)
    df.to_csv("lift_primitive_data.csv")

p.disconnect()
