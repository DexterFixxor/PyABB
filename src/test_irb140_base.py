import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from src.pybullet_gym.interfaces.robots.irb140interface import IRB140Interface
from src.pybullet_gym.interfaces.gripper.robotiq_2f import Robotiq2F
from src.pybullet_gym.interfaces.baseJointInterface import BaseJointInterface
from src.pybullet_gym.interfaces.groups.irb140_robotiq_interface import IRB1402FGripperInterface
import time



          
if __name__ == "__main__":
    
    import time
    
    client = bc.BulletClient(connection_mode=pb.GUI)
    
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    client.setGravity(0, 0, -9.81)
    
    
    cube_id = client.loadURDF('cube_small.urdf', useMaximalCoordinates=True, )
    client.resetBasePositionAndOrientation(cube_id, [0.712, 0, 0.026], [0, 0, 0, 1])

    client.changeDynamics(
        bodyUniqueId = cube_id,
        linkIndex = -1, 
        mass = 0.2,
        lateralFriction = 1,
        spinningFriction = 0.1,
        rollingFriction = 0.1,
        restitution = 0,
        linearDamping = 0.4,
        angularDamping = 0.4,
        contactStiffness = 1000,
        contactDamping = 1,
        frictionAnchor = False,        
    )
    
    robot = IRB1402FGripperInterface(client)
    
    
    q_init = robot._joint_positions
    pos, rot = robot._robotInterface.calculateDrectKinematics(q_init)
    pos[2] = 0.0
    
    # rot = np.array([0, 1, 0, 0])

    client.addUserDebugPoints(
        pointPositions=[[0.612, 0, 0]],
        pointColorsRGB = [[1.0, 1.0, 0.0]],
        pointSize=10
    )
    start = time.time()
    
    gripper_action = 0
    x_pos = pos[0]
    y_pos = pos[1]
    z_pos = pos[2]
    
    
    gripper_action_id = client.addUserDebugParameter("gripper", 0, 1, 0)
    x_id = client.addUserDebugParameter("x", -0.7, 0.7, 0.515)
    y_id = client.addUserDebugParameter("y", -0.7, 0.7, 0.0)
    z_id = client.addUserDebugParameter("z", 0, 0.5, 0.5)
    
    dt = 0
    
    target_pos = np.array([0.712, 0, 0.02])
    target_q = np.array([0, 1, 0, 0])
    # robot.attach_body(cube_id)
    start = time.time()
    
    open_len = 0.049
    angle = (0.715 - np.sin((open_len - 0.01)/0.1143)) / 0.14
    while True:

        robot._update_states()
        actions = np.random.randn(8)
        # actions[-1] = 1
       
        # robot.step(actions)
        
        if time.time() - start < 3:
            robot._gripper_interface.setGripperAction(0)
            robot._set_ee_pose(target_pos, target_q)
        elif time.time() - start > 3 and time.time() - start < 5:
            robot._gripper_interface.setGripperAction(0.65)
        elif time.time() - start < 7:
            target_pos[-1] = 0.3
            robot._set_ee_pose(target_pos, target_q)
            robot._gripper_interface.setGripperAction(0.65)
        else:
            target_pos[-1] = 0.1
            robot._gripper_interface.setGripperAction(0.65)
            robot._set_ee_pose(target_pos, target_q)
            
        client.stepSimulation()
        time.sleep(1.0/240.0)
    print(f"F: {1000/(time.time() - start)}")
