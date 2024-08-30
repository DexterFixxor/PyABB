import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from robot.gripper.robotiq.robotiq_interface import Robotiq2F140Interface

import time


if __name__ == "__main__":
    
  
    client = bc.BulletClient(connection_mode=pb.GUI)
    client.setAdditionalSearchPath('/home/dexter/programming/src/pyabb/urdfs')
    
    gripper = Robotiq2F140Interface(
        client,
    )
    
    a = 0
    
    while True:
        # robot.setJointControlArray(q_desired)
        a += 0.01
        gripper.move_gripper(a)
        client.stepSimulation()
        time.sleep(1. / 30.)