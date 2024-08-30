import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
from pybullet_utils import urdfEditor as ed
import numpy as np

from robot.manipulator.abb.irb140interface import IRB140Interface
from robot.gripper.robotiq.robotiq_interface import Robotiq2F140Interface

import time


p0 = bc.BulletClient(connection_mode=pb.DIRECT)
p0.setAdditionalSearchPath(pybullet_data.getDataPath())


robot = IRB140Interface(
        "IRB140",
        p0,
        basePosition=np.array([0.0, 0.0, 0.0], dtype=np.float32)
)

gripper = Robotiq2F140Interface(
        p0,
        
)
    

irb = robot.body_id
robotiq = gripper.body_id

ed0 = ed.UrdfEditor()
ed0.initializeFromBulletBody(irb, p0._client)
ed1 = ed.UrdfEditor()
ed1.initializeFromBulletBody(robotiq, p0._client)

parentLinkIndex = 6
jointPivotXYZInParent = [0, 0, 0]
jointPivotRPYInParent = [0, np.pi/2, 0]

jointPivotXYZInChild = [0, 0, 0]
jointPivotRPYInChild = [0, 0, 0]

newjoint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                        jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p0._client)

newjoint.joint_type = p0.JOINT_FIXED


ed0.saveUrdf("./urdfs/abb_irb140/irb140_robotiq.urdf")
