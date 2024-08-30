from typing import List
import numpy as np
from .baseGripperInterface import GripperInterface


class Robotiq2F(GripperInterface):
    
    def __init__(self, client, body_id):
        joint_names = [
            "finger_joint", 
            "left_inner_knuckle_joint", 
            "left_inner_finger_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint" ,
            "right_inner_finger_joint"
        ]
    
        mimic_config = {
            "finger_joint":{
                "left_inner_knuckle_joint" : -1.0, 
                "left_inner_finger_joint" : 1.0,
                "right_outer_knuckle_joint" : -1.0,
                "right_inner_knuckle_joint" : -1.0 ,
                "right_inner_finger_joint" : 1.0
            }
        }
        
        super().__init__(client, body_id, joint_names, mimic_config, default_pose=np.zeros(len(joint_names)))
        
    def setGripperAction(self, action : float):
        """
        Robotiq 2F 140 joint range is [0, 0.725], but we want to clip it to 0.7
        so we rescale action in range [0, 0.7]
        """
        self._apply_action(action * 0.7)