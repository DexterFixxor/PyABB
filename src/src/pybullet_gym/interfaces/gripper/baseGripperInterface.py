import numpy as np
from typing import List
from ..baseJointInterface import BaseJointInterface


class GripperInterface(BaseJointInterface):
    
    def __init__(self, client, body_id, joint_names : List[str], mimic_config :dict, default_pose : np.ndarray):
        super().__init__(client, body_id, joint_names, default_pose)
        
        self._mimic_config = mimic_config
        # self._mimic_joint_multipliers = np.zeros(len(joint_names)) # sorted in id asscending
        
        self._update_mimic_parameters()
        
       
    def _update_mimic_parameters(self):
        
        mimic_joints = []
        # Loop trough parents
        for key, value in self._mimic_config.items():
            
            for joint in self.joints:
                if joint.joint_name == key:
                    mimic_joints.append((joint.joint_id, 1))
            
            if isinstance(value, dict):
                # Loop trough mimic joints for given parent
                for key_mimic, value_mimic in self._mimic_config[key].items():
                   for joint in self.joints:
                        if joint.joint_name == key_mimic:
                            mimic_joints.append((joint.joint_id, value_mimic))
            

               
        mimic_joints.sort(key=lambda x: x[0])
        
        self._mimic_joint_multipliers = np.array(
            [x[1] for x in mimic_joints]
        )
        
    def _apply_action(self, action : float):
        self.setJointTargetPosition(action * self._mimic_joint_multipliers)
        