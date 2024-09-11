from typing import List, Tuple
import numpy as np
 
import pybullet as pb
import pybullet_utils.bullet_client as bc
from pybullet_utils.bullet_client import BulletClient

from ..baseJointInterface import BaseJointInterface
from ...dh_params.abb.irb140.irb140DH import IRB140_DH

class IRB140Interface(BaseJointInterface):
    
    def __init__(self, client: BulletClient, rigid_body_id, tool_pose : np.ndarray = np.array([0.0, 0.0, 0.0])):
        
        joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            ]
        
        
        self._ik_solver = IRB140_DH(tool_pos=tool_pose)
    
        super().__init__(client, rigid_body_id, joint_names, self._ik_solver.q_default)
    
        self.reset()

    def calculateDrectKinematics(self, q : np.ndarray):
        """
            Computes forward kinematics for given joint positions. 
            Returns position vector, and quaternion in (x,y,z,w)
        """
        return self._ik_solver._foward(q)
        
    def calculateInverseKinematics(self, q_init : np.ndarray, position : np.ndarray, quaterion : np.ndarray):
        """
            Computes inverse kinematics for given position [x,y,z] and quaterion [x,y,z,w].
            Returns: q_target for desired position and orientation.
        """
        return  self._ik_solver._inverse(pose=position, orientation=quaterion,q_init=q_init)
            
    def reset(self):
        for id, state in zip(self._joint_indices, self.home_position):
            self._client.resetJointState(self._body_id, id, state)      
            
    def resetJointStatesWithNoise(self, std: float = 0.4):
        delta_q = np.random.randn(len(self.joint_indices)) * std
        reset_state = self.home_position + delta_q
        self._resetJointStatesToTarget(reset_state)
            