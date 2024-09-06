import numpy as np
from pybullet_utils.bullet_client import BulletClient
from spatialmath.base.quaternions import qnorm, qqmul, r2q, q2r
import pybullet as pb

from ..gripper.robotiq_2f import Robotiq2F
from ..robots.irb140interface import IRB140Interface

class IRB1402FGripperInterface:
    
    def __init__(self, client : BulletClient):
        
        self._client = client
        self._urdf_path = 'urdfs/abb_irb140/irb140_robotiq.urdf' 
        self._body_id = self._client.loadURDF(self._urdf_path)
        
        self._dt = 1./240.
        
        # relative to link_6
        self._tool_pos = np.array([0.0, 0.0, 0.22])
        
        self._robotInterface = IRB140Interface(
            self._client,
            self._body_id,
            self._tool_pos
        )
        
        self._gripper_interface = Robotiq2F(
            self._client,
            self._body_id
        )
        
        self._joint_positions, self._joint_velocities, _ = self._robotInterface.getJointStates()
        self._gripper_state : float = 0
        
        self._ee_position, self._ee_orientation = self._robotInterface.calculateDrectKinematics(self._joint_positions)
        
        self._attached_bodies = {}
        
        self._update_states()
        
        
    def _update_states(self):
        self._joint_positions, self._joint_velocities, _ = self._robotInterface.getJointStates()
        gripper_angles, _, _ = self._gripper_interface.getJointStates()
        self._gripper_state = gripper_angles[0] / self._gripper_interface.joint_ranges[0] # normalize gripper state
        self._ee_position, self._ee_orientation = self._robotInterface.calculateDrectKinematics(self._joint_positions)
    
    def _set_ee_pose(self, position : np.ndarray, orientation : np.ndarray):
        q_target = self._robotInterface.calculateInverseKinematics(self._joint_positions, position, orientation)

        if q_target is not None:
            self._robotInterface.setJointTargetPosition(q_target)
        else:
            self._robotInterface.setJointTargetPosition(self._joint_positions)
        
    def step(self, actions : np.ndarray):
        """
            actions should be of len == 8
            [dx, dy, dz], [x,y,z,w], gripper
        """
        gripper_action = np.clip(actions[-1], 0., 1.)
        
        delta_pos = actions[:3] * self._dt
        
        delta_rot = actions[3:6]
        delta_rot = self._ee_orientation * np.append(delta_rot, 0.0) * self._dt
        delta_rot = delta_rot / 2
        
        
        new_pos = self._ee_position + delta_pos
        new_rot = delta_rot / qnorm(delta_rot)
        
        self._set_ee_pose(new_pos, new_rot)
                        
        target_gripper = self._gripper_state + gripper_action
        target_gripper = np.clip(target_gripper, 0, 1)

        self._gripper_interface.setGripperAction(target_gripper)
        
        
    def attach_body(self, child_id):
        
        constraint_id = self._client.createConstraint(
            parentBodyUniqueId=self._body_id, 
            parentLinkIndex = 9, 
            childBodyUniqueId = child_id, 
            childLinkIndex = -1,
            jointType=pb.JOINT_FIXED,
            jointAxis = [0, 0, 1],
            parentFramePosition=self._tool_pos + 1,
            childFramePosition = [0, 0, 0])
    
        self._attached_bodies[child_id] = constraint_id 
        
        
    def dettach_bod(self, child_id):
        self._client.removeConstraint(
             self._attached_bodies.pop(child_id)
        )
        
       
        
        
        
        