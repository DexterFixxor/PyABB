import numpy as np
import pybullet as pb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from typing import List, Tuple


class BaseJointInterface:
    
    def __init__(self,
                 client : BulletClient,
                 rigid_body_id,
                 joint_names : List[str],
                 home_position : np.ndarray
                 ):
        
        self._client = client
        self._body_id = rigid_body_id
        
        self.home_position = home_position
        
        self._joint_names = joint_names
        
        # Initialize joint parameters based on joint_name        
        num_joints = self._client.getNumJoints(self._body_id)
    
        self.joints :List[JointInfo] = []
                
        
        for i in range(num_joints):
            info = self._client.getJointInfo(self._body_id, i)
            
            joint_name = info[1].decode("utf8")
            joint_type = info[2]
            
            lower_limit = info[8]
            upper_limit = info[9]
            
            maxForce = info[10]
            maxVelocity = info[11]
            
            linkName = info[12]
            parent_link_index = info[-1]
            
            if joint_name in self._joint_names and joint_type != pb.JOINT_FIXED:
                
                new_joint = JointInfo(self._body_id, joint_name, i, joint_type, lower_limit, upper_limit, maxVelocity, maxForce)
                self.joints.append(new_joint)
                self._initialize_joint(joint_type, i, lower_limit, upper_limit, maxVelocity, maxForce)
                
            self._client.changeDynamics(
                bodyUniqueId = self._body_id,
                linkIndex = parent_link_index, 
                lateralFriction = 1,
                spinningFriction = 0,
                rollingFriction = 0,
                restitution = 0,
                linearDamping = 0.4,
                angularDamping = 0.4,
                contactStiffness = 1000,
                contactDamping = 1,
                frictionAnchor = False,        
            )
            
        
        self.joints.sort(key = lambda x: x.joint_id)
        
        self._joint_lower_limit = np.array([x.limit_lower for x in self.joints])
        self._joint_upper_limit = np.array([x.limit_upper for x in self.joints])
        self._joint_ranges = self._joint_upper_limit - self._joint_lower_limit
        
        self._max_velocities = np.array([x.maxVel for x in self.joints])
        self._max_forces = np.array([x.maxForce for x in self.joints])
        
        self._joint_indices = np.array([x.joint_id for x in self.joints])
        
        self._target_vel = np.zeros(len(self._joint_indices))
        
        self._pos_gain = 0.8
        self._vel_gain = 0.71
        self._pos_gains = np.ones(len(self._joint_indices))
        self._vel_gains = np.ones(len(self._joint_indices))

    @property
    def joint_indices(self):
        return self._joint_indices
    
    @property
    def joint_ranges(self):
        return self._joint_ranges
    
    def setJointTargetPosition(self, positions : np.ndarray):
        self._client.setJointMotorControlArray(
            bodyUniqueId = self._body_id,
            jointIndices = self._joint_indices,
            controlMode = pb.POSITION_CONTROL,
            targetPositions=positions,
            targetVelocities =  self._target_vel,
            positionGains =   self._pos_gains * self._pos_gain,
            velocityGains =  self._vel_gains  * self._vel_gain
        )
    
    def reset_home(self):
        self._resetJointStatesToTarget(self.home_position)
    
    def getJointStates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, vel, _, applied_joint_torque = zip(*self._client.getJointStates(self._body_id, self._joint_indices))
        return np.array(pos), np.array(vel), np.array(applied_joint_torque)

    def resetJointStatesWithNoise(self, std : float = 0.4):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def _resetJointStatesToTarget(self, q_reset : np.ndarray):
        """
            This is supposed to be called before simulation starts
        """
        for id, q in zip(self.joint_indices, q_reset):
            self._client.resetJointState(self._body_id, id, q) 
          
    def _initialize_joint(self, joint_type, joint_id, lower_limit, upper_limit, maxVel, maxForce):
        if joint_type == pb.JOINT_REVOLUTE:
            self._client.setJointMotorControl2(self._body_id, joint_id, controlMode=pb.VELOCITY_CONTROL, targetVelocity=0, force = 0)
            self._client.changeDynamics(self._body_id, joint_id, jointLowerLimit = lower_limit, jointUpperLimit=upper_limit)   
        
        self._client.changeDynamics(self._body_id, joint_id, maxJointVelocity=maxVel)
        self._client.changeDynamics(self._body_id, joint_id, jointLimitForce=maxForce)
        
  

class JointInfo:       
    def __init__(self, parent_body_id, joint_name, joint_id, joint_type, limit_lower = 0, limit_upper = 0, maxVelocity = 0, maxForce = 0):
        
        self.parent_body_id = parent_body_id
        self.joint_name = joint_name
        self.joint_id = joint_id
        self.joint_type = joint_type
        self.limit_lower = limit_lower
        self.limit_upper = limit_upper
        self.maxVel = maxVelocity
        self.maxForce = maxForce