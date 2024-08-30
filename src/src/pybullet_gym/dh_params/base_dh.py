from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteDH
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import q2r, t2r, r2q
from spatialmath.base import transforms3d
from numpy import pi, deg2rad
import numpy as np


class BaseDH:
    
    def __init__(self, tool_pos :np.ndarray = np.array([0, 0, 0], dtype=np.float32)):
        
    
        self._name = "BaseDH"
        self._offset_list = np.array([])
        self._d_list = np.array([])
        self._a_list = np.array([])
        self._alpha_list = np.array([])
    
        self._limits = np.array([])
        self._q_default = np.array([])
        
        self._update_params()
        
        self.dh_robot = DHRobot(
            links =  [
                RevoluteDH(
                    offset=offset,
                    a = a, 
                    d= d, 
                    alpha = alpha, 
                    qlim=limit) 
                
                    for offset, a, d, alpha, limit in zip(
                        self._offset_list, 
                        self._a_list, 
                        self._d_list, 
                        self._alpha_list, 
                        self._limits)
                ], 
                name=self._name,
                tool = SE3(tool_pos))
        
        self.ets_robot = self.dh_robot.ets()
      
    def _update_params(self):
        """
            Override this function and update:
                - name
                - offset_list
                - d_list
                - a_list
                - alpha_list
                - limits
                - q_default : np.ndarray
        """
        raise NotImplementedError

    def _foward(self, q : np.ndarray):
        """
            Pose: [x,y,z]
            Orientation: [x,y,z,w] quaternion
        """
        T = self.ets_robot.fkine(q)
        return  T.t, r2q(T.R, order='xyzs')
    
    def _inverse(self, q_init: np.ndarray, pose : np.ndarray, orientation : np.ndarray):
        """
            Pose: [x,y,z]
            Orientation: [x,y,z,w] quaternion
        """
        T = SE3.Rt(SO3(q2r(orientation, order="xyzs")), pose)
        q, success, _, _, _ = self.ets_robot.ik_LM(T, q0=q_init)
        if success:
            return q
        return None
    
    @property
    def q_default(self):
        return self._q_default