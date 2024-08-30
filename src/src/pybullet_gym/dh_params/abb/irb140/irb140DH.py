from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteDH
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import q2r, t2r, r2q
from spatialmath.base import transforms3d
from numpy import pi, deg2rad
import numpy as np

from ...base_dh import BaseDH

class IRB140_DH(BaseDH):
    def _update_params(self):
        
        self.name = "IRB140"
        self._offset_list = [0, -pi/2, 0, 0, 0, pi]
        self._d_list = [0.352, 0, 0, 0.380, 0, 0.065]
        self._a_list = [0.070, 0.360, 0, 0, 0, 0]
        self._alpha_list = [-pi/2, 0, -pi/2, pi/2, -pi/2, 0]
    
        self._limits = [
                [   -pi,            pi      ], # 1
                [deg2rad(-90), deg2rad(110) ], # 2
                [deg2rad(-230), deg2rad(50) ], # 3
                [deg2rad(-200), deg2rad(200)], # 4
                [deg2rad(-115), deg2rad(115)], # 5
                [deg2rad(-400), deg2rad(400)]  # 6
            ]
        
        self._q_default = np.array([0.0, 0.0, 0.0, 0.0, 0.77, 0.0], dtype=np.float32)
    
    