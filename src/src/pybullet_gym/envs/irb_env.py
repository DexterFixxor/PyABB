import gymnasium as gym
import numpy as np
import pybullet_data

from src.pybullet_gym.interfaces.groups.irb140_robotiq_interface import IRB1402FGripperInterface
from src.pybullet_gym.interfaces.robots.irb140interface import IRB140Interface
#from ...irb140_robotiq_interface import IRB1402FGripperInterface

class IRBReachEnv(gym.Env):

    def __init__(self, pb_client) -> None:
        super().__init__()
        self.client = pb_client
        # ovde izvrsiti inicijalizaciju sveta, podloga, svetlo, itd...
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        self.client.setGravity(0, 0, -9.81)
        
        # zatim ucitati robota
        self.robot = IRB1402FGripperInterface(self.client)
        
        
        
        
        
        
        self.delta_distance = 0.2
        # definises cilj (goal), random sample u nekom prostoru
        self.goal = np.concatenate([np.random.normal(0.7,0.1,1),np.random.normal(0.,0.3,1),np.array([0.]),np.array([0,1,0,0])])

        self.point =   self.client.addUserDebugPoints(
                                    pointPositions=[self.goal[:3]],
                                    pointColorsRGB = [[1.0, 1.0, 0.0]],
                                    pointSize=10
                                    )
        


        
    def reward(self, state,goal = None):
        if goal == None:
            goal = self.goal
        
        
        distance = np.linalg.norm(goal[:3]-state[:3], axis = -1)
        return np.array([(distance < self.delta_distance) - 1])

    def step(self,state, action : np.ndarray):
        if type(action) is not np.ndarray:
            action = np.array(action)
        
        self.robot.step(action)
        for i in range(8):
            self.client.stepSimulation()
        self.robot._update_states()
        
        next_state = np.concatenate([self.robot._ee_position, self.robot._ee_orientation])
        reward = self.reward(state)
        
        done = np.array([0])
        
        return reward, done, next_state, self.goal
        #return np.array of next_state, goal, reward, done
        # jos bolje tensor
        #return torch.Tensor of next_state, goal, reward, done
        """ 
        return {
            "next_state": torch.Tensor(next_state),
            "goal" : self.goal
            "reward" : torch.Tensor(reward)
            "done": torch.Tensor(done)
                }
        """

    def reset(self):
        # pri svakom resetu random sample new goal
        #self.goal = np.concatenate([np.random.normal(0.7,0.1,1),np.random.normal(0.,0.1,1),np.array([0.]),np.array([0,1,0,0])])
        self.robot._robotInterface.reset()
        self.robot._update_states()
        
        #self.client.removeUserDebugItem(self.point)
        #self.point = self.client.addUserDebugPoints(
        #pointPositions=[self.goal[:3]],
       # pointColorsRGB = [[1.0, 1.0, 0.0]],
        #pointSize=10
        #)
        
        return np.concatenate([self.robot._ee_position, self.robot._ee_orientation])
        
        
        
    
    