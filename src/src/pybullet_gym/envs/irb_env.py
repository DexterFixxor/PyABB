import gymnasium as gym
import numpy as np

class IRBReachEnv(gym.Env):

    def __init__(self, pb_client) -> None:
        super().__init__()
        self.client = pb_client
        # ovde izvrsiti inicijalizaciju sveta, podloga, svetlo, itd...
        # zatim ucitati robota
        self.delta_distance = 0.2
        # definises cilj (goal), random sample u nekom prostoru
    def reward(self, state):
        pass

    def step(self, action : np.ndarray):
        if type(action) is not np.ndarray:
            action = np.array(action)
        

        #for i in range(8):
        #    self.client.stepSimulation()
        
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
        pass