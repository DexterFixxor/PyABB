from kuka_env import KukaBaseEnv
from torchrl.envs import ParallelEnv , GymEnv
from torchrl.envs.utils import check_env_specs
import pybullet
from time import time
import numpy as np

def make_env(id):
    return KukaBaseEnv(id=id, state_size=3, action_size=2)
if __name__ == "__main__":
    n_env = 32
    config_list = [
        {
            "id":i
        } 
        for i in range(n_env)
    ] 
    env = ParallelEnv(n_env, create_env_fn=make_env, create_env_kwargs=config_list)


    td = env.reset()

    frequencies = []
    for i in range(100):
        start = time()
        td = env.reset()

        end = time()

        frequency = 1 / (end - start)
        frequencies.append(frequency)
        
        
    #print(frequencies)
    print(f"Average frequency: {np.average(frequencies)}")

