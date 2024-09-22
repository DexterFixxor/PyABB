import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import gymnasium as gym
import torch
import ddpg
import math
from REM_implementation import reward_func
import random
from buffer import HERBuffer

from src.pybullet_gym.envs.irb_env import IRBReachEnv



from src.pybullet_gym.interfaces.robots.irb140interface import IRB140Interface
from src.pybullet_gym.interfaces.gripper.robotiq_2f import Robotiq2F
from src.pybullet_gym.interfaces.baseJointInterface import BaseJointInterface
from src.pybullet_gym.interfaces.groups.irb140_robotiq_interface import IRB1402FGripperInterface
import time
import matplotlib.pyplot as plt


def test():
    
    robot._robotInterface.reset()
    robot._update_states()
    state = *robot._ee_position, *robot._ee_orientation
    score = 0
    time_step = 0
    agent.actor.eval()
    while time_step <200:
        
        action = agent.choose_action(state,1)
        action = torch.Tensor.cpu(action).detach().numpy()
        
        action[-1] = 1
        
        robot.step(action)
        robot._update_states()
        #next_state = robot._ee_position, robot._ee_orientation, robot._gripper_state
        next_state = *robot._ee_position, *robot._ee_orientation
        for i in range(8):
            client.stepSimulation()

        score += reward_func(state, goal,proximity)
        state = next_state
        
    
    
        time_step +=1
    agent.actor.train()
    return score


if __name__ == "__main__":
    
    client = bc.BulletClient(connection_mode=pb.GUI)
    env = IRBReachEnv(client)
    
    
    
    """
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    client.setGravity(0, 0, -9.81)
    
    
    cube_id = client.loadURDF('cube_small.urdf', useMaximalCoordinates=True, )
    client.resetBasePositionAndOrientation(cube_id, [0.712, 0, 0.026], [0, 0, 0, 1])

    client.changeDynamics(
        bodyUniqueId = cube_id,
        linkIndex = -1, 
        mass = 0.2,
        lateralFriction = 1,
        spinningFriction = 0.1,
        rollingFriction = 0.1,
        restitution = 0,
        linearDamping = 0.4,
        angularDamping = 0.4,
        contactStiffness = 1000,
        contactDamping = 1,
        frictionAnchor = False,        
    )
    
    
    robot = IRB1402FGripperInterface(client)
    
    
    q_init = robot._joint_positions
    print("q_init-------------", q_init)
    start_pos, start_rot = robot._robotInterface.calculateDrectKinematics(q_init)
    print("pos -----", start_pos)
    print("rot -----", start_rot)
    start_pos[2] = 0.0
    
    # rot = np.array([0, 1, 0, 0])

    client.addUserDebugPoints(
        pointPositions=[[0.612, 0, 0]],
        pointColorsRGB = [[1.0, 1.0, 0.0]],
        pointSize=10
    )
    start = time.time()
    
    gripper_action = 0
    x_pos = start_pos[0]
    y_pos = start_pos[1]
    z_pos = start_pos[2]
    
    
    gripper_action_id = client.addUserDebugParameter("gripper", 0, 1, 0)
    x_id = client.addUserDebugParameter("x", -0.7, 0.7, 0.515)
    y_id = client.addUserDebugParameter("y", -0.7, 0.7, 0.0)
    z_id = client.addUserDebugParameter("z", 0, 0.5, 0.5)
    
    dt = 0
    
    target_pos = np.array([0.712, 0, 0.1])
    target_rot = np.array([0, 1, 0, 0])
    print("target pos -----", target_pos)
    print("target rot -----", target_rot)
    # robot.attach_body(cube_id)
    start = time.time()
    
    open_len = 0.049
    angle = (0.715 - np.sin((open_len - 0.01)/0.1143)) / 0.14
    """
    input_dims = 14
    n_actions = 8
    memory_buffer = HERBuffer()
    
    agent = ddpg.Agent(alpha=0.001, beta = 0.001, input_dims=input_dims, tau = 0.005,
                       n_actions= n_actions,buffer= memory_buffer , layer1_size= 128, layer2_size=128, batch_size=256)
    
    num_of_episodes = 20000
    scores = []
    test_scores = []
    
    
    for index in range(num_of_episodes):
      
       
        time_step = 0
        state = env.reset() # vraca poz i orient ee i goal as numpy
       
        
        score = 0
        obs = np.concatenate([state,env.goal], axis=-1)
        while time_step < 200:
            action = agent.choose_action(obs,0).numpy()
            
            action[-1] = 1
           

            transition = env.step(state,action)
            transition = np.concatenate([state,action,transition],axis=0)
            agent.memory.push(transition)
            #robot.step(action)
            #robot._update_states()
            #next_state = robot._ee_position, robot._ee_orientation, robot._gripper_state
            #next_state = np.concatenate([robot._ee_position, robot._ee_orientation])
        
            #time.sleep(0.1/240.0)
        #print(f"F: {1000/(time.time() - start)}")
       
       
        scores.append(np.sum(agent.memory.rewards))
        #---------------Memory loading -------------------#
        rem_goal = agent.memory.states[-1]
        state_ids = np.array(range(time_step))

        states = np.array(agent.memory.states)
        actions = np.array(agent.memory.actions)
        next_states = np.array(agent.memory.next_states)
        dones = np.array(agent.memory.dones)
        
        rem_reward = reward_func(states[state_ids],rem_goal)
        agent.memory.her_buffer.append(states[state_ids],actions[state_ids], rem_reward, 
                                       next_states[state_ids], dones[state_ids],rem_goal)
            
        #score = sum(agent.memory.real_buffer.reward_memory[-100:])
        
        
        #---------------Learning -------------------#
        if len(agent.memory.buffer.state_memory) > 512:
            for i in range(100):
                batch = agent.memory.sample()
                agent.learn(batch)
                #print("learning")
            
        if index % 1000 ==0:
            agent.epsilon = max(agent.epsilon-0.1, 0.01)
            
        print("Epizoda:", index, "score: %.2f" %score, "epsilon %.2f" %agent.epsilon,
              "average score: %.2f" %np.mean(scores[-100:]))
        if  index > 15000 :
            proximity = 0.1
            
        if index %20 ==0:
            test_score = test()
            test_scores.append(test_score)
            print("Test result: %.2f" %test_score, "Average test result: %.2f" %np.mean(test_scores[-100:]))
                
            pass
        
        
        agent.memory.reset_episode()
        time_step +=1
        
        #elif index >1500 and index <2000:
        #    target_pos = np.array([0.712, 0.4, 0.1])
        #print(scores)



    
      
plt.plot(scores)
plt.show()

    
 
