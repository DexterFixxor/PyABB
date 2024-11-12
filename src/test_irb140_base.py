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
from tqdm import tqdm, trange
import SAC_with_temperature


from src.pybullet_gym.envs.irb_env import IRBReachEnv



from src.pybullet_gym.interfaces.robots.irb140interface import IRB140Interface
from src.pybullet_gym.interfaces.gripper.robotiq_2f import Robotiq2F
from src.pybullet_gym.interfaces.baseJointInterface import BaseJointInterface
from src.pybullet_gym.interfaces.groups.irb140_robotiq_interface import IRB1402FGripperInterface
import time
import matplotlib.pyplot as plt


def test(env):
    
    
    state = env.reset()
    score = 0
    time_step = 0
    sac_agent.actor.eval()
    
    while time_step <episode_length:
        obs = np.concatenate([state,env.goal], axis=-1,dtype=np.float32)
        
        obs = torch.from_numpy(obs).to(sac_agent.actor.device)
        
        mu_v,var_v = sac_agent.target_actor(obs)
        action = torch.distributions.Normal(loc= mu_v, scale= var_v).sample().detach().numpy()
        action[-1] = 1
        
        reward, done,next_state , goal = env.step(state,action)
        
        
        
        state = next_state
        score += reward
        time_step +=1
        
    sac_agent.actor.train()
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
    input_dims_actor = 6
    input_dims_critic = 6
    n_actions = 8
    memory_buffer = HERBuffer()
    lr_actor = 0.001
    lr_critic = 0.001
    input_dims = 6
    max_action = 1
    
    #agent = ddpg.Agent(lr_actor=0.001, lr_critic = 0.001, input_dims_actor=input_dims_actor, input_dims_critic = input_dims_critic, tau = .05,
    #                   n_actions= n_actions,buffer= memory_buffer , layer1_size= 64, layer2_size=64, batch_size=128)
    
    sac_agent = SAC_with_temperature.Agent(lr_actor,lr_critic,input_dims,n_actions,env,max_action,reward_scale=2)


    num_of_episodes = 5
    scores = []
    test_scores = []
    episode_length = 100
    
    for episode in trange(num_of_episodes):

        time_step = 0
        score = 0
        done = False
        state = env.reset()
        

        while not done and time_step < episode_length:
            
            obs = np.concatenate([state,env.goal], axis=-1)

            if episode < 10:
                action =np.random.uniform(low = -1., high= 1., size=(n_actions,))
            else:
                action = sac_agent.choose_action(obs)
            action[-1] = 1

            next_state, reward, done, goal = env.step(state,action)
            sac_agent.memory.push(state,action,reward,done,next_state,goal)
            if episode > 10:
                buffer_sample, her_buffer_sample = sac_agent.memory.sample()
                sac_agent.learn(buffer_sample)
                sac_agent.learn(her_buffer_sample)
            
            score += reward
            state = next_state
            time_step+=1

        env.episode_index = episode
        her_goal = state
       
        
        states = np.array(sac_agent.memory.states)
        actions = np.array(sac_agent.memory.actions)
        next_states = np.array(sac_agent.memory.next_states)
        her_rewards = env.reward(sac_agent.memory.states,her_goal)
        her_rewards = np.reshape(her_rewards,(time_step,1))
        dones = her_rewards.copy()
        dones = np.add(dones,1)
        #dones = env.check_if_done(sac_agent.memory.states,her_goal)
        #dones = np.reshape(dones,(time_step,1))
        sac_agent.memory.her_buffer.append(states,actions, her_rewards,next_states, dones,[her_goal]*time_step)
        #for i in range(5)


        scores.append(score)
        print("episode" , episode, "score %.2f" % score, "100 game average %.2f" % np.mean(scores[-100:]))
        sac_agent.memory.reset_episode()

    """
    for index in trange(num_of_episodes):
      
       
        time_step = 0
        state = env.reset() # vraca poz i orient ee i goal as numpy
       
        done = 0
        score = 0
        
        while time_step < episode_length and not done :
            obs = np.concatenate([state,env.goal], axis=-1)
            action = agent.choose_action(obs,0).detach().numpy()
            action[-1] = 1
          
            next_state, reward, done , goal = env.step(state,action)
            
            agent.memory.push(state,action,reward, done,next_state, goal)
            state = next_state
            
            score += reward
            time_step +=1
            
    
        scores.append(score)

        #---------------Memory loading -------------------#
        her_goal = state
        state_ids = np.array(range(time_step))

        states = np.array(agent.memory.states)
        actions = np.array(agent.memory.actions)
        next_states = np.array(agent.memory.next_states)
     
        #her_reward = reward_func(states[state_ids],her_goal)
        her_reward = env.reward(states[state_ids],her_goal)
        her_reward = np.reshape(her_reward,(time_step,1))
        dones = her_reward.copy()
        dones = np.add(dones,1)
        
        agent.memory.her_buffer.append(states[state_ids],actions[state_ids], her_reward[state_ids], 
                                       next_states[state_ids], dones[state_ids],[her_goal])
            
        
        
        
        #---------------Learning -------------------#
        if len(agent.memory.buffer.state_memory) > 6000:
            for i in range(50):
                buff_batch, her_buff_batch = agent.memory.sample()
                agent.learn(buff_batch)
                agent.learn(her_buff_batch)
                #print("learning")
            
        if index % 50 ==0 and index >= 100:
            agent.epsilon = max(agent.epsilon*agent.epsilon_decay, 0.01)
            
        print("Epizoda:", index, "score: %.2f" %score, "epsilon %.2f" %agent.epsilon,
              "average score: %.2f" %np.mean(scores[-100:]))
        if  index > 15000 :
            proximity = 0.1
            
        if index %50 ==0 and index > 100:
            test_score = test(env)
            test_scores.append(test_score)
            print("Test result: %.2f" %test_score, "Average test result: %.2f" %np.mean(test_scores[-100:]))
                
            
        
        
        agent.memory.reset_episode()
        
        
        #elif index >1500 and index <2000:
        #    target_pos = np.array([0.712, 0.4, 0.1])
        #print(scores)
    """


    
      
plt.plot(scores)
plt.show()

    
 
