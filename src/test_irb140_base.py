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
    
    
    
    client = bc.BulletClient(connection_mode=pb.DIRECT)
    
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    client.setGravity(0, 0, -9.81)
    
    
    # cube_id = client.loadURDF('cube_small.urdf', useMaximalCoordinates=True, )
    # client.resetBasePositionAndOrientation(cube_id, [0.712, 0, 0.026], [0, 0, 0, 1])

    # client.changeDynamics(
    #     bodyUniqueId = cube_id,
    #     linkIndex = -1, 
    #     mass = 0.2,
    #     lateralFriction = 1,
    #     spinningFriction = 0.1,
    #     rollingFriction = 0.1,
    #     restitution = 0,
    #     linearDamping = 0.4,
    #     angularDamping = 0.4,
    #     contactStiffness = 1000,
    #     contactDamping = 1,
    #     frictionAnchor = False,        
    # )
    
    
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
    
    input_dims = 7
    n_actions = 8
    agent = ddpg.Agent(alpha=0.001, beta = 0.001, input_dims=input_dims, tau = 0.005,
                       n_actions= n_actions, layer1_size= 128, layer2_size=128, batch_size=256)
    
    num_of_episodes = 20000
    scores = []
    test_scores = []
    for index in range(num_of_episodes):
      
        goal = np.concatenate([np.array(target_pos).flatten(), np.array(target_rot).flatten()])
        proximity = 0.15
        time_step = 0
        states = []
        actions = []
        next_states = []
        dones = []
        
        robot._robotInterface.reset()
        robot._update_states()
        state = np.concatenate([robot._ee_position, robot._ee_orientation])
        score = 0
        
        while time_step < 200:
            action = agent.choose_action(state,0)
            
            #print(action)
            action = torch.Tensor.cpu(action).detach().numpy()
            actions.append(action)
            #print(action)
            
            action[-1] = 1
            #print(action)
            #time.sleep(123)
            
            
            robot.step(action)
            robot._update_states()
            #next_state = robot._ee_position, robot._ee_orientation, robot._gripper_state
            next_state = np.concatenate([robot._ee_position, robot._ee_orientation])
            
            
            
            
            """
            robot._gripper_interface.setGripperAction(0.0)
            if time.time() - start <3:
                target_pos = np.array([0.7, 0, 0.1])
                target_q = np.array([0, 0, 0, 1])
            
            elif time.time() - start > 3 and time.time() - start < 5:
                target_pos = np.array([0.7, 0, 0.1])
                target_q = np.array([0, 1, 0, 0])
            
            elif time.time() - start > 5 and time.time() - start < 7:
                target_pos = np.array([0.7, -0.3, 0.1])
                target_q = np.array([0, 1, 0, 0])
            elif time.time() - start > 7 and time.time() - start < 9:
                target_pos = np.array([0.7, 0.3, 0.1])
                target_q = np.array([0, 1, 0, 0])    
            """
            """
            if time.time() - start < 3:
                robot._gripper_interface.setGripperAction(0)
                robot._set_ee_pose(target_pos, target_q)
            elif time.time() - start > 3 and time.time() - start < 5:
                robot._gripper_interface.setGripperAction(0.65)
            elif time.time() - start < 7:
                target_pos[-1] = 0.3
                robot._set_ee_pose(target_pos, target_q)
                robot._gripper_interface.setGripperAction(0.65)
            else:
                target_pos[-1] = 0.1
                robot._gripper_interface.setGripperAction(0.65)
                robot._set_ee_pose(target_pos, target_q)
            """
            #robot._set_ee_pose(target_pos,target_q)
            
            for i in range(8):
                client.stepSimulation()
            
            real_reward = reward_func(state, goal,proximity)
            if real_reward == 0:
                done = 1
            else:
                done = 0
                
            states.append(state)
            
            next_states.append(next_state)    
            dones.append(done)
            agent.memory.real_buffer.append(state,action, real_reward, next_state, done)
            
            score += real_reward
            state = next_state
            
            time_step +=1
            
            #time.sleep(0.1/240.0)
        #print(f"F: {1000/(time.time() - start)}")
        dones[-1] = 1
        scores.append(score)
        #---------------Memory loading -------------------#
        rem_goal = states[-1]
        #for t in range(time_step):
            #real_reward = reward_func(states[t], goal)
            #agent.memory.real_buffer.append(states[t],actions[t], real_reward, next_states[t], dones[t])
        state_ids = np.array(range(time_step))

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        rem_reward = reward_func(states[state_ids],rem_goal, proximity)
        agent.memory.rem_buffer.append(states[state_ids],actions[state_ids], rem_reward, next_states[state_ids], dones[state_ids])
            
        #score = sum(agent.memory.real_buffer.reward_memory[-100:])
        
        
        #---------------Learning -------------------#
        if len(agent.memory.real_buffer.state_memory) > 512:
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
            
        
        #elif index >1500 and index <2000:
        #    target_pos = np.array([0.712, 0.4, 0.1])
        #print(scores)



    
      
plt.plot(scores)
plt.show()

    
 
