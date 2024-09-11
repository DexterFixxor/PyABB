import numpy as np
import random 
import torch
from collections import deque
import math

class Buffer(object):
    def __init__(self, max_len,batch_size):
        self.max_len = max_len
        self.batch_size = batch_size
        self.state_memory = deque([],self.max_len)
        self.reward_memory = deque([],self.max_len)
        self.goal_memory = deque([],self.max_len)
        self.actions_memory = deque([],self.max_len)
        self.next_state_memory = deque([],self.max_len)
        self.done_memory = deque([],self.max_len)
        
    def append(self, state, action, reward, next_state, done, goal):
        self.state_memory.append(state)
        self.actions_memory.append(action)
        self.reward_memory.append(reward)
        self.next_state_memory.append(next_state)
        self.goal_memory.append(goal)
        self.done_memory.append(done)
    
    
    def sample(self):
        max_memory = len(self.state_memory)
        batch = random.sample(range(max_memory), self.batch_size)
        states = [self.state_memory[i] for i in batch]
        next_states = [self.next_state_memory[i] for i in batch]
        actions = [self.actions_memory[i] for i in batch]
        rewards = [self.reward_memory[i] for i in batch]
        dones = [self.done_memory[i] for i in batch]
        goals = [self.goal_memory[i] for i in batch]
        
        return states, actions, rewards, next_states, dones, goals
        

        
class REPLAY_BUFFER(object):
    
    def __init__(self, max_len,buffer_length,batch_size, k):
        self.max_len = max_len
        self.batch_size = batch_size
        self.k = k
        self.rem_buffer = Buffer(buffer_length,self.batch_size)
        self.real_buffer = Buffer(buffer_length,self.batch_size)
        
        
    def sample(self):
        epsilon = random.random()
        
        if epsilon < self.k:
            return self.rem_buffer.sample()
        else:
            return self.real_buffer.sample()
        
def reward_func(state, goal, epsilon = .2):
    
    x,y,z = state[:3]
    goal_x, goal_y, goal_z = goal[:3]
    
    distance = math.sqrt((x- goal_x)**2 + (y- goal_y)**2 +(z- goal_z)**2)
    
    if distance < epsilon:
        return 0
    
    
    
    return -1