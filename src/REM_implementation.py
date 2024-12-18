import numpy as np
import random 
import torch
from collections import deque
import math
import time

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
        print("RADI Buffer")
        time.sleep(100)
        
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
        states = self.state_memory[batch]#[self.state_memory[i] for i in batch]
        next_states = self.next_state_memory[batch]#[self.next_state_memory[i] for i in batch]
        actions = self.actions_memory[batch]#[self.actions_memory[i] for i in batch]
        rewards = self.reward_memory[batch]#[self.reward_memory[i] for i in batch]
        dones = self.done_memory[batch]#[self.done_memory[i] for i in batch]
        goals = self.goal_memory[batch]#[self.goal_memory[i] for i in batch]
        
        return states, actions, rewards, next_states, dones, goals
        

        
class REPLAY_BUFFER(object):
    
    def __init__(self, max_len,buffer_length,batch_size, k):
        self.max_len = max_len
        self.batch_size = batch_size
        self.k = k
        self.rem_buffer = Buffer(buffer_length,self.batch_size)
        self.real_buffer = Buffer(buffer_length,self.batch_size)
        print("RADI REPLAY_BUFFER REM")
        time.sleep(100)
        
    def sample(self):
        epsilon = random.random()
        
        if epsilon < self.k:
            return self.rem_buffer.sample()
        else:
            return self.real_buffer.sample()
        
def reward_func(state, goal, epsilon = .2):
    state = np.array(state)
    goal = np.array(goal)
    distance = np.linalg.norm(goal-state, axis = -1)
    return (distance < epsilon) - 1