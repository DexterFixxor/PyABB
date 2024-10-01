import numpy as np
import random
import torch
from collections import deque

class HERBuffer():

    def __init__(self, odnos = 0.5, batch_size = 256, buffer_len = int(1e6)):

        self.her_batch = int(odnos * batch_size)
        self.batch = int((1-odnos) * batch_size)
        self.buffer_len = buffer_len
        # instanciras redovan buffer
        # instanciras HER buffer
        self.her_buffer = ReplayBuffer(buffer_len, self.her_batch)
        self.buffer = ReplayBuffer(buffer_len, self.batch)
        # cuvanje cele epizode
        self.states = []
        self.actions  = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.goals = []


    def push(self, *transition):       # ----------------------------zbog čega "*" ???
        state,action, reward, done, next_state, goal = transition
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.goals.append(goal)
        
        self.buffer.append(state, action, reward, next_state, done, goal)
        pass

    def reset_episode(self):
        
        self.states = []
        self.actions  = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.goals = []
        
        #return new_rewards
        pass

    def sample(self):
        buffer_sample = self.buffer.sample()
        her_buffer_sample = self.her_buffer.sample()
        
        
      
        return buffer_sample, her_buffer_sample
        # sample redovan buffer
        # sample HER buffer
        #torch.cat() axis = 0
        #np.concatenate
        
    
    
    

class ReplayBuffer(object):
    def __init__(self, max_size,batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        
        self.state_memory = deque([],maxlen=self.mem_size)
        self.next_state_memory = deque([],self.mem_size)
        self.reward_memory = deque([],self.mem_size)
        self.done_memory = deque([],self.mem_size)
        self.action_memory = deque([],self.mem_size)
        self.goal_memory = deque([],self.mem_size)
        
        
    def append(self,state,action,reward,next_state, done,goal):
        if len(state.shape) == 1:
            self.state_memory.append(state)
            self.next_state_memory.append(next_state)
            self.action_memory.append(action)
            self.reward_memory.append(reward)
            self.done_memory.append(done)
            self.goal_memory.append(goal)
        else:
            self.state_memory.extend(state)
            self.next_state_memory.extend(next_state)
            self.action_memory.extend(action)
            self.reward_memory.extend(reward)
            self.done_memory.extend(done)
            self.goal_memory.extend(goal*len(state)) ## zbog istog goal u funkciji append 
                                                     #za her buffer mi ubaci samo jednom goal, 
                                                     #pa ne bude odgovarajuće  dimenzije
        
    def sample(self):
        max_memory = len(self.state_memory)
        
        batch = np.array(random.sample(range(max_memory), self.batch_size), dtype=np.int32)
        #print(batch)
        #print(self.state_memory)
        states = np.array(self.state_memory)[batch.astype(int)]
        next_states = np.array(self.next_state_memory)[batch.astype(int)]#[self.next_state_memory[i] for i in batch]
        actions = np.array(self.action_memory)[batch.astype(int)]#[self.actions_memory[i] for i in batch]
        rewards = np.array(self.reward_memory)[batch.astype(int)]#[self.reward_memory[i] for i in batch]
        dones = np.array(self.done_memory)[batch.astype(int)]#[self.done_memory[i] for i in batch]
        goals = np.array(self.goal_memory)[batch]#[self.goal_memory[i] for i in batch]
        
        
        
        return states, actions, rewards, next_states, dones, goals
