import os
import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import time
import random
from buffer import HERBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


#https://arxiv.org/pdf/1812.05905 - SAC with automatic entropy adjustment


DEVICE = 'cpu'

class CriticNetwork(nn.Module):
    def __init__(self,input_dims, n_actions, fc1_dims, fc2_dims, lr_critic):
        super(CriticNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_actor = lr_critic

        self.fc1 = nn.Sequential(
                nn.Linear(self.input_dims + self.n_actions, self.fc1_dims),
                nn.ReLU()
            )

        self.fc2 = nn.Sequential(
                nn.Linear(self.fc1_dims,self.fc2_dims),
                nn.ReLU()
            )
        self.q = nn.Linear(self.fc2_dims,1)

        self.optimizer = optim.Adam(self.parameters(),lr=lr_critic)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state,action):
        action_value = self.fc1(torch.cat((state,action), dim=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q



"""
class ValueNetwork(nn.Module):
    def __init__(self,input_dims, n_actions, fc1_dims, fc2_dims, lr_value):
        super(ValueNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_actor = lr_value

        self.fc1 = nn.Sequential(
                nn.Linear(self.input_dims , self.fc1_dims),
                nn.ReLU()
            )

        self.fc2 = nn.Sequential(
                nn.Linear(self.fc1_dims,self.fc2_dims),
                nn.ReLU()
            )
        self.v = nn.Linear(self.fc2_dims,1)
        self.optimizer = optim.Adam(self.parameters(),lr=lr_value)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)
        v = self.v(state_value)

        return v
"""

class ActorNetwork(nn.Module):
    def __init__(self,input_dims, n_actions, max_action, fc1_dims, fc2_dims, lr_actor):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_actor = lr_actor
        self.reparam_noise = float(1e-6)

        self.fc1 = nn.Sequential(
                nn.Linear(self.input_dims, self.fc1_dims),
                nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dims,self.fc2_dims),
            nn.ReLU()
        )
        self.mu = nn.Linear(self.fc2_dims,self.n_actions)
        self.var = nn.Linear(self.fc2_dims,self.n_actions)
            
        self.optimizer = optim.Adam(self.parameters(),lr=lr_actor)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
    def forward(self,state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        log_var = self.var(prob)
        #var = torch.clamp(log_var,min=self.reparam_noise, max=1)
        log_var = torch.clamp(log_var,min=-20, max=2)
        return mu, log_var


    def sample(self,state,reparametrization = True):
        mu, log_var = self.forward(state)
        var = log_var.exp()
        probabilities = torch.distributions.Normal(mu,var)

        if reparametrization:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-torch.tanh(actions).pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        #zbog cega se suma radi na kraju umesto u koraku sa minusom
        


        return action, log_probs
    

class Agent(object):
    def __init__(self, lr_actor, lr_critic, input_dims, n_actions
                 , env,max_action,tau = 0.005, gamma= 0.99, max_size= 1000000
                 , fc1_dim=256, fc2_dim=256, batch_size=256, reward_scale=2):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.tau = tau
        self.max_action = max_action


        self.actor = ActorNetwork(self.input_dims,self.n_actions,self.max_action, fc1_dim,fc2_dim,lr_actor)
        self.critic_1 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)
        self.target_critic_1 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)
        self.critic_2 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)
        self.target_critic_2 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)


        self.target_entropy = - self.n_actions
        self.temperature = 0.2
        self.log_temperature = torch.tensor([0.0], requires_grad=True, device= DEVICE)
        self.temperature_optimizer = optim.Adam(params=[self.log_temperature],lr=lr_actor)



        #self.value = ValueNetwork(self.input_dims,self.n_actions,fc1_dim,fc2_dim,lr_value)
        #self.target_value = ValueNetwork(self.input_dims,self.n_actions,fc1_dim,fc2_dim,lr_value)
#   
        self.update_network_params(1)
        self.memory = HERBuffer(0.5,self.batch_size,max_size)
        
    def update_network_params(self, tau= None):
        if tau is None:
            tau = self.tau

        for eval_param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
        for eval_param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
            

    def learn(self,batch):

        if len(self.memory.buffer.state_memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, goals = batch

        states = torch.tensor(states, dtype= torch.float32).to(self.actor.device)
        actions= torch.tensor(actions, dtype= torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype= torch.float32).to(self.actor.device).squeeze()
        next_states = torch.tensor(next_states, dtype= torch.float32).to(self.actor.device)
        dones = torch.tensor(dones, dtype= torch.int).to(self.actor.device).squeeze()
        goals = torch.tensor(goals, dtype= torch.float32).to(self.actor.device)
       
        obs = torch.concat((states,goals),dim=1)
        obs_ = torch.concat((next_states,goals),dim=1)

        #-------Critic networks update-------#

        old_critic_values_1 = self.critic_1.forward(obs,actions).squeeze()
        #old_critic_values_2 = self.critic_2.forward(obs,actions).squeeze()
        with torch.no_grad():
            new_actions, log_probs = self.actor.sample(obs_,reparametrization=False)
            log_probs = log_probs.view(-1)


            target_values_next_states_1 = self.target_critic_1.forward(obs_,new_actions).squeeze()
            #target_values_next_states_2 = self.target_critic_2.forward(obs_,new_actions).squeeze() 
            #target_values_next_states = torch.min(target_values_next_states_1,target_values_next_states_2)  - self.temperature* log_probs
            #target_values_next_states_1[dones] = 0
            q_hat = rewards +self.gamma*(1-dones)*(target_values_next_states_1 - self.temperature* log_probs) # target_values_next_states and without temp*log_probs if 2 critics
            #skloni gradijent sa q_hat


        self.critic_1.optimizer.zero_grad()
        #self.critic_2.optimizer.zero_grad()

        #proveri koji gradijent se koristi
        critic_loss_1 = F.mse_loss(old_critic_values_1,q_hat) # *0.5 
        #critic_loss_2 = 0.5 * F.mse_loss(old_critic_values_2,q_hat)

        critic_loss = critic_loss_1 #+ critic_loss_2
        critic_loss.backward()

        self.critic_1.optimizer.step()
        #self.critic_2.optimizer.step()

        #-------Actor network update-------#
        new_actions, log_probs = self.actor.sample(obs,reparametrization=True)

        critic_values_1 = self.critic_1.forward(obs,new_actions)
        #critic_values_2 = self.critic_2.forward(obs,new_actions)
        #critic_values = torch.min(critic_values_1,critic_values_2).squeeze()

        
        log_probs_temp = self.temperature * log_probs
        

        self.actor.optimizer.zero_grad()
        actor_loss = log_probs_temp - critic_values_1 # critic_value if using 2 critics
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        #-------Temperature network update-------#
        self.temperature = torch.exp(self.log_temperature)
        new_actions, log_probs = self.actor.sample(obs,reparametrization=False)
        with torch.no_grad():
            loss = log_probs + self.target_entropy


        temperature_loss =-1 * torch.mean( self.log_temperature *loss)
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature = torch.exp(self.log_temperature)


        self.update_network_params()


    def choose_action(self,obs):
        obs = torch.tensor([obs],dtype=torch.float32).to(self.actor.device)
        actions, _ = self.actor.sample(obs,reparametrization=False)

        return actions.cpu().detach().numpy()[0]

def reward(state,goal):
        
        
        distance = np.linalg.norm(goal-state, axis = -1)
        
      

        return np.array([(distance < 0.05) - 1])