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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,state,action):
        action_value = self.fc1(torch.cat((state,action), dim=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q




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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def forward(self,state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        var = self.var(prob)
        var = torch.clamp(var,min=self.reparam_noise, max=1)

        return mu, var


    def sample(self,state,reparametrization = True):
        mu, var = self.forward(state)
        probabilities = torch.distributions.Normal(mu,var)

        if reparametrization:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-torch.tanh(actions).pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        

        


        return action, log_probs
    

class Agent(object):
    def __init__(self, lr_actor, lr_critic, lr_value, input_dims, n_actions
                 , env,max_action,tau = 0.005, gamma= 0.99, max_size= 1000000
                 , fc1_dim=256, fc2_dim=256, batch_size=256, reward_scale=2):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.tau = 0.005
        self.max_action = max_action

        self.actor = ActorNetwork(self.input_dims,self.n_actions,self.max_action, fc1_dim,fc2_dim,lr_actor)
        self.critic_1 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)
        self.critic_2 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic)
        self.value = ValueNetwork(self.input_dims,self.n_actions,fc1_dim,fc2_dim,lr_value)
        self.target_value = ValueNetwork(self.input_dims,self.n_actions,fc1_dim,fc2_dim,lr_value)

        self.update_network_params(1)
        self.memory = HERBuffer(0.0,self.batch_size,max_size)
        
    def update_network_params(self, tau= None):
        if tau is None:
            tau = self.tau

        for eval_param, target_param in zip(self.value.parameters(), self.target_value.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
            

    def learn(self):

        if len(self.memory.states) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, _ = self.memory.sample()[0]

        states = torch.tensor(states, dtype= torch.float32).to(self.actor.device)
        actions= torch.tensor(actions, dtype= torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype= torch.float32).to(self.actor.device).squeeze()
        next_states = torch.tensor(next_states, dtype= torch.float32).to(self.actor.device)
        dones = torch.tensor(dones, dtype= torch.int).to(self.actor.device).squeeze()

        #-------Value network update-------#
        self.value.optimizer.zero_grad()
        values = self.value.forward(states).view(-1)
        new_actions, log_probs = self.actor.sample(states,reparametrization=False)
        log_probs = log_probs.view(-1)
        critic_values_1 = self.critic_1.forward(states,new_actions)
        critic_values_2 = self.critic_2.forward(states,new_actions)

        critic_values = torch.min(critic_values_1,critic_values_2).squeeze()
        values_target = critic_values - log_probs
        value_loss = 0.5* F.mse_loss(values,values_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()


        #-------Actor network update-------#
        new_actions, log_probs = self.actor.sample(states,reparametrization=True)
     
        critic_values_1 = self.critic_1.forward(states,new_actions)
        critic_values_2 = self.critic_2.forward(states,new_actions)
        critic_values = torch.min(critic_values_1,critic_values_2).squeeze()

        self.actor.optimizer.zero_grad()
        actor_loss = log_probs - critic_values
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        #-------Critic networks update-------#

        old_critic_values_1 = self.critic_1.forward(states,actions).squeeze()
        old_critic_values_2 = self.critic_2.forward(states,actions).squeeze()

        target_values_next_states = self.target_value.forward(next_states).squeeze()
        target_values_next_states[dones] = 0
        q_hat = rewards +self.gamma*(1-dones)*target_values_next_states # might have to make (1-dones) tensor

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        critic_loss_1 = 0.5 * F.mse_loss(old_critic_values_1,q_hat)
        critic_loss_2 = 0.5 * F.mse_loss(old_critic_values_2,q_hat)

        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_params()

    def choose_action(self,obs):
        obs = torch.tensor([obs],dtype=torch.float32).to(self.actor.device)
        actions, _ = self.actor.sample(obs,reparametrization=False)

        return actions.cpu().detach().numpy()[0]

        

env = gym.make("HalfCheetah-v5")
max_action = env.action_space.high
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
lr_actor = 0.001
lr_critic = 0.001
lr_value = 0.001
max_episodes = 10000



agent = Agent(lr_actor,lr_critic,lr_value,input_dims,n_actions,env,max_action,reward_scale=2)
env.reset()


scores = []
for episode in trange(max_episodes):
    state = env.reset()[0]
    done = False
    trunc = False
    time_step = 0
    
    score = 0
    while not done and not trunc:

        action = agent.choose_action(state)
        next_state, reward, done, trunc, _ = env.step(action)

        score += reward
        agent.memory.push(state, action, reward, int(done), next_state, 1)

        agent.learn()


        state = next_state
        
        time_step+=1
    scores.append(score)
    print("episode" , episode, "score %.2f" % score, "100 game average %.2f" % np.mean(scores[-100:])
)

plt.plot(scores)
plt.show()

print("gotov")
        
