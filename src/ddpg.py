import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import time
from collections import deque
import random
import buffer

CHECKPOINT_DIR_PATH = "/home/viktor/Documents/DIPLOMSKI/reinforcement learning/RL_Exercises_1/models/ddpg"

      
class CriticNetwork(nn.Module):
    def __init__(self,beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir= CHECKPOINT_DIR_PATH):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.beta = beta
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
      
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2_dims, self.fc2_dims),
            nn.ReLU()
        )
        
        
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        action_value = self.action_value(action)
        state_action_value = F.relu(torch.add(x,action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    def save_checkpoint(self):
        print("----- saving model -----")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self):
        print("----- loading model -----")
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self,alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = CHECKPOINT_DIR_PATH):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg")
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2_dims, self.fc2_dims),
            nn.ReLU()
        )
      
        self.mu = nn.Sequential(
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Tanh()
        )
        
        self.var = nn.Sequential(
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Softplus()
        )
       
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,state):
        x = self.fc1(state)
        #x = self.bn1(x)
        #x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        mu = self.mu(x)
        var = self.var(x)
        
        return mu,var
        
    def save_checkpoint(self):
        print("----- saving model -----")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self):
        print("----- loading model -----")
        self.load_state_dict(torch.load(self.checkpoint_file))
            
class Agent(object):
    def __init__(self, lr_actor, lr_critic, input_dims, tau, n_actions,buffer, gamma = 0.99,
                 max_size = 1000000, layer1_size = 256, layer2_size = 256, batch_size = 256
                 ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        #self.memory = REPLAY_BUFFER(max_size, self.batch_size,0.7)
        self.memory = buffer
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.actor = ActorNetwork(lr_actor, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Actor")
        self.target_actor = ActorNetwork(lr_actor, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Target_Actor")
        self.critic = CriticNetwork(lr_critic, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name="Critic")
        self.target_critic = CriticNetwork(lr_critic, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name="Target_Critic")
        
        self.update_network_parameters(tau=1)
        self.learn_flag = False
        self.learn_counter = 0
        
    
    
    def choose_action(self, obs, test_flag):
    
        if test_flag == 1:
            choice = 1.
        elif test_flag == 0:
            choice = random.random()
        self.epsilon = max(self.epsilon,0.01)
        if choice > self.epsilon:
            obs = torch.tensor(obs,dtype=torch.float32).to(self.actor.device)
            mu_v,var_v = self.actor(obs)
            mu = torch.normal(mean= mu_v, std= var_v)
           
            noise = torch.distributions.Normal(0,0.1)
            mu_prime = mu + noise.sample().to(self.actor.device)
          
        else:
            mu_prime = np.random.uniform(low = -1., high= 1., size=(8,))
           
            mu_prime = torch.tensor(mu_prime)
          
        return mu_prime
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state,action,reward,next_state,done)
        
    def learn(self,batch): #ubacen batch u argumente da bi mogli da samplujemo memoriju van ove funkcije
        """
        if len(self.memory.state_memory) < 100000:
            return
        if len(self.memory.state_memory) > 100000 and not self.learn_flag:
            self.learn_flag = True
            print("Poceo da uci")
            print(len(self.memory.state_memory))
        """
        self.learn_counter+= 1
        #state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, goals = batch
        
        #print(action, type(action))
        
        
        states = torch.tensor(states, dtype= torch.float32).to(self.critic.device)
        #print(state, type(state))
        actions= torch.tensor(actions, dtype= torch.float32).to(self.critic.device)
        #action = action.squeeze()
        #print(action)
        #time.sleep(122)
        rewards = torch.tensor(rewards, dtype= torch.float32).to(self.critic.device).squeeze()
        next_states = torch.tensor(next_states, dtype= torch.float32).to(self.critic.device)
        dones = torch.tensor(dones, dtype= torch.float32).to(self.critic.device).squeeze()
        dones[-1] = 1
        goals = torch.tensor(goals, dtype= torch.float32).to(self.critic.device)
        
        mu, var = self.target_actor(torch.cat([next_states,goals],dim=1))# da li treba next states ili states
        target_actions = torch.normal(mu, var)
        
        obs = torch.cat([next_states,goals],dim=1)
        
        critic_value = self.critic(obs,actions).squeeze()   
        with torch.no_grad(): 
            target_critic_next_value = self.target_critic(obs,target_actions).squeeze()
            
            target_ = rewards + (self.gamma* target_critic_next_value*(1-dones))
           
            #target_ = torch.tensor(target_).to(self.critic.device)
           # target_ = target_.clone()
        #target_ = target_.view(self.batch_size,1)
        
        #print(target_)
        #print(critic_value)
        #time.sleep(22)
        
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss( critic_value,target_)
        
        #print("critic_loss",critic_loss)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        
        self.actor.optimizer.zero_grad()
        #mu, var = self.actor(state)
        
        #print("actions mean",mu.mean())
        #self.actor.train()
        actor_loss = -self.critic(torch.cat([next_states,goals],dim=1), actions).mean()
        #actor_loss = torch.mean(actor_loss)
        #print("actor loss",actor_loss)

        actor_loss.backward()
        self.actor.optimizer.step()
        #print("actor_loss",actor_loss)
        #print("critic_loss" ,critic_loss)
        self.update_network_parameters()
        
    def update_network_parameters(self,tau = None):
        if tau is None:
            tau = self.tau
        
        
        
        """
        for eval_param, target_param in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            #target_param.data.copy_(1e-3*eval_param.data + (1.0-1e-3)*target_param.data)
            target_param.data.copy_(eval_param.data)
        """ 
        
        for eval_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
            
        for eval_param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
        
      
        
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        
"""    
env = gym.make("Pendulum-v1")
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

env_render = gym.make("Pendulum-v1", render_mode = "human")
print(input_dims, n_actions)

time.sleep(5)
agent = Agent(alpha=0.0001, beta = 0.001, input_dims=input_dims, tau = 0.005, 
              n_actions = n_actions,layer1_size=400,layer2_size=300,batch_size=100)

np.random.seed(0)
num_of_episodes = 10000
score_history = []

for i in range(num_of_episodes):
    done = False
    trunc = False
    state = env.reset()[0]
    score = 0
    
    while not done and not trunc:
        action = agent.choose_action(state)
        next_state, reward, done, trunc, _ = env.step(action.cpu().detach().numpy())
        agent.memory.append(state,action,reward,next_state,done)
        agent.learn()
        score += reward
        state = next_state

    score_history.append(score)
    print("episode" , i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]),
          "broj ucenja", agent.learn_counter, "epsilon %.5f" %agent.epsilon)
    
    if score > -200:
        env = env_render
    #if i %25 == 0:
        #agent.save_models()
    if i > 500:
        agent.epsilon *= agent.epsilon_decay   
    
        
plt.plot(score_history)
plt.show()

"""

