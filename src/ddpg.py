import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np


import random
from buffer import HERBuffer



import matplotlib.pyplot as plt

CHECKPOINT_DIR_PATH = "/home/viktor/Documents/DIPLOMSKI/reinforcement learning/RL_Exercises_1/models/ddpg"

      
class CriticNetwork(nn.Module):
    def __init__(self,lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir= CHECKPOINT_DIR_PATH):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr_critic = lr_critic
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
      
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dims + self.n_actions, self.fc1_dims),
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
        
        
        self.action_value = nn.Sequential(
            nn.Linear(self.n_actions, self.fc1_dims),
            nn.ReLU()

        )
        
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr_critic)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state, action):
       
        x = self.fc1(torch.cat([state,action], dim =1))
        #x = self.fc1(state)
        x = self.fc2(x)
        #actions = self.action_value(action)
        #x = self.fc2(torch.cat([x,actions],dim = 1))
        x = self.fc3(x)
        
        #state_action_value = torch.add(x,action_value)

        state_action_value = self.q(x)
        
        return state_action_value
    
   
        
class ActorNetwork(nn.Module):
    def __init__(self,lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = CHECKPOINT_DIR_PATH):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr_actor = lr_actor
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
       
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_actor)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,obs):
        x = self.fc1(obs)
        #x = self.bn1(x)
        #x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
       
        mu = self.mu(x)
        var = torch.clip(self.var(x),max= 0.2)
        #var = self.var(x)
        return mu,var
        
    def save_checkpoint(self):
        print("----- saving model -----")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self):
        print("----- loading model -----")
        self.load_state_dict(torch.load(self.checkpoint_file))
            
class Agent(object):
    def __init__(self, lr_actor, lr_critic, input_dims_actor, input_dims_critic, tau, n_actions,buffer, gamma = 0.99,
                 max_size = 1000000, layer1_size = 256, layer2_size = 256, batch_size = 256
                 ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions


        #self.memory = REPLAY_BUFFER(max_size, self.batch_size,0.7)
        self.memory = buffer
        self.epsilon = 0.9
        self.epsilon_decay = 0.95
        self.actor = ActorNetwork(lr_actor, input_dims_actor, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Actor")
        self.target_actor = ActorNetwork(lr_actor, input_dims_actor, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Target_Actor")
        self.critic = CriticNetwork(lr_critic, input_dims_critic, layer1_size, layer2_size,
                                    n_actions=n_actions, name="Critic")
        self.target_critic = CriticNetwork(lr_critic, input_dims_critic, layer1_size, layer2_size,
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
            mu = torch.distributions.Normal(loc= mu_v, scale= var_v).rsample()
            
           
            #noise = torch.distributions.Normal(0,0.1)
            #mu_prime = torch.clip(mu + noise.sample().to(self.actor.device),min= -1, max = 1)
            mu = torch.clip(mu,min= -1, max = 1)
          
        else:
            #mu = np.random.uniform(low = -1., high= 1., size=(self.n_actions,))
            mu = torch.normal(0,.5,(self.n_actions,))
            #mu = torch.tensor(mu)
          
        return mu
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state,action,reward,next_state,done)
        
    def learn(self,batch): #ubacen batch u argumente da bi mogli da samplujemo memoriju van ove funkcije
      
        self.learn_counter+= 1

        states, actions, rewards, next_states, dones, goals = batch

        states = torch.tensor(states, dtype= torch.float32).to(self.critic.device)
        actions= torch.tensor(actions, dtype= torch.float32).to(self.critic.device)
        rewards = torch.tensor(rewards, dtype= torch.float32).to(self.critic.device).squeeze()
        next_states = torch.tensor(next_states, dtype= torch.float32).to(self.critic.device)
        dones = torch.tensor(dones, dtype= torch.float32).to(self.critic.device).squeeze()
        #dones[-1] = 1
        goals = torch.tensor(goals, dtype= torch.float32).to(self.critic.device)
        
        obs = torch.cat([states,goals],dim=1)
        obs_ = torch.cat([next_states,goals],dim=1)

        #target_mu, target_var = self.target_actor(obs_)
        #target_actions = torch.clip(torch.normal(target_mu, target_var),min=-1, max=1)
        target_actions, _ = self.target_actor(obs_)
        
        
        critic_value = self.critic(obs,actions).squeeze()
        with torch.no_grad(): 
            target_critic_next_value = self.target_critic(obs_,target_actions).squeeze()
            
            target_ = rewards + (self.gamma* target_critic_next_value*(1-dones))
           
          
        
        
        
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target_,critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        

        new_actions = self.choose_action(obs,1)
        self.actor.optimizer.zero_grad()
        actor_loss = -1 * self.critic(obs, new_actions)                             
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
    def update_network_parameters(self,tau = None):
        if tau is None:
            tau = self.tau
   
        for eval_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
            
        for eval_param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
        
  
     
        
"""
env = gym.make("Pendulum-v1")
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

env_render = gym.make("Pendulum-v1", render_mode = "human")
print(input_dims, n_actions)




buf = HERBuffer()
agent = Agent(lr_actor =0.001, lr_critic= 0.001, input_dims_actor= input_dims, input_dims_critic= input_dims,
              tau = 0.005, n_actions = n_actions,buffer=buf,layer1_size=128,layer2_size=128,batch_size=100)


np.random.seed(0)
num_of_episodes = 10000
score_history = []

for i in range(num_of_episodes):
    done = False
    trunc = False
    state = env.reset()[0]
    score = 0
    time_step = 0
    
    while not done and not trunc:
        action = agent.choose_action(state,0).detach().numpy()
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.push(state,action,reward,done,next_state,[0])
        if len(agent.memory.states) > agent.batch_size:
       
            batch = agent.memory.sample()
            agent.learn(*batch)
             
    
        score += reward
        state = next_state
        time_step+=1
    agent.epsilon *= agent.epsilon_decay  
    
            

    score_history.append(score)
    print("episode" , i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:]),
          "broj ucenja", agent.learn_counter, "epsilon %.5f" %agent.epsilon)
    
    #if agent.epsilon < 0.1:
    #    env = env_render
    #if i %25 == 0:
        #agent.save_models()
    if i > 20:
        agent.epsilon *= agent.epsilon_decay   
        #env = env_render
    
        
plt.plot(score_history)
plt.show()


"""
