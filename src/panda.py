import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from SAC_with_temperature import Agent
from SAC_with_temperature import reward as rewww
import time
from tqdm import trange
import numpy as np

env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()
input_dims = observation['observation'].shape[0] + observation["desired_goal"].shape[0]
n_actions = env.action_space.shape[0]
max_action = 1.0
lr_actor = 0.001
lr_critic = 0.001
max_episodes = 3000
episode_length = 50
start_learning_at_episode = 5

agent = Agent(lr_actor,lr_critic,input_dims,n_actions,env,max_action)

scores = []
for episode in trange(max_episodes):
    score = 0
    time_step = 0
    observation, info = env.reset()
    truncated = False
    terminated = False

    while not truncated and not terminated and time_step < episode_length:

        #action = env.action_space.sample() # random action
        obs = np.concatenate([observation["observation"],observation["desired_goal"]], axis= -1)
        action = agent.choose_action(obs)
        next_observation, reward, terminated, truncated, info = env.step(action)
    
        agent.memory.push(observation["observation"], action, reward,
                          next_observation["observation"], int(terminated),
                          observation["desired_goal"]
                          )
        
        if episode > start_learning_at_episode :
            buffer_sample, her_buffer_sample = agent.memory.sample()
            agent.learn(buffer_sample)
            agent.learn(her_buffer_sample)

        score += reward
        observation = next_observation
        time_step += 1
    
    
    
    states = np.array(agent.memory.states)
    states3 = states[:, :3]
    her_goal = states3[-1]
    actions = np.array(agent.memory.actions)
    next_states = np.array(agent.memory.next_states)
    her_rewards = rewww(states3,her_goal)
    her_rewards = np.reshape(her_rewards,(time_step,1))
    dones = her_rewards.copy()
    dones = np.add(dones,1)

    agent.memory.her_buffer.append(states,actions,her_rewards,next_states, dones, [her_goal]*time_step)

    scores.append(score)
    print("episode" , episode, "score %.2f" % score, "100 game average %.2f" % np.mean(scores[-100:]))
    agent.memory.reset_episode()



env.close()

plt.plot(scores)
plt.show()

print("kraj")



     
# env_name = 'PandaReach-v3'

# env = gym.make(env_name)
# env_test = gym.make(env_name,render_mode = "human")
# max_action = env.action_space.high
# input_dims = env.observation_space.shape
# n_actions = env.action_space.shape[0]
# lr_actor = 0.001
# lr_critic = 0.001

# max_episodes = 10000

# agent = Agent(lr_actor,lr_critic,input_dims,n_actions,env,max_action,reward_scale=2)
# env.reset()
# #env_test.reset()

# def test():
    
#     done = False
#     trunc = False
#     state = env_test.reset()[0]
#     time_step = 0
#     score = 0
#     while not done and not trunc:
#         action = agent.choose_action(state)
#         next_state, reward, done, trunc, _ = env_test.step(action)
#         state = next_state
#         score += reward
#         time_step+=1
#         time.sleep(1/60)
    
#     return score
    

# def train():
#     scores = []
#     test_scores = []
#     for episode in trange(max_episodes):
#         state = env.reset()[0]
#         done = False
#         trunc = False
#         time_step = 0
        
#         score = 0
#         while not done and not trunc:

#             action = agent.choose_action(state)
#             next_state, reward, done, trunc, _ = env.step(action)

#             score += reward
#             agent.memory.push(state, action, reward, int(done), next_state, 1)

#             agent.learn()


#             state = next_state
            
#             time_step+=1
#         scores.append(score)
#         print("episode" , episode, "score %.2f" % score, "100 game average %.2f" % np.mean(scores[-100:]))
            
#         # if episode % 10 == 0:
#         #     test_score = test()
#         #     test_scores.append(test_score)
#         #     print("test number", int(episode/10), "score %.2f" % test_score, "100 test average %.2f" % np.mean(test_scores[-100:]))
    
#     # for i in range(10):
#     #     test()
#     #     time.sleep(1)


#     plt.plot(scores)
#     plt.show()

#     print("gotov")
            
# #train()