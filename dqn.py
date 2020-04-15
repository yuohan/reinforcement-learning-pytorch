import gym
import math
import copy
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', 
    ('obs', 'action', 'reward', 'next_obs', 'done'))

class ReplayBuffer:

    def __init__(self, size):

        self.memory = []
        self.size = size

    def add(self, obs, action, reward, next_obs, done):

        if len(self.memory) > self.size:
            self.memory.pop(0)
        
        self.memory.append(Transition(obs, action, reward, next_obs, int(done)))

    def sample(self, batch_size):

        if len(self.memory) < batch_size:
            raise ValueError("Memory size is smaller than batch size")

        return Transition(*zip(*random.sample(self.memory, batch_size)))

class EpsilonGreedy:
    """
    Epsilon Greedy Policy
    """
    def __init__(self, agent, num_actions, init_epsilon, final_epsilon, epsilon_decay):

        self.agent = agent
        self.num_actions = num_actions
        
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

    def act(self, obs, t):

        epsilon = self.final_epsilon + (self.init_epsilon - self.final_epsilon) * math.exp(-1. * t / self.epsilon_decay)
        
        if random.random() > epsilon:
            with torch.no_grad():
                action = self.agent.get_q(torch.Tensor(obs).unsqueeze(0)).max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)

        return action

class DQN():
    """
    DQN (Deep Q Network) Agent Class
    """
    def __init__(self, device, obs_dim, num_actions, hidden_units, lr, gamma=0.99):

        self.device = device

        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

        self.target_net = copy.deepcopy(self.q_net)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma

    def train(self, transitions):

        obs = torch.tensor(transitions.obs, dtype=torch.float32).to(self.device)
        action = torch.tensor(transitions.action).to(self.device)
        reward = torch.tensor(transitions.reward).to(self.device)
        next_obs = torch.tensor(transitions.next_obs, dtype=torch.float32).to(self.device)
        done = torch.tensor(transitions.done).to(self.device)

        # Q(s_t, a)
        q = self.q_net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        # y_t = r_t                                         if episode terminate at step t+1
        #       r_t + gamma * max_{a'} Q_targ(s_{t+1}, a')  otherwise
        y = reward + (self.gamma * self.target_net(next_obs).max(-1)[0])*(1-done)
        # MSE
        loss = self.criterion(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_q(self, obs):
        with torch.no_grad():
            return self.q_net(obs)

    def update_target(self):

        for param, param_target in zip(self.q_net.parameters(), self.target_net.parameters()):
                param_target.data.copy_(param.data)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    gamma = 0.99
    init_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = 200
    seed = 123

    batch_size = 128
    memory_size = 500
    hidden_units = 128

    train_timesteps = 15000
    train_start_time = 128
    target_update_frequency = 100

    log_timesteps = 200

    env = gym.make('CartPole-v0')
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memory = ReplayBuffer(memory_size)

    agent = DQN(device, obs_dim, num_actions, hidden_units, lr, gamma)
    policy = EpsilonGreedy(agent, num_actions, init_epsilon, final_epsilon, epsilon_decay)

    # populate replay memory
    obs = env.reset()
    for t in range(train_start_time):

        # uniform random policy
        action = random.randrange(num_actions)
        next_obs, reward, done, _ = env.step(action)
        memory.add(obs, action, reward, next_obs, done)

        obs = next_obs

        if done:
            # start a new episode
            obs = env.reset()

    # for monitoring
    episode_reward = 0
    reward_list = []

    # train start
    obs = env.reset()
    for t in range(1, train_timesteps+1):

        # choose action
        action = policy.act(obs, t)
        next_obs, reward, done, _ = env.step(action)
        memory.add(obs, action, reward, next_obs, done)

        obs = next_obs
        
        # sample batch transitions from memory
        transitions = memory.sample(batch_size)
        # train
        loss = agent.train(transitions)

        # record reward
        episode_reward += reward

        if t % log_timesteps == 0:
            print(f't: {t:5}, reward: {sum(reward_list[-10:])/10:5}')

        # update target network at every C timesteps
        if t % target_update_frequency == 0:
            agent.update_target()

        if done:
            # start a new episode
            obs = env.reset()

            reward_list.append(episode_reward)
            episode_reward = 0

if __name__ == '__main__':
    main()