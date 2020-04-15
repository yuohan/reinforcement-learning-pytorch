import gym
import math
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:

    def __init__(self, size):

        self.memory = []
        self.size = size

    def add(self, obs, action, reward, next_obs, done):

        if len(self.memory) > self.size:
            self.memory.pop(0)
        
        self.memory.append((obs, action, reward, next_obs, int(done)))

    def sample(self, batch_size):

        if len(self.memory) < batch_size:
            raise ValueError("Memory size is smaller than batch size")

        return tuple(zip(*random.sample(self.memory, batch_size)))

class DQN(nn.Module):

    def __init__(self, obs_dim, num_actions, hidden_units):
        super().__init__()
        
        self.hidden1 = nn.Linear(obs_dim, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, num_actions)

    def forward(self, x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)

        return x

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    gamma = 0.99
    init_explore = 1
    final_explore = 0.01
    explore_decay = 200
    seed = 123

    batch_size = 128
    memory_size = 500

    train_timesteps = 15000
    train_start_time = 128
    target_update_frequency = 100

    log_timesteps = 200

    env = gym.make('CartPole-v0')
    env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    hidden_units = 128

    memory = ReplayBuffer(memory_size)
    main_net = DQN(obs_dim, num_actions, hidden_units)
    target_net = copy.deepcopy(main_net)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(main_net.parameters(), lr=lr)

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
        epsilon = final_explore + (init_explore - final_explore) * math.exp(-1. * t / explore_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                action = main_net(torch.Tensor(obs).unsqueeze(0)).max(1)[1].item()
        else:
            action = random.randrange(num_actions)

        next_obs, reward, done, _ = env.step(action)
        memory.add(obs, action, reward, next_obs, done)

        obs = next_obs
        
        # sample batch transitions from memory
        transitions = memory.sample(batch_size)
        
        batch_obs = torch.tensor(transitions[0], dtype=torch.float32).to(device)
        batch_a = torch.tensor(transitions[1]).to(device)
        batch_r = torch.tensor(transitions[2]).to(device)
        batch_next_obs = torch.tensor(transitions[3], dtype=torch.float32).to(device)
        batch_d = torch.tensor(transitions[4]).to(device)

        # Q(s_t, a)
        q = main_net(batch_obs).gather(1, batch_a.unsqueeze(1)).squeeze(1)
        # y_j = r_j                                         if episode terminate at step j+1
        #       r_j + gamma * max_{a'} Q_targ(s_{t+1}, a')  otherwise
        y = batch_r + (gamma * target_net(batch_next_obs).max(-1)[0])*(1-batch_d)
        # MSE
        loss = criterion(q, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record reward
        episode_reward += reward

        if t % log_timesteps == 0:
            print(f't: {t:5}, reward: {sum(reward_list[-10:])/10:5}')

        # update target network at every C timesteps
        if t % target_update_frequency == 0:
            for param_main, param_target in zip(main_net.parameters(), target_net.parameters()):
                param_target.data.copy_(param_main.data)

        if done:
            # start a new episode
            obs = env.reset()

            reward_list.append(episode_reward)
            episode_reward = 0

if __name__ == '__main__':
    main()