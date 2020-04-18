import math
import copy
import random
from collections import namedtuple

import numpy as np
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

class Greedy:
    """
    Greedy Policy
    """
    def __init__(self, agent):

        self.agent = agent

    def act(self, obs, t):
        with torch.no_grad():
            return self.agent.get_q(torch.Tensor(obs).unsqueeze(0)).max(1)[1].item()

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
    def __init__(self, obs_shape, num_actions, lr=1e-3, gamma=0.99, device='cuda', model='conv'):

        # define q network
        if model == 'conv':

            # channel first
            input_channel, height, width = obs_shape

            kernel_size = [8, 4, 3]
            stride = [4, 2, 1]
            channels = [32, 64, 64]

            def conv2d_output_size(input_size, kernel_size, stride):
                return (input_size - (kernel_size - 1) - 1) // stride  + 1
            
            for k, s in zip(kernel_size, stride):
                height = conv2d_output_size(height, k, s)
                width = conv2d_output_size(width, k, s)

            linear_input_size = channels[2] * height * width
            linear_units = 512

            model = nn.Sequential(
                nn.Conv2d(input_channel, channels[0], kernel_size[0], stride[0]),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(),
                nn.Conv2d(channels[0], channels[1], kernel_size[1], stride[1]),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(),
                nn.Conv2d(channels[1], channels[2], kernel_size[2], stride[2]),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(linear_input_size, linear_units),
                nn.ReLU(),
                nn.Linear(linear_units, num_actions)
            )
        else:

            linear_units = 128

            model = nn.Sequential(
            nn.Linear(np.prod(obs_shape), linear_units),
            nn.ReLU(),
            nn.Linear(linear_units, linear_units),
            nn.ReLU(),
            nn.Linear(linear_units, num_actions)
        )

        self.gamma = gamma
        self.device = device
        self.q_net = model.to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

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
            return self.q_net(obs.to(self.device))

    def update_target(self):

        for param, param_target in zip(self.q_net.parameters(), self.target_net.parameters()):
            param_target.data.copy_(param.data)

    def save(self, path, info=None):

        state = {
            'state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        if info != None:
            state = {**info, **state}
        torch.save(state, path)

    def load(self, path):

        state = torch.load(path)

        self.q_net.load_state_dict(state['state_dict'])
        self.target_net.load_state_dict(state['target_state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        return state