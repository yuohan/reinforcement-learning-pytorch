import gym
import math
import copy
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch

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
    def __init__(self, obs_shape, num_actions, lr=1e-3, gamma=0.99, device='cuda', model='atari'):

        # define q network
        if model == 'atari':

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

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 1e-4
    gamma = 0.99
    init_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = 30000
    seed = 123

    batch_size = 32
    memory_size = 100000

    train_timesteps = 1000000
    train_start_time = 10000
    target_update_frequency = 10000

    MODEL_PATH = 'models/dqn_pong_checkpoint.pth.tar'
    LOG_PATH = 'dqn_pong_log.txt'

    #env = gym.make('CartPole-v0')
    #env.seed(seed)
    env_id = 'PongNoFrameskip-v4'
    env    = make_atari(env_id)
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    memory = ReplayBuffer(memory_size)

    agent = DQN(obs_shape, num_actions, lr, gamma, device, 'atari')
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
    ep_num = 1
    ep_start_time = 1
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

        # update target network at every C timesteps
        if t % target_update_frequency == 0:
            agent.update_target()

        if done:
            # start a new episode
            obs = env.reset()

            # write log
            with open(LOG_PATH, 'a') as f:
                f.write(f'{ep_num}\t{episode_reward}\t{ep_start_time}\t{t}\n')

            # save model
            info = {
                'epoch': ep_num,
                'timesteps': t,
            }
            agent.save(MODEL_PATH, info)

            ep_num += 1
            ep_start_time = t+1
            reward_list.append(episode_reward)
            episode_reward = 0

if __name__ == '__main__':
    main()