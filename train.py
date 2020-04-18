import gym
import yaml
import tqdm
import random
import argparse

import torch

from atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch
from dqn import DQN, EpsilonGreedy, ReplayBuffer

def get_env_type(env_id):
    return gym.envs.registry.env_specs[env_id].entry_point.split(':')[0].split('.')[-1]

def train(env_id, lr=1e-4, gamma=0.99,
         memory_size=1000, batch_size=32,
         train_timesteps=10000, train_start_time=1000, target_update_frequency=1000,
         init_epsilon=1, final_epsilon=0.1, epsilon_decay=300,
         model_path=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    LOG_PATH = f'logs/dqn_log_{env_id}.txt'

    if get_env_type(env_id) == 'atari':
        env = make_atari(env_id)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)

        model_type = 'conv'
    else:
        env = gym.make(env_id)

        model_type = 'linear'

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    memory = ReplayBuffer(memory_size)

    agent = DQN(obs_shape, num_actions, lr, gamma, device, model_type)
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
    for t in tqdm.tqdm(range(1, train_timesteps+1)):

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

            if model_path is not None:
                # save model
                info = {
                    'epoch': ep_num,
                    'timesteps': t,
                }
                agent.save(model_path, info)

            ep_num += 1
            ep_start_time = t+1
            reward_list.append(episode_reward)
            episode_reward = 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train RL Model')

    parser.add_argument('env', type=str,
                    help='Open AI gym env ID')
    parser.add_argument('--config',
                    help='path of configure yaml file')

    args = parser.parse_args()
    
    env_id = args.env
    config = args.config

    if config is not None:
        stream = open(args.config, 'r')
        kwargs = yaml.load(stream, Loader=yaml.FullLoader)
        stream.close()
    else:
        kwargs = {}

    train(env_id, **kwargs)