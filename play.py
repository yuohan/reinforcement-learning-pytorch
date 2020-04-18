import gym
import argparse
from itertools import count

import torch
import skvideo.io

from atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch
from dqn import DQN, Greedy

def get_env_type(env_id):
    return gym.envs.registry.env_specs[env_id].entry_point.split(':')[0].split('.')[-1]

def play(env_id, model_path, max_ep, video):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if get_env_type(env_id) == 'atari':
        env = make_atari(env_id)
        env = wrap_deepmind(env, False, False, False, False)
        env = wrap_pytorch(env)

        model_type = 'conv'
    else:
        env = gym.make(env_id)

        model_type = 'linear'

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQN(obs_shape, num_actions, device=device, model=model_type)
    agent.load(model_path)

    policy = Greedy(agent)

    ep = 1
    episode_reward = 0

    obs = env.reset()
    screen = env.render(mode='rgb_array')
    if video:
        writer = skvideo.io.FFmpegWriter(f'videos/{env_id}-ep-{ep}.mp4')

    for t in count():

        action = policy.act(obs, t)
        next_obs, reward, done, _ = env.step(action)

        episode_reward += reward

        screen = env.render(mode='rgb_array')
        if video:
            writer.writeFrame(screen)

        obs = next_obs

        if done:

            print(f'ep: {ep:4} reward: {episode_reward}')

            if ep >= max_ep:
                break

            ep += 1
            episode_reward = 0
            ebs = env.reset()

            if video:
                writer.close()
                writer = skvideo.io.FFmpegWriter(f'videos/{env_id}-ep-{ep}.mp4')

    if video:
        writer.close()
    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Play RL Model')

    parser.add_argument('env', type=str,
                    help='Open AI gym env ID')
    parser.add_argument('ep', type=int,
                    help='number of episodes to play')
    parser.add_argument('model_path', type=str,
                    help='path of model to load')
    parser.add_argument('--video', action='store_true',
                    help='save result video')

    args = parser.parse_args()
    
    env_id = args.env
    max_ep = args.ep
    model_path = args.model_path
    video = args.video != None

    play(env_id, model_path, max_ep, video)