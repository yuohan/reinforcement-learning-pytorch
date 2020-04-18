# reinforcement-learning-pytorch
Pytorch implementation of RL algorithms.

## Algorithms
DQN (Reference: [Human-level control through deep reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning))

## Usage
Train
```bash
# Train a model with CartPole env
python train.py PongNoFrameskip-v4 --config configs/pong_config.yaml

# Train a model with Pong env
python train.py PongNoFrameskip-v4 --config configs/pong_config.yaml
```
Play
```bash
# Play CartPole
python play.py CartPole-v0 5 models/cartpole_checkpoint.pth.tar --video

# Play Pong
python play.py PongNoFrameskip-v4 models/pong_checkpoint.pth.tar --video
```
## Result
All trained model, log files, plot images and result videos are available [here](https://drive.google.com/open?id=11Kk4TuVTLuxU68Ix5fUG72mFN9bpVtvd).