# Hockey RL Agent

## Quick Links 🔗
- [Hockey Environment Repository](https://github.com/martius-lab/laser-hockey-env)
- [Competition Server](http://comprl.cs.uni-tuebingen.de)
- [Client Code Repository](https://github.com/martius-lab/comprl-hockey-agent)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Project Overview 🎯
This repository contains the implementation of a Reinforcement Learning agent for a simulated hockey game. The project is part of the RL course curriculum, focusing on developing and evaluating different RL algorithms.

### Environment
The project uses a custom hockey environment built on the Gymnasium API (formerly OpenAI Gym). The environment provides different training modes:
- `NORMAL`: Standard gameplay against another agent
- `TRAIN_SHOOTING`: Practice hitting a static puck into the goal
- `TRAIN_DEFENSE`: Practice defending against incoming shots

## Getting Started 🚀

### Installation
```bash
pip install laser-hockey-env
```

```bash
import gymnasium as gym
import laser_hockey_env

env = gym.make('LaserHockey-v0', mode='NORMAL')
```

- Try out Hockeyenv.ipynb for environment exploration
