# Hockey RL Agent

## Quick Links 🔗
- [Hockey Environment Repository](https://github.com/martius-lab/laser-hockey-env)
- [Competition Server](http://comprl.cs.uni-tuebingen.de)
- [Client Code Repository](https://github.com/martius-lab/comprl-hockey-agent)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Competition Server](https://github.com/martius-lab/comprl/)

## Project Overview 🎯
This repository contains the implementation of a Reinforcement Learning agent for a simulated hockey game. The project is part of the RL course curriculum, focusing on developing and evaluating different RL algorithms.

### Environment
The project uses a custom hockey environment built on the Gymnasium API (formerly OpenAI Gym). The environment provides different training modes:
- `NORMAL`: Standard gameplay against another agent
- `TRAIN_SHOOTING`: Practice hitting a static puck into the goal
- `TRAIN_DEFENSE`: Practice defending against incoming shots

## Competition Setup 🏆

- Competition server is running at: http://comprl.cs.uni-tuebingen.de
- Weak and strong baseline agents are available for testing
- Use the provided client code at https://github.com/martius-lab/comprl-hockey-agent to connect your agent

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

### Using the TCML Cluster

Building the container:

```bash
singularity build --fakeroot /path/to/container.sif container.def
```

Running Scripts in the Container:

```bash
singularity run /path/to/container.sif python3 ./my_script.py
```

- Try out **Hockeyenv.ipynb** for environment exploration
