# Hockey RL Agent

## Quick Links 🔗
- [Hockey Environment Repository](https://github.com/martius-lab/laser-hockey-env)
- [Competition Server Repository](https://github.com/martius-lab/comprl/)
- [Client Code Repository](https://github.com/martius-lab/comprl-hockey-agent)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Competition Server](http://comprl.cs.uni-tuebingen.de)


## Project Overview 🎯
This repository contains the implementation of a Reinforcement Learning agent for a simulated hockey game. The project is part of the RL course curriculum, focusing on developing and evaluating different RL algorithms.

### Environment
The project uses a custom hockey environment built on the Gymnasium API (formerly OpenAI Gym). The environment provides different training modes:
- `NORMAL`: Standard gameplay against another agent
- `TRAIN_SHOOTING`: Practice hitting a static puck into the goal
- `TRAIN_DEFENSE`: Practice defending against incoming shots

## Competition Setup 🏆
Team name:
PytorchPedalPushers-SAC

- Competition server is running at: http://comprl.cs.uni-tuebingen.de
- Weak and strong baseline agents are available for testing
- Use the provided client code at https://github.com/martius-lab/comprl-hockey-agent to connect your agent

## Getting Started 🚀

### Installation
```bash
pip install -r requirements.txt
```

- Try out **Hockeyenv.ipynb**  from the [env repository](https://github.com/martius-lab/hockey-env) for environment exploration

### Using the TCML Cluster

Set up a .env file and set TCML_USERNAME and TCML_PASSWORD. Afterwards, to connect to the TCML cluster using ssh run 

```bash
TCML.bat
```

Building the container:

```bash
singularity build --fakeroot /path/to/container.sif container.def
```

Running Scripts in the Container:

```bash
singularity run /path/to/container.sif python3 ./my_script.py
```

#### How to start a Job on the TCML Cluster

Use sbatch command to queue a job to the cluster:

```bash
sbatch train.sbatch
```

To see status of jobs use command *squeue*. To cancel a job use command *scancel <jobid>*

The output can be found in job.JOBNUMBER.out and the errors in job.JOBNUMBER.err which will be created in the same directory as the .sbatch file.


#### Run Model Training and Evaluation

Run the **main.py** and specify the corresponding configs that contain the desired method and mode (training, evaluation) using hydra. For example to train SAC with default parameters, type:

```bash
python3 ./main.py algorithm=sac mode=train 
```

and hydra will automatically compose configs/mode/train.yaml, configs/algorithm/sac.yaml and configs/config.yaml together.

To see the results in tensorboard:

```bash
tensorboard --logdir=./logs/
```

### Hyperparameter Tuning
- [Smooth Exploration for RL](https://arxiv.org/pdf/2005.05719) contains optimal HPs on Pybullet envs for PPO, SAC, TD3 with gSDE
- [rl-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py) contains code to conduct HP search with optuna

### References

- For the implementation of the ERE+PER ReplayBuffer confer [Link](https://github.com/BY571/Soft-Actor-Critic-and-Extensions/tree/master)
- For the implementation of the SAC-CEPO confer [Link](https://github.com/wcgcyx/SAC-CEPO/tree/master)

