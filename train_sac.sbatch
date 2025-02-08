#!/bin/bash
#SBATCH --job-name=ta_train

#SBATCH --cpus-per-task=4
# max 24 per node

#SBATCH --partition=day
# choose out of day, week, month depending on job duration. Day holds for 24h

#SBATCH --mem-per-cpu=3G
# max 251GB per node

#SBATCH --gres=gpu:1
# how many gpus to use
# each node has 4 gpus

#SBATCH --time=23:59:59
# job length: the job will run either until completion or until this timer runs out; "hours:minutes:seconds", "days-hours", "days-hours:minutes"

#SBATCH --error=tcml_logs/job.%J.err
# %J is the job ID, errors will be written to this file

#SBATCH --output=tcml_logs/job.%J.out
# the output will be written in this file

#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=florian.sprick@uni-tuebingen.de

singularity run ~/container.sif python3 ./main.py algorithm=sac mode=train \
  algorithm.params.learning_rate=0.00073 \
  algorithm.params.gamma=0.98 \
  algorithm.params.buffer_size=300000 \
  algorithm.params.net_arch.pi=[400,300] \
  algorithm.params.net_arch.qf=[400,300] \
  algorithm.params.batch_size=256 \
  algorithm.params.activation_fn=ReLU \
  algorithm.params.tau=0.02 \
  algorithm.params.train_freq=[1,episode] \
  algorithm.params.learning_starts=10000 \
  algorithm.params.use_sde=true \
  algorithm.params.log_std_init=-3 \
  algorithm.params.sde_sample_freq=8 \
  mode.total_timesteps=5000000

