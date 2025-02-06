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

#SBATCH --error=job.%J.err
# %J is the job ID, errors will be written to this file

#SBATCH --output=job.%J.out
# the output will be written in this file

#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=florian.sprick@uni-tuebingen.de

singularity exec --nv ~/container.sif python3 ./main.py --config ./configs/TD3/sb3/train.yaml --opponent weak
