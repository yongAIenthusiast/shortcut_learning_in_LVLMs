#!/bin/bash

#SBATCH --job-name="clipscore"
#SBATCH --partition=Abaki
#SBATCH --qos=abaki                     
#SBATCH --comment="caption evaluation with clipscore"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.zhang@campus.lmu.de
#SBATCH --chdir=/home/z/zhangyon/shortcut_learning
#SBATCH --output=/home/z/zhangyon/shortcut_learning/captions_esnlive_variation_score.slurm.%j.%N.out
#SBATCH --requeue

export HF_HOME="/home/z/zhangyon/BIG/hf_cache/hub"


python3 -u clipscore.py