#!/bin/bash

#SBATCH --job-name="evaluate_models1"
#SBATCH --partition=Abaki              
#SBATCH --qos=abaki                       
#SBATCH --comment="shortcut learning"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.zhang@campus.lmu.de
#SBATCH --chdir=/home/z/zhangyon/shortcut_learning
#SBATCH --output=/home/z/zhangyon/shortcut_learning/output/qwen/6.2.zero_shot.start.tautologie2.slurm.%j.%N.out
#SBATCH --requeue

export HF_HOME="/home/z/zhangyon/BIG/hf_cache/hub"


python3 -u evaluate_models.py \
  --dataset e-ViL/data/esnlive_position_start_tautologie2.csv \
  --image_dir /home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images \
  --model qwen \
  --strategy zero_shot \
  --variation start \
  --output output/qwen/6.2.zero_shot.start.tautologie2.csv