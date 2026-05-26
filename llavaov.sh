#!/bin/bash

#SBATCH --job-name="evaluate_models1"
#SBATCH --partition=Abaki              
#SBATCH --qos=abaki                       
#SBATCH --comment="shortcut learning"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.zhang@campus.lmu.de
#SBATCH --chdir=/home/z/zhangyon/shortcut_learning
#SBATCH --output=/home/z/zhangyon/shortcut_learning/output/llavaov/0.1.zero_shot.original.slurm.%j.%N.out
#SBATCH --requeue

export HF_HOME="/home/z/zhangyon/BIG/hf_cache/hub"


python3 -u evaluate_models.py \
  --dataset e-ViL/data/esnlive.csv \
  --image_dir /home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images \
  --model llava_ov \
  --strategy zero_shot \
  --variation original \
  --output output/llavaov/0.1.zero_shot.original.csv \