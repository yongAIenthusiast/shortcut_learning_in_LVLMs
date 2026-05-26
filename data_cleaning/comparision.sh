#!/bin/bash

#SBATCH --job-name="evaluate_models"
#SBATCH --partition=NvidiaAll                   
#SBATCH --comment="shortcut learning"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.zhang@campus.lmu.de
#SBATCH --chdir=/home/z/zhangyon/shortcut_learning
#SBATCH --output=/home/z/zhangyon/shortcut_learning/compare_models_testdata/comparision_final.slurm.%j.%N.out
#SBATCH --requeue


export HF_HOME="/home/z/zhangyon/BIG/hf_cache/hub"


python3 -u compare_models_3lc.py \
    --dataset e-ViL/data/esnlive.csv \
    --image_dir /home/z/zhangyon/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images \
    --result_csvs output/qwen/9.zero_shot.end.qwen.csv output/gemma/9.zero_shot.end.gemma.csv output/molmo/9.zero_shot.end.molmo.csv output/llavaov/9.zero_shot.end.llavaov.csv 