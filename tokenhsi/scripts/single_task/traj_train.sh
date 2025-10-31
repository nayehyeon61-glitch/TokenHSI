#!/bin/bash
set -e

python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --num_envs 4096 \
    --headless

python ./tokenhsi/world_model/trainer.py \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --output_path output/single_task/world_model_traj.pth \
    --prediction_horizon 1 \
    --num_envs 512 \
    --num_batches 32 \
    --batch_size 2048 \
    --epochs 15 \
    --learning_rate 5e-4 \
    --hidden_dim 256 \
    --device auto \
    --seed 42

