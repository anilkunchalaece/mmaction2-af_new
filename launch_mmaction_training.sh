#!/bin/sh

#SBATCH --job-name=MMACT_FR
#SBATCH --mem=40000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=mmaction_frames_training.log
#SBATCH --error=mmaction_frames_training_error.log
#SBATCH --partition=MEDIUM-G2

# source /home/ICTDOMAIN/d20125529/action_tracklet_parser/venv/bin/activate
. /home/ICTDOMAIN/d20125529/action_tracklet_parser/venv3_8/bin/activate
python --version
which python

# pip install -U openmim
# mim install mmengine
# mim install mmcv
# mim install mmdet
# mim install mmpose

# pip install -v -e .

# mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .

# python -u demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
#     tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
#     demo/demo.mp4 tools/data/kinetics/label_map_k400.txt

# python -u tools/train.py configs/action_tracklets/kth_tracklets/c3d_from_scratch_config.py 
python -u tools/train.py configs/action_tracklets/kth_frames/c3d_frames_from_scratch_config.py
