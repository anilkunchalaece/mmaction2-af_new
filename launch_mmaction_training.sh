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


###### C3D ##############

# Train tracklets from scratch
python -u tools/train.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_from_scratch_config.py

# Train frames from scratch
python -u tools/train.py configs/action_tracklets/c3d/c3d_frames_ntu_rgb_from_scratch_config.py

# fine tune the kintetics-400 model using tracklets
python -u tools/train.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_config.py

# fine tune the ntu_rgb frames model using tracklets
# python -u tools/train.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_ntu_rgb_frames_config.py

# Train from scratch on frames and test on tracklets
# python -u tools/test.py configs/action_tracklets/c3d/c3d_frames_ntu_rgb_from_scratch_test_tubelets_config.py \
#           work_dirs/c3d_frames_ntu_rgb_from_scratch_config/best_acc_top1_epoch_44.pth \
#           --dump train_frames_test_tubelets.pkl  

# Train from scratch on tubelets and test on tubelets
# python -u tools/test.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_from_scratch_test_tubelets_config.py \
#             work_dirs/c3d_tubelets_ntu_rgb_from_scratch_config/best_acc_top1_epoch_38.pth \
#             --dump train_tubelets_test_tubelets.pkl

# Transferlearning on Kinetics-400 and test on tubelets
# python -u tools/test.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_test_tubelet_config.py \
#           work_dirs/c3d_tubelets_ntu_rgb_transfer_learning_config/best_acc_top1_epoch_10.pth \
#           --dump kintetics_finetune_test_tubelets.pkl  

# Transferlearning on Kinetics-400 and test on tubelets
# python -u tools/test.py configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_ntu_rgb_frames_test_tubelet_config.py \
#             work_dirs/c3d_tubelets_ntu_rgb_transfer_learning_ntu_rgb_frames_config/best_acc_top1_epoch_40.pth \
#           --dump ntu_rgb_frames_finetune_test_tubelets.pkl  

###### I3D ##############
python -u tools/train.py configs/action_tracklets/i3d/i3d_frames_ntu_rgb_from_scratch_config.py

python -u tools/train.py configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config.py

python -u tools/train.py configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_config.py