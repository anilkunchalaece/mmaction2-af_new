# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# from mmaction.evaluation.functional import (get_weighted_score,
#                                             mean_class_accuracy,
#                                             top_k_accuracy)



results_compiled = {
    "configs/action_tracklets/c3d/c3d_frames_ntu_rgb_from_scratch_test_frames_config.py" :
      "cvip_results/c3d_train_frames_test_frames.pkl",

    "configs/action_tracklets/c3d/c3d_frames_ntu_rgb_from_scratch_test_tubelets_config.py" : 
    "cvip_results/c3d_train_frames_test_tubelets.pkl",

    "configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_from_scratch_test_tubelets_config.py" : 
    "cvip_results/c3d_train_tubelets_test_tubelets.pkl",

    "configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_ntu_rgb_frames_50_perc_data_config.py" :
      "cvip_results/c3d_ntu_rgb_frames_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/c3d/c3d_tubelets_ntu_rgb_transfer_learning_50_perc_data_config.py" :
      "cvip_results/c3d_kintetics_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/i3d/i3d_frames_ntu_rgb_from_scratch_test_frames_config.py" :
    "cvip_results/i3d_train_frames_test_frames.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_test_tubelets_config.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config.py" : 
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_frames_50_perc_data_config.py" :
    "cvip_results/i3d_ntu_rgb_frames_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/slowfast/slowfast_frames_ntu_rgb_from_scratch_test_frames_config.py":
    "cvip_results/slowfast_train_frames_test_frames.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_test_tubelets_config.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config.py":
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_ntu_rgb_frames_50_perc_data_config.py" :
    "cvip_results/slowfast_ntu_rgb_frames_finetune_test_tubelets_50_perc.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_config_16x16.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets_16x16.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_config_32x32.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets_32x32.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_config_64x64.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets_64x64.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_config_128x128.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets_128x128.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_from_scratch_config_256x256.py" :
    "cvip_results/slowfast_train_tubelets_test_tubelets_256x256.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config_16x16.py" :
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_16x16.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config_32x32.py" :
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_32x32.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config_64x64.py" :
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_64x64.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config_128x128.py" :
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_128x128.pkl",

    "configs/action_tracklets/slowfast/slowfast_tubelets_ntu_rgb_transfer_learning_50_perc_data_config_256x256.py" :
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_256x256.pkl"
}




def main():
    
    results = {}

    for config, pkl_results in results_compiled.items() :

        # # load config
        cfg = Config.fromfile(config)

        init_default_scope(cfg.get('default_scope', 'mmaction'))

        data_samples = mmengine.load(pkl_results)
        
        test_evaluator = dict(
                        type='AccMetric',
                        metric_options=dict(
                            top_k_accuracy=dict(topk=(1,3))
                            ),
                        metric_list= ('top_k_accuracy', 'mean_class_accuracy')
                            )
        evaluator = Evaluator(test_evaluator)
        eval_results = evaluator.offline_evaluate(data_samples)
        print(os.path.basename(pkl_results),eval_results)

        results[os.path.basename(pkl_results)] = eval_results
    
    f_to_save = "cvip_results.json"
    with open(f_to_save,'w') as fw :
        json.dump(results,fw)
    
    print(F"Results are saved to {f_to_save}")

def plot_graphs(f_name) :
    with open(f_name) as fd :
        data = json.load(fd)

    tubelets_from_scratch = {}
    kinetics_pretrain = {}

    for k,v in data.items() :
        if k.find("slowfast_train_tubelets_test_tubelets_") != -1 :
            tubelets_from_scratch[k.split("_")[-1].replace(".pkl","").split("x")[0]] = v
        elif k.find("slowfast_kintetics_finetune_test_tubelets_50_perc_") != -1 :
            kinetics_pretrain[k.split("_")[-1].replace(".pkl","").split("x")[0]] = v
    
    # print(tubelets_from_scratch)
    # print(kinetics_pretrain)
    # matplotlib.style.use('ggplot')
    # matplotlib.rcParams['lines.linestyle'] = '-*'
    # matplotlib.rcParams['lines.linewidth'] = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'yellow'
    # matplotlib.rcParams['axes.facecolor'] = 'lightblue'
    # matplotlib.rcParams['font.family'] = 'serif'
    # matplotlib.rcParams['font.size'] = 8
    # matplotlib.rcParams['font.weight'] = 'bold'
    # matplotlib.rcParams['text.color'] = 'red'

    fig, ax = plt.subplots()  # Adjust figure size as per your paper requirements
    df = pd.DataFrame.from_dict(tubelets_from_scratch,orient='index')
    # print(df.columns)
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    df = df.drop(["16"])
    df[["top1","top3"]].plot(kind='line', title="Tubelets from scratch",ax=ax, linewidth=2.5, marker="*", markersize=10)
    # df.plot(kind='scatter', )
    # ax.grid(True, linestyle="--", alpha=0.5)
    # ax.set_ylim([0.7,1])
    # plt.xlabel("Image size")

    # fig2, ax2 = plt.subplots()  # Adjust figure size as per your paper requirements
    df = pd.DataFrame.from_dict(kinetics_pretrain,orient='index')
    df = df.drop(["16"])
    # print(df.columns)
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    df[["top1","top3"]].plot(kind='line', title="Kinetics pretrained",ax=ax, linewidth=2.5, marker="o", markersize=10)
    # df.plot(kind='scatter', )
    ax.grid(True, linestyle="--", alpha=0.5)
    # ax.set_ylim([0.6,1])
    plt.xlabel("Image size")

    plt.show()
    


if __name__ == '__main__':
    # main()
    plot_graphs("cvip_results.json")
