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
import pickle
from sklearn.metrics import accuracy_score
import numpy as np

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
    "cvip_results/slowfast_kintetics_finetune_test_tubelets_50_perc_256x256.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config_16x16.py" :
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc_16x16.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config_32x32.py" :
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc_32x32.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config_64x64.py" :
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc_64x64.pkl",
    
    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config_128x128.py" :
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc_128x128.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_transfer_learning_kintetics_50_perc_data_config_256x256.py" :
    "cvip_results/i3d_kintetics_finetune_test_tubelets_50_perc_256x256.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config_16x16.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets_16x16.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config_32x32.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets_32x32.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config_64x64.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets_64x64.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config_128x128.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets_128x128.pkl",

    "configs/action_tracklets/i3d/i3d_tubelets_ntu_rgb_from_scratch_config_256x256.py" :
    "cvip_results/i3d_train_tubelets_test_tubelets_256x256.pkl",

}



tubelet_results_compiled = {
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_config.py" :
      "tubelet_dataset_results/sf_tubelet.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_config_8x8.py" : 
        "tubelet_dataset_results/sf_tubelet_8x8.pkl",

    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_config_32x32.py" : 
        "tubelet_dataset_results/sf_tubelet_32x32.pkl",

    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_config_64x64.py" :
         "tubelet_dataset_results/sf_tubelet_64x64.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_kinetic_pretrained_config_8x8.py" :
         "tubelet_dataset_results/sf_tubelet_pretrained_8x8.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_kinetic_pretrained_config_16x16.py" :
         "tubelet_dataset_results/sf_tubelet_pretrained_16x16.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_kinetic_pretrained_config_32x32.py" :
         "tubelet_dataset_results/sf_tubelet_pretrained_32x32.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_focal_loss_kinetic_pretrained_config_64x64.py" :
         "tubelet_dataset_results/sf_tubelet_pretrained_64x64.pkl"
}

TUBELET_LABAL_MATCHER = [
    "WALKING", 
    "RUNNING",
    "SITTING",
    "STANDING",
    "GESTURING",
    "CARRYING",
    "USING_PHONE"
]

def calculate_class_wise_accuracy(predictions_file, num_classes):
    # Load the pkl file with predictions
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)

    # Initialize lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Extract true and predicted labels from the predictions
    for prediction in predictions:
        true_label = prediction['gt_labels']['item'].item()
        predicted_label = prediction['pred_labels']['item'].item()
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(true_labels, predicted_labels)

    # Initialize dictionary to store class-wise accuracy
    class_accuracy = {}

    # Calculate class-wise accuracy
    for class_id in range(num_classes):
        class_indices = [i for i, true_label in enumerate(true_labels) if true_label == class_id]
        class_true_labels = [true_labels[i] for i in class_indices]
        class_predicted_labels = [predicted_labels[i] for i in class_indices]

        if len(class_indices) > 0:
            class_accuracy[TUBELET_LABAL_MATCHER[class_id]] = accuracy_score(class_true_labels, class_predicted_labels)
        else:
            class_accuracy[TUBELET_LABAL_MATCHER[class_id]] = 0.0

    return overall_accuracy, class_accuracy


def main():
    
    results = {}

    for config, pkl_results in tubelet_results_compiled.items() :
        results[os.path.basename(pkl_results)] = {
            "average" : {},
            "class_wise" : {}
        }
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

        results[os.path.basename(pkl_results)]["average"] = eval_results
        no_of_classes = 7
        cls_wise_acc = calculate_class_wise_accuracy(pkl_results,no_of_classes)
        results[os.path.basename(pkl_results)]["class_wise"] = cls_wise_acc
        print(cls_wise_acc)
        print("### \n")
        # return
    f_to_save = "tubelet_results.json"
    with open(f_to_save,'w') as fw :
        json.dump(results,fw)
    
    print(F"Results are saved to {f_to_save}")

def plot_graphs(f_name) :
    font = {
        # 'weight' : 'bold',
        'size'   : 17}

    matplotlib.rc('font', **font)
    with open(f_name) as fd :
        data = json.load(fd)

    slowfast_tubelets_from_scratch = {}
    slowfast_kinetics_pretrain = {}
    i3d_tubelets_from_scratch = {}
    i3d_kinetics_pretrain = {}

    for k,v in data.items() :
        # print(k)
        if k.find("slowfast_train_tubelets_test_tubelets_") != -1 :
            slowfast_tubelets_from_scratch[int(k.split("_")[-1].replace(".pkl","").split("x")[0])] = v
        elif k.find("slowfast_kintetics_finetune_test_tubelets_50_perc_") != -1 :
            slowfast_kinetics_pretrain[int(k.split("_")[-1].replace(".pkl","").split("x")[0])] = v
        elif k.find("i3d_train_tubelets_test_tubelets_") != -1 :
            i3d_tubelets_from_scratch[int(k.split("_")[-1].replace(".pkl","").split("x")[0])] = v
        elif k.find("i3d_kintetics_finetune_test_tubelets_50_perc_") != -1 :
            # print(k)
            i3d_kinetics_pretrain[int(k.split("_")[-1].replace(".pkl","").split("x")[0])] = v    


    fig, ax = plt.subplots(num="SlowFast")  # Adjust figure size as per your paper requirements

    df = pd.DataFrame.from_dict(slowfast_tubelets_from_scratch,orient='index')
    df = df.sort_index()
    print(df)
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    # df = df.drop(["16"])
    df[["top1"]].plot(kind='line',ax=ax, linewidth=3, marker="*", markersize=8,linestyle="--", color='green')
    df[["top3"]].plot(kind='line',ax=ax, linewidth=3, marker="o", markersize=8, linestyle="-.", color='green')

    df = pd.DataFrame.from_dict(slowfast_kinetics_pretrain,orient='index')
    # df = df.drop(["16"])
    df = df.sort_index()
    print(df)
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    df[["top1"]].plot(kind='line',ax=ax, linewidth=2.25, marker="*", markersize=8,linestyle="--", color='orange')
    df[["top3"]].plot(kind='line',ax=ax, linewidth=2.25, marker="o", markersize=8, linestyle="-.", color='orange')
    # df.plot(kind='scatter', )
    ax.grid(True, linestyle="--", alpha=0.5)
    # ax.set_ylim([0.6,1])
    plt.xlabel("Image size")
    plt.legend(["tubelets-top1","tubelets-top3","pt-large-top1","pt-large-top3"])
    # plt.title("SlowFast")


    fig, ax = plt.subplots(num="I3D")  # Adjust figure size as per your paper requirements
    df = pd.DataFrame.from_dict(i3d_tubelets_from_scratch,orient='index')
    df = df.sort_index()
    print(df)
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    
    # df = df.drop(["16"])
    df[["top1"]].plot(kind='line',ax=ax, linewidth=2.25, marker="*", markersize=8,linestyle="--", color='green')
    df[["top3"]].plot(kind='line',ax=ax, linewidth=2.25, marker="o", markersize=8, linestyle="-.", color='green')

    df = pd.DataFrame.from_dict(i3d_kinetics_pretrain,orient='index')
    df = df.sort_index()
    print(df)
    # df = df.drop(["16"])
    df = df.rename(columns={"acc/top1": "top1", "acc/top3": "top3", "acc/mean1":"mAcc"})
    df[["top1"]].plot(kind='line',ax=ax, linewidth=2.25, marker="*", markersize=8,linestyle="--", color='orange')
    df[["top3"]].plot(kind='line',ax=ax, linewidth=2.25, marker="o", markersize=8, linestyle="-.", color='orange')
    # df.plot(kind='scatter', )
    ax.grid(True, linestyle="--", alpha=0.5)
    # ax.set_ylim([0.6,1])
    plt.xlabel("Image size")
    plt.legend(["tubelets-top1","tubelets-top3","pt-large-top1","pt-large-top3"])

    plt.show()
    


if __name__ == '__main__':
    main()
    # plot_graphs("cvip_results.json")
