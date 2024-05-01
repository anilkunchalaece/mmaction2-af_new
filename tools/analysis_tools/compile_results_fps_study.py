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
    "configs/fps_vs_size/c3d/c3d_mcad_dataset_config.py" :
    "fps_study_results/mcad/fps_vs_size_test_output.pkl",
    "configs/fps_vs_size/slowfast/slowfast_mcad_dataset_config.py" :
    "fps_study_results/mcad/fps_vs_size_test_output_slowfast.pkl",
    "configs/fps_vs_size/videoswin/swin_mcad_dataset.py" : 
    "fps_study_results/mcad/fps_vs_size_test_output_swin.pkl",
    
    "configs/fps_vs_size/c3d/c3d_mmact_dataset_config.py" :
        "fps_study_results/mmact/fps_vs_size_test_output_mmact.pkl",
    
    "configs/fps_vs_size/slowfast/slowfast_mmact_dataset_config.py" :
        "fps_study_results/mmact/fps_vs_size_test_output_slowfast_mmact.pkl",
        
    "configs/fps_vs_size/videoswin/swin_mmact_dataset.py" :
        "fps_study_results/mmact/fps_vs_size_test_output_mmact_swin.pkl"  
}

for idx in [20,15,10,5,3,1] :
    results_compiled[F"configs/fps_vs_size/c3d/c3d_mcad_dataset_config_test_{idx}fps.py"] = F"fps_study_results/mcad/fps_vs_size_test_output_{idx}fps.pkl"
    results_compiled[F"configs/fps_vs_size/slowfast/slowfast_mcad_dataset_config_{idx}fps.py"] = F"fps_study_results/mcad/fps_vs_size_test_output_{idx}fps_slowfast.pkl"
    results_compiled[F"configs/fps_vs_size/videoswin/swin_mcad_dataset_test_{idx}fps.py"] = F"fps_study_results/mcad/fps_vs_size_test_output_{idx}fps_swin.pkl"

    results_compiled[F"configs/fps_vs_size/c3d/c3d_mmact_dataset_config_test_{idx}fps.py"] = F"fps_study_results/mmact/fps_vs_size_test_output_mmact_{idx}fps.pkl"
    results_compiled[F"configs/fps_vs_size/slowfast/slowfast_mmact_dataset_config_{idx}fps.py"] = F"fps_study_results/mmact/fps_vs_size_test_output_{idx}fps_slowfast_mmact.pkl"
    results_compiled[F"configs/fps_vs_size/videoswin/swin_mmact_dataset_test_{idx}fps.py"] = F"fps_study_results/mmact/fps_vs_size_test_output_mmact_{idx}fps_swin.pkl"
results_compiled = dict(sorted(results_compiled.items()))


MCAD_LABEL_MATCHER = ["Point", "Wave", "Jump", "Crouch", "Sneeze", "SitDown", "StandUp", 
                      "Walk", "PersonRun", "CellToEar", "UseCellPhone", "DrinkingWater", 
                      "TakePicture","ObjectGet","ObjectPut","ObjectLeft","ObjectCarry","ObjectThrow"]


MMACT_LABEL_MATCHER = ['carrying', 'carrying_heavy', 'carrying_light','checking_time', 
                     'closing', 'crouching', 'entering', 'exiting', 'fall', 'jumping', 
                     'kicking', 'loitering', 'looking_around', 'opening', 'picking_up', 
                     'pointing', 'pulling', 'pushing', 'running', 'setting_down', 'standing', 
                     'talking', 'talking_on_phone', 'throwing', 'transferring_object', 'using_phone',
                     'walking', 'waving_hand', 'drinking', 'pocket_in', 'pocket_out', 'sitting',
                     'sitting_down', 'standing_up', 'talking_on_phone_desk', 'using_pc',
                     'using_phone_desk']

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
    
    if 'mmact' in [x.split(".")[0] for x in predictions_file.split("_")]:
        label_matcher = MMACT_LABEL_MATCHER
    else :
        label_matcher = MCAD_LABEL_MATCHER
    
    # label_matcher = MMACT_LABEL_MATCHER

    # Calculate class-wise accuracy
    for class_id in range(num_classes):
        class_indices = [i for i, true_label in enumerate(true_labels) if true_label == class_id]
        class_true_labels = [true_labels[i] for i in class_indices]
        class_predicted_labels = [predicted_labels[i] for i in class_indices]

        if len(class_indices) > 0:
            class_accuracy[label_matcher[class_id]] = accuracy_score(class_true_labels, class_predicted_labels)
        else:
            class_accuracy[label_matcher[class_id]] = 0.0

    return overall_accuracy, class_accuracy


def main():
    
    results = {}

    for config, pkl_results in results_compiled.items() :
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
                            top_k_accuracy=dict(topk=(1,5))
                            ),
                        metric_list= ('top_k_accuracy', 'mean_class_accuracy')
                            )
        evaluator = Evaluator(test_evaluator)
        eval_results = evaluator.offline_evaluate(data_samples)
        print(os.path.basename(pkl_results),eval_results)

        results[os.path.basename(pkl_results)]["average"] = eval_results
        # DONT NEED CLASS WISE ACCURACY NOW
        if 'mmact' in [x.split(".")[0] for x in pkl_results.split("_")]:
            no_of_classes = 37
        else:
            no_of_classes = 18
        cls_wise_acc = calculate_class_wise_accuracy(pkl_results,no_of_classes)
        results[os.path.basename(pkl_results)]["class_wise"] = cls_wise_acc
        print(cls_wise_acc)
        print("### \n")
        # return
    f_to_save = "fps_study_results.json"
    with open(f_to_save,'w') as fw :
        json.dump(results,fw)
    
    print(F"Results are saved to {f_to_save}")

def plot_graphs(f_name) :
    # font = {
        # 'weight' : 'bold',
        # 'size'   : 17}
    
    # slowfast_org = "fps_vs_size_test_output_swin.pkl"
    # slowfast_10fps = "fps_vs_size_test_output_10fps_swin.pkl"
    # slowfast_5fps = "fps_vs_size_test_output_5fps_swin.pkl"
    # slowfast_3fps = "fps_vs_size_test_output_3fps_swin.pkl"
    # slowfast_1fps = "fps_vs_size_test_output_1fps_swin.pkl"
    
    
    slowfast_org = "fps_vs_size_test_output_mmact_swin.pkl"
    slowfast_20fps = "fps_vs_size_test_output_mmact_20fps_swin.pkl"
    slowfast_15fps = "fps_vs_size_test_output_mmact_15fps_swin.pkl"
    slowfast_10fps = "fps_vs_size_test_output_mmact_10fps_swin.pkl"
    slowfast_5fps = "fps_vs_size_test_output_mmact_5fps_swin.pkl"
    slowfast_3fps = "fps_vs_size_test_output_mmact_3fps_swin.pkl"
    slowfast_1fps = "fps_vs_size_test_output_mmact_1fps_swin.pkl"

    out = dict()

    # matplotlib.rc('font', **font)
    with open(f_name) as fd :
        data = json.load(fd)
        
    out["Org FPS"] = data[slowfast_org]['class_wise'][1]
    # out["20 FPS"] = data[slowfast_20fps]['class_wise'][1]
    # out["15 FPS"] = data[slowfast_15fps]['class_wise'][1]
    out["10 FPS"] = data[slowfast_10fps]['class_wise'][1]
    # out["5 FPS"] = data[slowfast_5fps]['class_wise'][1]
    # out["3 FPS"] = data[slowfast_3fps]['class_wise'][1]
    # out["1 FPS"] = data[slowfast_1fps]['class_wise'][1]

    df = pd.DataFrame.from_dict(out)
    df=df.drop(['talking_on_phone_desk','using_phone_desk', 'carrying_heavy', 'carrying_light'])
    print(df)
    # df.plot.bar(edgecolor='white', linewidth=1)
    df.plot.line(linewidth=2, marker='*', linestyle='--')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(ticks=range(len(df.index)), labels=df.index, rotation=90, ha="right")
    # plt.xticks(rotation=90)
    ax = plt.gca()
    # ax.set_ylim([0.4, 1])
    plt.tight_layout()
    plt.show()
    
def plot_perc_change():
    font = {
        'weight' : 'normal',
        'size'   : 15}    
    matplotlib.rc('font', **font)
    with open("fps_study_results.json") as fd :
        data = json.load(fd)
    
    mcad_models = {
        "C3D" : {
            "Org FPS" : "fps_vs_size_test_output.pkl",
            # "20 FPS" : "fps_vs_size_test_output_20fps.pkl",
            # "15 FPS" : "fps_vs_size_test_output_15fps.pkl",
            "10 FPS" : "fps_vs_size_test_output_10fps.pkl",
            "5 FPS" : "fps_vs_size_test_output_5fps.pkl",
            "3 FPS" : "fps_vs_size_test_output_3fps.pkl",
            "1 FPS" : "fps_vs_size_test_output_1fps.pkl"
            
            # "Org FPS" : "fps_vs_size_test_output_mmact.pkl",
            # "20 FPS" : "fps_vs_size_test_output_mmact_20fps.pkl",
            # "15 FPS" : "fps_vs_size_test_output_mmact_15fps.pkl",
            # "10 FPS" : "fps_vs_size_test_output_mmact_10fps.pkl",
            # "5 FPS" : "fps_vs_size_test_output_mmact_5fps.pkl",
            # "3 FPS" : "fps_vs_size_test_output_mmact_3fps.pkl",
            # "1 FPS" : "fps_vs_size_test_output_mmact_1fps.pkl"
        },
        "SlowFast" : {
            "Org FPS" : "fps_vs_size_test_output_slowfast.pkl",
            # "20 FPS" : "fps_vs_size_test_output_20fps_slowfast.pkl",
            # "15 FPS" : "fps_vs_size_test_output_15fps_slowfast.pkl",
            "10 FPS" : "fps_vs_size_test_output_10fps_slowfast.pkl",
            "5 FPS" : "fps_vs_size_test_output_5fps_slowfast.pkl",
            "3 FPS" : "fps_vs_size_test_output_3fps_slowfast.pkl",
            "1 FPS" : "fps_vs_size_test_output_1fps_slowfast.pkl"

            # "Org FPS" : "fps_vs_size_test_output_slowfast_mmact.pkl",
            # "20 FPS" : "fps_vs_size_test_output_20fps_slowfast_mmact.pkl",
            # "15 FPS" : "fps_vs_size_test_output_15fps_slowfast_mmact.pkl",
            # "10 FPS" : "fps_vs_size_test_output_10fps_slowfast_mmact.pkl",
            # "5 FPS" : "fps_vs_size_test_output_5fps_slowfast_mmact.pkl",
            # "3 FPS" : "fps_vs_size_test_output_3fps_slowfast_mmact.pkl",
            # "1 FPS" : "fps_vs_size_test_output_1fps_slowfast_mmact.pkl"              
        },
        "VideoSwin" : {
            "Org FPS" : "fps_vs_size_test_output_swin.pkl",
            # "20 FPS" : "fps_vs_size_test_output_20fps_swin.pkl",
            # "15 FPS" : "fps_vs_size_test_output_15fps_swin.pkl",
            "10 FPS" : "fps_vs_size_test_output_10fps_swin.pkl",
            "5 FPS" : "fps_vs_size_test_output_5fps_swin.pkl",
            "3 FPS" : "fps_vs_size_test_output_3fps_swin.pkl",
            "1 FPS" : "fps_vs_size_test_output_1fps_swin.pkl"

            # "Org FPS" : "fps_vs_size_test_output_mmact_swin.pkl",
            # "20 FPS" : "fps_vs_size_test_output_mmact_20fps_swin.pkl",
            # "15 FPS" : "fps_vs_size_test_output_mmact_15fps_swin.pkl",
            # "10 FPS" : "fps_vs_size_test_output_mmact_10fps_swin.pkl",
            # "5 FPS" : "fps_vs_size_test_output_mmact_5fps_swin.pkl",
            # "3 FPS" : "fps_vs_size_test_output_mmact_3fps_swin.pkl",
            # "1 FPS" : "fps_vs_size_test_output_mmact_1fps_swin.pkl"
        },
        
    }
    
    mcad_results = {
        'C3D' : {},
        'SlowFast' : {},
        'VideoSwin' : {}
    }
    
    for m in mcad_models :
        for fps in mcad_models[m] :
            mcad_results[m][fps] = data[mcad_models[m][fps]]['average']["acc/top1"]
 
    # print(mcad_results)
    df = pd.DataFrame.from_dict(mcad_results)
    perc_var = df.apply(lambda x: ((x - x['Org FPS']) / x['Org FPS']) * 100)

    print(df)
    fig, ax = plt.subplots()
    perc_var.plot.line(ax=ax, linestyle='--', marker='o')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylabel("% Variation")
    
    for line in ax.get_lines():
        line.set_linewidth(3)
    
    plt.yticks(np.arange(-95, 0, 10))
    # plt.xlabel("FPS")
    plt.tight_layout()
    plt.show()

       
if __name__ == '__main__':
    # main()
    # plot_graphs("fps_study_results.json")
    plot_perc_change()
