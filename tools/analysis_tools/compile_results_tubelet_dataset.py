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
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# from mmaction.evaluation.functional import (get_weighted_score,
#                                             mean_class_accuracy,
#                                             top_k_accuracy)


results_compiled = {
    "configs/tubelet_dataset/c3d/c3d_tubelet_dataset_config.py" :
    "tubelet_results/tubelet_test_output_c3d.pkl",
    
    "configs/tubelet_dataset/i3d/i3d_tubelet_dataset_config.py" :
    "tubelet_results/tubelet_test_output_i3d.pkl",
    
    "configs/tubelet_dataset/slowfast/slowfast_tubelet_dataset_config.py" : 
    "tubelet_results/tubelet_test_output_slowfast.pkl",
    
    "configs/tubelet_dataset/timesformer/timesformer_tubelet_dataset.py" :
        "tubelet_results/tubelet_test_output_timesformer.pkl",
    
    "configs/tubelet_dataset/videoswin/swin_tubelet_dataset.py" :
        "tubelet_results/tubelet_test_output_videoswin.pkl",

}


label_matcher = ["WALKING","RUNNING","SITTING","STANDING","GESTURING","CARRYING","USING_PHONE"]
label_matcher = [x.title() for x in label_matcher]



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
                            top_k_accuracy=dict(topk=(1,3))
                            ),
                        metric_list= ('top_k_accuracy', 'mean_class_accuracy')
                            )
        evaluator = Evaluator(test_evaluator)
        eval_results = evaluator.offline_evaluate(data_samples)
        print(os.path.basename(pkl_results),eval_results)

        results[os.path.basename(pkl_results)]["average"] = eval_results

        no_of_classes = len(label_matcher)
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
    
    C3D = "tubelet_test_output_c3d.pkl"
    I3D = "tubelet_test_output_i3d.pkl"
    SlowFast = "tubelet_test_output_slowfast.pkl"
    TimesFormer = "tubelet_test_output_timesformer.pkl"
    VideoSwin = "tubelet_test_output_videoswin.pkl"

    out = dict()

    # matplotlib.rc('font', **font)
    with open(f_name) as fd :
        data = json.load(fd)
        
    for x in ["C3D","I3D","SlowFast","TimesFormer","VideoSwin"] :
        out[F"{x}"] = data[eval(x)]['class_wise'][1]
    # out["C3D"] = data[C3D]['class_wise'][1]
    # out["10 FPS"] = data[slowfast_10fps]['class_wise'][1]
    # out["5 FPS"] = data[slowfast_5fps]['class_wise'][1]
    # out["3 FPS"] = data[slowfast_3fps]['class_wise'][1]
    # out["1 FPS"] = data[slowfast_1fps]['class_wise'][1]

    df = pd.DataFrame.from_dict(out)
    # df=df.drop(['talking_on_phone_desk','using_phone_desk', 'carrying_heavy', 'carrying_light'])
    print(df)
    df.plot.bar(edgecolor='white', linewidth=1)
    # df.plot.line(linewidth=2, marker='*', linestyle='--')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(ticks=range(len(df.index)), labels=df.index, rotation=90, ha="right")
    plt.xticks(rotation=0)
    ax = plt.gca()
    # ax.set_ylim([0.4, 1])
    plt.tight_layout()
    plt.legend(ncol=5)
    plt.show()

       
if __name__ == '__main__':
    main()
    plot_graphs("tubelet_results.json")
