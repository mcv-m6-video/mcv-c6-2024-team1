import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PLOTS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/plots/"

CLASS_NAMES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
    "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat",
    "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit",
    "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour",
    "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run",
    "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp",
    "smile", "smoke", "somersault", "stand", "swing_baseball", "sword",
    "sword_exercise", "talk", "throw", "turn", "walk", "wave"
]


class Plots:
    @staticmethod
    def generate_confusion_matrix(y_test, y_pred, save_path):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Labels', fontsize=14)
        plt.ylabel('True Labels', fontsize=14)
        plt.xticks(rotation=90, fontsize=14)  # Rotating labels
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)

    @staticmethod
    def generate_per_class_accuracy_plot(y_test, y_pred, save_path):
        cm = confusion_matrix(y_test, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.figure(figsize=(20, 10))
        plt.bar(CLASS_NAMES, per_class_accuracy, color='skyblue')
        plt.xlabel('Classes', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Per-Class Accuracy', fontsize=16)
        plt.xticks(rotation=90, fontsize=16)  # Rotating labels
        plt.yticks(fontsize=16)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path)


results_path = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_posec3d.pkl"
gt_path = "/ghome/group01/mcv-c6-2024-team1/Week7/results/gt.pkl"

with open(results_path, "rb") as f:
    r = pickle.load(f)

with open(gt_path, "rb") as f:
    gt = pickle.load(f)

hits = count = 0
y_test = []
y_pred = []
for k in r.keys():
    argmax = np.argmax(r[k])
    hits += argmax == gt[k]
    y_pred.append(argmax)
    y_test.append(gt[k])
    count += 1

Plots.generate_confusion_matrix(y_test, y_pred, PLOTS_PATH + "confusion.png")