import pickle
import numpy as np

results1_path = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_x3d.pkl"
results2_path = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_posec3d.pkl"
gt_path = "/ghome/group01/mcv-c6-2024-team1/Week7/results/gt.pkl"

with open(results1_path, "rb") as f:
    r1 = pickle.load(f)

with open(results2_path, "rb") as f:
    r2 = pickle.load(f)

with open(gt_path, "rb") as f:
    gt = pickle.load(f)

hits_r1 = 0
hits_r2 = 0
count = len(r1)

print("Length:", count)

# for k in r1.keys():
#     argmax_r1 = np.argmax(r1[k])
#     argmax_r2 = np.argmax(r2[k])
#     gt_k = gt[k]
    
#     hits_r1 += argmax_r1 == gt_k
#     hits_r2 += argmax_r2 == gt_k

# print("Accuracy (r1):", 100 * hits_r1 / count)
# print("Accuracy (r2):", 100 * hits_r2 / count)

fusion_results = {}
for k in r1.keys():
    fusion_results[k] = (r1[k] + r2[k]) / 2

hits_fusion = 0
for k in fusion_results.keys():
    argmax_fusion = np.argmax(fusion_results[k])
    gt_k = gt[k]
    hits_fusion += argmax_fusion == gt_k

print("Accuracy (Fusion - Averaging):", 100 * hits_fusion / count)

fusion_results = {}
for k in r1.keys():
    fusion_results[k] = np.maximum(r1[k], r2[k])

hits_fusion = 0
for k in fusion_results.keys():
    argmax_fusion = np.argmax(fusion_results[k])
    gt_k = gt[k]
    hits_fusion += argmax_fusion == gt_k

print("Accuracy (Fusion - Maximum Voting):", 100 * hits_fusion / count)


