import os
import pickle
import matplotlib.pyplot as plt

frame_dir = "/ghome/group01/mcv-c6-2024-team1/Week7/frames/shoot_ball/KELVIN_shoot_ball_u_cm_np1_ba_med_7/"
img_paths = sorted([frame_dir + img for img in os.listdir(frame_dir)])

with open("data/hmdb51_2d.pkl", "rb") as f:
    skeletons = pickle.load(f)

# element 0 is KELVIN_shoot_ball_u_cm_np1_ba_med_7 (shoot_ball category)
# person 0, frame 0     
len_keypoints = len(skeletons["annotations"][0]["keypoint"][0, :, :])

for i in range(len_keypoints):
    keypoints = skeletons["annotations"][0]["keypoint"][0, i, :]
    image = plt.imread(img_paths[i])
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[0, 320, 240, 0])
    ax.scatter(keypoints[:, 0], keypoints[:, 1], color='red', s=10)  # red color, point size 10
    plt.savefig(f"/ghome/group01/mcv-c6-2024-team1/Week7/plots/frame_{i+1}.png")
    plt.show()
