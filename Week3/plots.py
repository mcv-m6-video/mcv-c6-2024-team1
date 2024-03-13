import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


import matplotlib.cm as cm

from matplotlib.colors import Normalize
from PIL import Image
import math


def visualize_optical_flow_error(GT, OF_pred, output_dir="./results/"):
    u_diff, v_diff = GT[:, :, 0] - \
                                  OF_pred[:, :, 0], GT[:, :, 1] - OF_pred[:, :, 1]
    error_dist = np.sqrt(u_diff ** 2 + v_diff ** 2)

    max_range = int(math.ceil(np.amax(error_dist)))

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.title('MSEN Distribution')
    plt.hist(error_dist.ravel(),
             bins=30, range=(0.0, max_range))
    plt.ylabel('Count')
    plt.xlabel('Mean Square Error in Non-Occluded Areas')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "MSEN_hist.png"))
    plt.close()



def visualize_arrow(im1_path: str, flow, filename: str):
    im1 = cv2.imread(im1_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    h, w = flow.shape[:2]
    flow_horizontal = flow[:, :, 0]
    flow_vertical = flow[:, :, 1]

    step_size = 12
    X, Y = np.meshgrid(np.arange(0, w, step_size), np.arange(0, h, step_size))
    U = flow_horizontal[Y, X]
    V = flow_vertical[Y, X]

    magnitude = np.sqrt(U ** 2 + V ** 2)
    norm = Normalize()
    norm.autoscale(magnitude)
    cmap = cm.inferno

    plt.figure(figsize=(10, 10))
    plt.imshow(im1)
    plt.quiver(X, Y, U, V, norm(magnitude), angles='xy', scale_units='xy', scale=1, cmap=cmap, width=0.0015)
    plt.axis('off')
    plt.savefig(f'./results/arrow_{filename}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_direction_idx_plot(im1_path: str, flow, filename):
    im1 = np.array(Image.open(im1_path).convert('RGB'))
    im1 = im1.astype(float) / 255.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Background
    ax.imshow(im1)

    step = 10
    x, y = np.meshgrid(np.arange(0, flow.shape[1], step), np.arange(0, flow.shape[0], step))
    u = flow[y, x, 0]
    v = flow[y, x, 1]

    direction = np.arctan2(v, u)
    norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm(direction))
    colors = colors.reshape(-1, colors.shape[-1])
    quiver = ax.quiver(x, y, u, v, color=colors, angles='xy', scale_units='xy', scale=1, width=0.0015, headwidth=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = fig.colorbar(sm, ax=ax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    cbar.set_label('Direction')
    plt.savefig(f'./results/{filename}_direction.png', dpi=300, bbox_inches='tight')
    plt.close()

def plotArrowsOP_save(flow_img, step, path):
    img = plt.imread(path)
    #flow_img = cv2.resize(flow_img, (0, 0), fx=1. / step, fy=1. / step)
    u = flow_img[:, :, 0]
    v = flow_img[:, :, 1]
    x = np.arange(0, np.shape(flow_img)[0] * step, step)
    y = np.arange(0, np.shape(flow_img)[1] * step, step)
    U, V = np.meshgrid(y, x)
    M = np.hypot(u, v)
    plt.quiver(U, V, u, -v, M, color='g')
    plt.imshow(img, alpha=0.5, cmap='gray')
    plt.title('Orientation OF')
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig('results/plotArrowsOP.jpg')