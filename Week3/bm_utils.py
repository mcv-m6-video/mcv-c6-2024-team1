import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_blocks(img, block_shape):
    '''
    Splits an image into blocks of a given shape.
    Args:
        img: NumPy array, image to be split.
        block_shape: tuple (block_height, block_width).
    Returns:
        blocks: list of NumPy arrays, blocks of the image.

    '''
    split_shape = np.ceil((
        img.shape[0] / block_shape[0],
        img.shape[1] / block_shape[1]
    )).astype(np.int32)

    blocks = []
    positions = []
    pos_u = 0
    for i in range(split_shape[0]):
        if i != split_shape[0]:
            stripe = img[i * block_shape[0]:i * block_shape[0] + block_shape[0]]
        else:
            stripe = img[i * block_shape[0]:]  # for uneven block/img shapes
        pos_v = 0
        for j in range(split_shape[1]):
            if j != split_shape[1]:
                block = stripe[:, j * block_shape[1]: \
                                  j * block_shape[1] + block_shape[1]]
            else:
                block = stripe[:, j * block_shape[1]:]
            blocks.append(block)
            positions.append((pos_u, pos_v))
            pos_v += block.shape[1]
        pos_u += block.shape[0]
    return blocks, positions, split_shape


def get_window(img, win_shape, block_shape, pos):
    '''
    Returns a window, centered in a block specified by its shape and pos.
    Args:
        img: NumPy array, image to be sliced.
        win_shape: tuple (win_height, win_width).
        block_shape: tuple (block_height, block_width).
        pos: tuple, top left position of block in img coordinates.

    Returns:
        window: slice of the img.

    '''
    center = [pos[0] + block_shape[0] // 2, pos[1] + block_shape[1] // 2]
    top = np.max([0, center[0] - win_shape[0] // 2])
    bot = np.min([img.shape[0], center[0] - win_shape[0] // 2 + win_shape[0]])
    left = np.max([0, center[1] - win_shape[1] // 2])
    right = np.min([img.shape[1], center[1] - win_shape[1] // 2 + win_shape[1]])
    window = img[top:bot, left:right]
    win_pos = (top, left)

    return window, win_pos



# METRICS


# SSD : Sum of Squared Differences
def ssd(a, b):
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=(-1, -2, -3))
    return sq_sum

# SAD: Sum of Absolute Differences
def sad(a, b):
    diff = a - b
    abs_diff = np.abs(diff)
    sum_abs_diff = np.sum(abs_diff, axis=(-1, -2, -3))
    return sum_abs_diff

 # NCC: Normalized Cross Correlation
def ncc(a, b):
    ncc = np.sum((a - np.mean(a)) * (b - np.mean(b))) / (
                np.std(a) * np.std(b) * a.size)
    return ncc





def estimate_block_flow(block_size, distance_type, blocks_pos, ref_img, curr_img):
    tlx_ref = blocks_pos['tlx_ref']
    tly_ref = blocks_pos['tly_ref']
    init_tlx_curr = blocks_pos['init_tlx_curr']
    init_tly_curr = blocks_pos['init_tly_curr']
    end_tlx_curr = blocks_pos['end_tlx_curr']
    end_tly_curr = blocks_pos['end_tly_curr']

    if distance_type == 'NCC':
        corr = cv2.matchTemplate(
            curr_img[init_tly_curr:end_tly_curr + block_size, init_tlx_curr:end_tlx_curr + block_size],
            ref_img[tly_ref:tly_ref + block_size, tlx_ref:tlx_ref + block_size],
            method=cv2.TM_CCORR_NORMED)

        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        flow_x = x + init_tlx_curr - tlx_ref
        flow_y = y + init_tly_curr - tly_ref

    else:

        min_dist = np.inf
        for y_curr in range(init_tly_curr, end_tly_curr):
            for x_curr in range(init_tlx_curr, end_tlx_curr):
                wind_ref = ref_img[tly_ref:tly_ref + block_size, tlx_ref:tlx_ref + block_size]
                wind_curr = curr_img[y_curr:y_curr + block_size, x_curr:x_curr + block_size]

                if distance_type == 'SAD':
                    dist = np.sum(np.abs(wind_ref - wind_curr))
                elif distance_type == 'SSD':
                    dist = np.sum((wind_ref - wind_curr) ** 2)
                else:
                    raise ValueError('This distance is unknown')

                if dist < min_dist:
                    min_dist = dist
                    flow_x = x_curr - tlx_ref
                    flow_y = y_curr - tly_ref

    return [flow_x, flow_y]

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