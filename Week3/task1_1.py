import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time

# Define a function to calculate Mean Square Error (MSE)
def calculate_mse(flow, ground_truth_flow):
    mse = np.mean(np.square(flow - ground_truth_flow))
    return mse

# Define a function to calculate Percentage of Erroneous Pixels (PEPN)
def calculate_pepn(flow, ground_truth_flow, threshold=3):
    erroneous_pixels = np.sum(np.sqrt(np.sum(np.square(flow - ground_truth_flow), axis=-1)) > threshold)
    total_pixels = flow.shape[0] * flow.shape[1]
    pepn = erroneous_pixels / total_pixels * 100
    return pepn
def get_blocks(img, block_shape):
    """Divide the input image into blocks of specified shape.

    Take into account cases when img can not be evenly divided into
    blocks of specified shapes. In such cases create blocks with unequal
    shapes.

    Args:
        img: array of shape H, W, C, represents an image.
        block_shape: tuple that represents (H, W) of a block.

    Returns:
        blocks: list, where each element is a block.
        position: list, where each element is a left-upper positions
            of blocks in img coordinates.
        split_shape: (n_vertical_blocks, n_horizontal_blocks)
    """
    # Number of blocks in each direction
    split_shape = np.ceil((
        img.shape[0] / block_shape[0],
        img.shape[1] / block_shape[1]
    )).astype(np.int32)

    blocks = []
    positions = []
    y_position = 0
    for i in range(split_shape[0]):
        if i != split_shape[0]:
            stripe = img[i * block_shape[0]:i * block_shape[0] + block_shape[0]]  # for even block/img shapes
        else:
            stripe = img[i * block_shape[0]:]  # for uneven block/img shapes
        x_position = 0
        block = None
        # In this loop we are going to create the blocks for each stripe of the image
        for j in range(split_shape[1]):
            if j != split_shape[1]:
                block = stripe[:, j * block_shape[1]: j * block_shape[1] + block_shape[1]] # case when the block is even, we can take the block shape as it is
            else:
                block = stripe[:, j * block_shape[1]:] # case when the block is uneven, we need to take the rest of the stripe
            blocks.append(block)
            positions.append((y_position, x_position))
            x_position += block.shape[1]
        y_position += block.shape[0]
    return blocks, positions, split_shape


def get_window(img, win_shape, block_shape, pos):
    """Returns a window, centered in a block specified by its shape and pos.

    Args:
        img: NumPy array, image to be sliced.
        win_shape: tuple (win_height, win_width).
        block_shape: tuple (block_height, block_width).
        pos: tuple, top left position of block in img coordinates.

    Returns:
        window: slice of the img.
        win_pos: coordinates of top left corner of the window in img coords.

    """
    center = [pos[0] + block_shape[0] // 2, pos[1] + block_shape[1] // 2]
    #  From center calculate top, bottom, left and right coordinates, taking into account not to exceed img shape
    top = np.max([0, center[0] - win_shape[0] // 2])
    bottom = np.min([img.shape[0], center[0] - win_shape[0] // 2 + win_shape[0]])
    left = np.max([0, center[1] - win_shape[1] // 2])
    right = np.min([img.shape[1], center[1] - win_shape[1] // 2 + win_shape[1]])
    window = img[top:bottom, left:right]
    win_pos = (top, left)

    return window, win_pos


import numpy as np


def block_match(ref_frame, curr_frame, block_shape, win_shape, metric_func):
    flow = []
    blocks, positions, split_shape = get_blocks(curr_frame, block_shape)

    for block, pos in zip(blocks, positions):
        # Get the window centered around the current block
        window, win_pos = get_window(ref_frame, win_shape, block_shape, pos)

        # Calculate the scores for matching the current block with the window
        scores = metric_func(window, block)

        # Find the index of the best matching position in the window
        best_idx_flat = np.argmin(scores)
        best_idx_2d = np.unravel_index(best_idx_flat, scores.shape)

        # Calculate the position of the best matching block in the reference frame
        best_idx_2d_img = (best_idx_2d[0] + win_pos[0], best_idx_2d[1] + win_pos[1])

        # Calculate the displacement between the current block and the best matching block
        displacement = (best_idx_2d_img[0] - pos[0], best_idx_2d_img[1] - pos[1])

        # Ensure the displacement stays within the bounds of the image
        displacement = (
            max(-pos[0], min(displacement[0], ref_frame.shape[0] - pos[0] - block_shape[0])),
            max(-pos[1], min(displacement[1], ref_frame.shape[1] - pos[1] - block_shape[1]))
        )

        flow.append(displacement)

    flow = np.array(flow).reshape((*split_shape, 2))
    return flow


def block_match_log(ref_frame, curr_frame, block_shape, win_shape, metric_func,
                    max_level=3):
    """Perform logarithmic block matching.

    Calculate apparent movement between blocks of two frames, using
    recoursive (logarithmic) search.

    Args:
        ref_frame: NumPy array, reference frame, image where
            correspondences should by found.
        curr_frame: NumPy array, current frame, image, from which blocks
            are obtained.
        block_shape: tuple, shape of a block (height, width).
        win_shape: tuple, shape of a search window (height, width).
        metric_func: callable, metric to compare current block with
            candidates withing search window. Outputs scores
            for n_candidates, when input is img_shape,
            and (n_candidates, *img_shape).
        max_level: int, maximum number of iteration in log search

    Returns:
        flow: NumPy array, representing u and v movement between frames

    """
    flow = []
    blocks, positions, split_shape = get_blocks(curr_frame, block_shape)
    for block, pos in zip(blocks, positions):
        window, win_pos = get_window(ref_frame, win_shape, block.shape, pos)
        sliding = np.lib.stride_tricks.sliding_window_view(window, block.shape)
        sliding_center = np.array(sliding.shape)[:2] // 2
        for level in range(max_level):
            shift = np.array(sliding.shape)[:2] // 2 ** (level + 1) - 1
            indices = np.array([
                sliding_center - shift,
                sliding_center,
                sliding_center + shift
            ]).T
            indices = np.array([
                [indices[0][0], indices[1][1]],

                [indices[0][1], indices[1][0]],
                [indices[0][1], indices[1][1]],
                [indices[0][1], indices[1][2]],

                [indices[0][2], indices[1][1]],
            ])
            indices = np.clip(indices, 0, np.array(sliding.shape)[:2] - 1)
            candidates = np.array([sliding[w[0]][w[1]] for w in indices])
            scores = metric_func(candidates, block)
            current_center = indices[np.argmin(scores)]
            if np.all(current_center == sliding_center):
                break
            sliding_center = current_center
            pass
        best_idx_2d = sliding_center
        best_idx_2d_img = (best_idx_2d[0] + win_pos[0],
                           best_idx_2d[1] + win_pos[1])
        displacement = (best_idx_2d_img[0] - pos[0],
                        best_idx_2d_img[1] - pos[1])
        # BUG: flow sometimes exceedes possible values
        # to fix the bug, we need to clip the displacement to the range of possible values in the image (ref_frame)
        displacement = (
            max(-pos[0], min(displacement[0], ref_frame.shape[0] - pos[0] - block_shape[0])),
            max(-pos[1], min(displacement[1], ref_frame.shape[1] - pos[1] - block_shape[1]))
        )
        flow.append(displacement)
    flow = np.array(flow).reshape((*split_shape, 2))

    return flow


# create a metric function
def l2(a, b):
    diff = a - b
    square = np.power(diff, 2)
    sq_sum = np.sum(square, axis=(-1, -2, -3))
    return sq_sum

# use an existing L2 metric function
if __name__ == "__main__":
    # USAGE EXAMPLE
    from time import time
    sequence_number = "45"
    CURRENT_IMAGE = f"../Data/data_stereo_flow/training/colored_0/0000{sequence_number}_10.png"
    REF_IMAGE = f"../Data/data_stereo_flow/training/colored_0/0000{sequence_number}_11.png"
    GT_PATH = f"../Data/data_stereo_flow/training/flow_noc/0000{sequence_number}_10.png"
    # Block size (N): It is related to the expected movement:
    block_size = 16
    # Search area (P): It is related to the range of expected movement:
    # P pixels in every direction: (2P+N)x(2P+N) pixels. Typically P = N
    pixels = 16
    search_area = 2*pixels*block_size
    # Quantization step: Related to the accuracy of the estimated motion.
    # Typically, 1 pixel but it can go down to 1/4 of pixel
    step_size = 1
    curr = np.asarray(Image.open(CURRENT_IMAGE))
    ref = np.asarray(Image.open(REF_IMAGE))
    initial_time = time()
    #flow = block_match(ref, curr, (block_size, block_size), (search_area, search_area), l2)
    flow = block_match_log(ref, curr, (block_size, block_size), (search_area, search_area), l2)
    final_time = time()
    # Assuming ground truth flow is available
    ground_truth_flow = np.zeros_like(flow)  # Replace with actual ground truth flow

    # Calculate MSEN and PEPN
    mse = calculate_mse(flow, ground_truth_flow)
    pepn = calculate_pepn(flow, ground_truth_flow)

    # Print or log the results
    print(f"MSEN: {mse}")
    print(f"PEPN: {pepn}%")
    print(f"Runtime: {final_time - initial_time} seconds")

    ace = np.sum(np.square(flow), axis=-1)
    plt.imshow(ace)
    plt.quiver(flow[:, :, 1], flow[:, :, 0], angles="xy")
    #plt.show()
    #save plot in results folder
    plt.savefig(f'results/flow_{sequence_number}.jpg')