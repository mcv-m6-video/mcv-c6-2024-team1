from bm_utils import get_blocks, get_window
import numpy as np

import numpy as np


def block_match_log(ref_frame, curr_frame, block_shape, win_shape, metric_func,
                    direction="backward", max_level=3):
    flow = []
    if direction == "backward":
        ref_frame, curr_frame = curr_frame, ref_frame

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

        best_idx_2d = sliding_center
        best_idx_2d_img = (best_idx_2d[0] + win_pos[0],
                           best_idx_2d[1] + win_pos[1])
        displacement = (best_idx_2d_img[0] - pos[0],
                        best_idx_2d_img[1] - pos[1])
        displacement = (
            max(-pos[0], min(displacement[0], ref_frame.shape[0] - pos[0] - block_shape[0])),
            max(-pos[1], min(displacement[1], ref_frame.shape[1] - pos[1] - block_shape[1]))
        )
        flow.append(displacement)

    flow = np.array(flow).reshape((*split_shape, 2))

    return flow
