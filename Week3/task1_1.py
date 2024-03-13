from argparse import ArgumentParser

from PIL import Image
from time import time
from block_matching import block_match_log
from bm_utils import ssd, sad, ncc
from plots import plotArrowsOP_save, visualize_optical_flow_error, visualize_arrow
from week_utils import compute_flow_metrics, load_flow_gt
import numpy as np
import cv2

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--sequence_number", type=str, default="45", help="Sequence number")
    parser.add_argument("--block_size", type=int, default=16, help="Block size")
    parser.add_argument("--search_area", type=int, default=21, help="Search area")
    parser.add_argument("--metric", type=str, default="ncc", help="Metric")
    parser.add_argument("--direction", type=str, default="forward", help="Direction")
    parser.add_argument("--max_level", type=int, default=3, help="Max level")
    args = parser.parse_args()

    sequence_number = args.sequence_number
    CURRENT_IMAGE = f"../Data/data_stereo_flow/training/image_0/0000{sequence_number}_10.png"
    REF_IMAGE = f"../Data/data_stereo_flow/training/image_0/0000{sequence_number}_11.png"
    GT_PATH = f"../Data/data_stereo_flow/training/flow_noc/0000{sequence_number}_10.png"
    # Block size (N): It is related to the expected movement:
    block_size = args.block_size
    # Search area (P): It is related to the range of expected movement
    search_area = args.search_area
    selected_dist = args.metric
    metric = {
        "ssd": ssd,
        "sad": sad,
        "ncc": ncc
    }[selected_dist]
    direction = args.direction
    max_level = args.max_level

    curr = np.asarray(Image.open(CURRENT_IMAGE))
    ref = np.asarray(Image.open(REF_IMAGE))
    initial_time = time()

    flow = block_match_log(ref, curr, (block_size, block_size), (search_area, search_area), metric, direction)
    final_time = time()

    gt_flow = load_flow_gt(GT_PATH)

    flow_reshaped = cv2.resize(flow, (gt_flow.shape[1], gt_flow.shape[0]),interpolation=cv2.INTER_NEAREST)
    # Calculate MSEN and PEPN
    mse, pepn , _ = compute_flow_metrics(flow_reshaped, gt_flow)
    # Print or log the results
    print(f"MSEN: {mse}")
    print(f"PEPN: {pepn}%")
    print(f"Runtime: {final_time - initial_time} seconds")

    # ace = np.sum(np.square(flow), axis=-1)
    # plt.imshow(ace)
    # plt.quiver(flow[:, :, 1], flow[:, :, 0], angles="xy")
    # # plt.show()
    # # save plot in results folder
    # plt.savefig(f'results/flow_{sequence_number}_match_log.jpg')

    #plotArrowsOP_save(flow, 10 , CURRENT_IMAGE)