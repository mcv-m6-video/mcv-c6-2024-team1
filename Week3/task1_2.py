import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflow

from week_utils import compute_flow_metrics, hsv_plot, load_flow_gt

SEQUENCE = "000045"

GT_PATH = f"../Data/data_stereo_flow/training/flow_noc/{SEQUENCE}_10.png"
DATASET_PATH = f"../Data/data_stereo_flow/training/image_0/{SEQUENCE}"


def optical_flow_pyflow(
    im1,
    im2,
    alpha,
    ratio,
    min_width,
    n_outer_FP_iterations,
    n_inner_FP_iterations,
    n_SOR_iterations,
    col_type,
):
    start = time.time()
    u, v, _ = pyflow.coarse2fine_flow(
        im1,
        im2,
        alpha,
        ratio,
        min_width,
        n_outer_FP_iterations,
        n_inner_FP_iterations,
        n_SOR_iterations,
        col_type,
    )
    flow_pyflow = np.dstack((u, v))
    end = time.time()
    total_time = end - start
    return flow_pyflow, total_time


def optical_flow_farneback(
    im1, im2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
):
    start = time.time()
    flow_farneback = cv2.calcOpticalFlowFarneback(
        im1,
        im2,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    end = time.time()
    total_time = end - start
    return flow_farneback, total_time


def task_1_2():
    # Read images
    img1 = cv2.imread(DATASET_PATH + "_10.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(DATASET_PATH + "_11.png", cv2.IMREAD_GRAYSCALE)

    gt = load_flow_gt(GT_PATH)
    im1 = np.atleast_3d(img1.astype(float) / 255.0)
    im2 = np.atleast_3d(img2.astype(float) / 255.0)

    # Pyflow
    alpha = 0.012
    ratio = 0.5
    min_width = 20
    n_outer_FP_iterations = 1
    n_inner_FP_iterations = 1
    n_SOR_iterations = 15
    col_type = 1
    pyflow_flow, pyflow_time = optical_flow_pyflow(
        im1,
        im2,
        alpha,
        ratio,
        min_width,
        n_outer_FP_iterations,
        n_inner_FP_iterations,
        n_SOR_iterations,
        col_type,
    )

    # Farneback (openCV)
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0
    farneback_flow, farneback_time = optical_flow_farneback(
        img1, img2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    )

    # Compute metrics
    msen, pepn, _ = compute_flow_metrics(pyflow_flow, gt)
    print(
        "Pyflow: -- Time: "
        + str(pyflow_time)
        + " | MSEN: "
        + str(msen)
        + " | PEPN: "
        + str(pepn)
    )

    msen, pepn, _ = compute_flow_metrics(farneback_flow, gt)
    print(
        "Farneback: -- Time: "
        + str(farneback_time)
        + " | MSEN: "
        + str(msen)
        + " | PEPN: "
        + str(pepn)
    )

    pyflow_hsv = hsv_plot(pyflow_flow)
    farneback_hsv = hsv_plot(farneback_flow)

    plt.figure()
    plt.subplot(211)
    plt.imshow(np.array(pyflow_hsv))
    plt.subplot(212)
    plt.imshow(np.array(farneback_hsv))
    plt.show()


if __name__ == "__main__":
    task_1_2()
