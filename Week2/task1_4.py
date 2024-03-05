from argparse import ArgumentParser

import numpy as np
import torch

from task1_3 import run_finetuning


def split_strategy_A(number_of_frames):
    FRAME_25_PERCENT = int(number_of_frames / 4)
    train_idxs = np.arange(0, FRAME_25_PERCENT)
    test_idxs = np.arange(FRAME_25_PERCENT, number_of_frames)
    return train_idxs, test_idxs


def split_strategy_B(fold_idx, number_of_frames):
    # Assuming 4 folds
    fold_size = number_of_frames / 4
    train_idxs = []
    test_idxs = []
    for i in range(4):
        if i == fold_idx:
            test_idxs.extend(range(i * fold_size, (i + 1) * fold_size))
        else:
            train_idxs.extend(range(i * fold_size, (i + 1) * fold_size))
    return train_idxs, test_idxs


def split_strategy_C(number_of_frames):
    train_size = int(0.25 * number_of_frames)
    train_idxs = np.random.choice(number_of_frames, train_size, replace=False)
    test_idxs = np.setdiff1d(np.arange(number_of_frames), train_idxs)
    return train_idxs, test_idxs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="Strategy to use",
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Fold index for strategy B",
    )
    args = parser.parse_args()

    run_finetuning(
        device="cuda" if torch.cuda.is_available() else "cpu",
        strategy=args.strategy,
        fold_idx=args.fold_index,
    )
