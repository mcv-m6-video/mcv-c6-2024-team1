import numpy as np


def split_strategy_A(number_of_frames):
    FRAME_25_PERCENT = number_of_frames / 4
    train_idxs = np.arange(0, FRAME_25_PERCENT)
    test_idxs = np.arange(FRAME_25_PERCENT, FRAME_25_PERCENT * 4)
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
