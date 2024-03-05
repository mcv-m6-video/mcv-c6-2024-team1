from argparse import ArgumentParser

import torch

from task1_3 import run_finetuning

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
        "--fold-index",
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
