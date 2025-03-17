#!/usr/bin/env python3
"""
presplit_embeddings.py

1) Scans a directory of .npy embeddings (like "014890_seg006.npy").
2) Sorts them by song ID and segment number.
3) Splits them into train/val/test using 80%/10%/10% ratios.
4) Moves them into `train/`, `val/`, `test/` subdirectories under `embeddings_dir`.

Run:
python presplit_embeddings.py --embeddings_dir ./../data/embeddings
"""

import os
import glob
import argparse
import shutil
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Path to the directory containing original .npy embeddings.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of files for train set (default: 0.8).")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of files for val set (default: 0.1).")
    parser.add_argument("--move", action="store_true",
                        help="If set, move files instead of copying (default: copy).")
    return parser.parse_args()

def main():
    args = parse_args()
    embeddings_dir = args.embeddings_dir

    # 1) Gather all .npy files
    all_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
    if not all_files:
        print(f"[Error] No .npy files found in {embeddings_dir}")
        return

    # 2) Create subdirectories for train/val/test
    train_dir = os.path.join(embeddings_dir, "train")
    val_dir   = os.path.join(embeddings_dir, "val")
    test_dir  = os.path.join(embeddings_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 3) Shuffle and split
    random.shuffle(all_files)

    N = len(all_files)
    train_end = int(N * args.train_ratio)
    val_end   = int(N * (args.train_ratio + args.val_ratio))

    train_files = all_files[:train_end]
    val_files   = all_files[train_end:val_end]
    test_files  = all_files[val_end:]

    print(f"Total files: {N}. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 4) Move or copy files
    def transfer_files(file_list, target_dir):
        for f in file_list:
            basename = os.path.basename(f)
            dest_path = os.path.join(target_dir, basename)
            if args.move:
                shutil.move(f, dest_path)
            else:
                shutil.copy2(f, dest_path)

    transfer_files(train_files, train_dir)
    transfer_files(val_files,   val_dir)
    transfer_files(test_files,  test_dir)

    print("[Done] Embeddings successfully split into train/val/test.")

if __name__ == "__main__":
    main()

