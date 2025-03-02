# #!/usr/bin/env python3
# """
# presplit_triplets.py
# --------------------
# 1) Scans a directory of .npy embeddings (like "014890_seg006.npy").
# 2) Parses (song_id, seg_num) from filenames.
# 3) Sorts them so consecutive segments in the same song are adjacent.
# 4) For each consecutive pair in the same song, build an (anchor, positive).
# 5) For each pair, choose a negative from a different song (at random).
# 6) Collect these triplets into a list:
#    [
#      {"anchor": "...", "positive": "...", "negative": "..."},
#      ...
#    ]
# 7) Shuffle them, then do a ratio-based train/val/test split (like 0.8/0.1/0.1).
# 8) Writes out three JSON files:
#    train_triplets.json, val_triplets.json, test_triplets.json
# """

# import os
# import glob
# import json
# import random
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--embeddings_dir", type=str, required=True,
#                         help="Directory containing all .npy embeddings.")
#     parser.add_argument("--train_ratio", type=float, default=0.8,
#                         help="Fraction for train (default: 0.8).")
#     parser.add_argument("--val_ratio", type=float, default=0.1,
#                         help="Fraction for val (default: 0.1).")
#     parser.add_argument("--output_dir", type=str, required=True,
#                         help="Where to write train_triplets.json, val_triplets.json, test_triplets.json.")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     embeddings_dir = args.embeddings_dir

#     # 1) Gather all .npy
#     all_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
#     if not all_files:
#         print(f"[Error] No .npy files found in {embeddings_dir}")
#         return

#     # 2) Parse (song_id, seg_num) from filenames
#     segments = []
#     for fpath in all_files:
#         fname = os.path.basename(fpath)
#         # Example: "014890_seg006.npy"
#         if "_" in fname:
#             song_id, seg_part = fname.split("_", 1)
#             seg_part = seg_part.replace(".npy", "")  # e.g. "seg006"
#         else:
#             song_id = fname.replace(".npy", "")
#             seg_part = "seg0"

#         # parse the numeric portion from segXXX
#         seg_num = 0
#         if seg_part.startswith("seg"):
#             try:
#                 seg_num = int(seg_part[3:])
#             except:
#                 seg_num = 0

#         segments.append((fpath, song_id, seg_num))

#     # 3) Sort them by (song_id, seg_num)
#     segments.sort(key=lambda x: (x[1], x[2]))

#     # 4) Build anchor->positive for consecutive segments in same song
#     #    Then we'll handle negative.
#     triplets = []
#     all_indices = list(range(len(segments)))
#     for i in range(len(segments) - 1):
#         fpath_i, song_i, seg_i = segments[i]
#         fpath_j, song_j, seg_j = segments[i+1]

#         if song_i == song_j:
#             # Consecutive segments => anchor-positive
#             # We'll pick a negative from a different song
#             anchor_path   = os.path.basename(fpath_i)  # store just the filename
#             positive_path = os.path.basename(fpath_j)

#             # pick negative
#             while True:
#                 neg_idx = random.choice(all_indices)
#                 if neg_idx not in [i, i+1]:
#                     fneg, song_neg, seg_neg = segments[neg_idx]
#                     if song_neg != song_i:
#                         negative_path = os.path.basename(fneg)
#                         break

#             triplets.append({
#                 "anchor":   anchor_path,
#                 "positive": positive_path,
#                 "negative": negative_path
#             })

#     print(f"[Info] Built {len(triplets)} triplets in total.")

#     # 5) Shuffle triplets
#     random.shuffle(triplets)

#     # 6) Train/Val/Test split
#     N = len(triplets)
#     train_end = int(N * args.train_ratio)
#     val_end   = int(N * (args.train_ratio + args.val_ratio))

#     train_data = triplets[:train_end]
#     val_data   = triplets[train_end:val_end]
#     test_data  = triplets[val_end:]

#     print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

#     # 7) Make output_dir
#     os.makedirs(args.output_dir, exist_ok=True)

#     # 8) Write out JSON for each split
#     def _save_json(data, filename):
#         path = os.path.join(args.output_dir, filename)
#         with open(path, "w") as f:
#             json.dump(data, f, indent=2)
#         print(f"[Saved] {path} with {len(data)} triplets.")

#     _save_json(train_data, "train_triplets.json")
#     _save_json(val_data,   "val_triplets.json")
#     _save_json(test_data,  "test_triplets.json")

#     print("[Done] Presplitting triplets complete!")

# if __name__ == "__main__":
#     main()
