import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class TransitionsDataset(Dataset):
    """
    A dataset that reads .npy embeddings from subfolders:
      embeddings_dir/train/*.npy
      embeddings_dir/val/*.npy
      embeddings_dir/test/*.npy
    or from a single folder if you'd prefer.

    Instead of splitting at runtime, we assume you either:
      1) pre-split the data into subfolders (train, val, test), or
      2) keep them in one folder and specify 'split' if you do custom logic.

    This version ensures a negative always exists (provided multiple songs exist).
    """

    def __init__(self, embeddings_dir, split="train",
                 overlap_frames=500):
        """
        overlap_frames: how many frames from end-of-anchor or start-of-positive/negative
        """
        super().__init__()
        self.overlap_frames = overlap_frames

        # 1) Because we do pre-splitting, we look in e.g. embeddings_dir/train
        data_dir = os.path.join(embeddings_dir, split)
        if not os.path.isdir(data_dir):
            raise ValueError(f"[Error] Directory {data_dir} does not exist. Check your presplit or path.")

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not all_files:
            raise ValueError(f"[Error] No .npy files found in {data_dir}")

        # 2) Parse (song_id, seg#)
        self.segments = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            if "_" in fname:
                song_id, seg_part = fname.split("_", 1)
                seg_part = seg_part.replace(".npy", "")
            else:
                song_id = fname.replace(".npy", "")
                seg_part = "seg0"

            seg_num = 0
            if seg_part.startswith("seg"):
                try:
                    seg_num = int(seg_part[3:])
                except:
                    seg_num = 0

            self.segments.append((fpath, song_id, seg_num))

        # 3) Sort by (song_id, seg_num) => anchor->positive
        self.segments.sort(key=lambda x: (x[1], x[2]))

        # 4) Build anchor-positive pairs
        self.pairs = []
        for i in range(len(self.segments) - 1):
            fpath_i, song_i, seg_i = self.segments[i]
            fpath_j, song_j, seg_j = self.segments[i+1]
            if song_i == song_j:
                self.pairs.append((i, i+1))

        print(f"[{split}] Found {len(self.segments)} segments, built {len(self.pairs)} anchor-positive pairs.")

        # 5) For negative sampling, we need at least 2 distinct songs
        self.all_indices = list(range(len(self.segments)))
        self.song_to_indices = {}
        for idx, (fp, s_id, s_num) in enumerate(self.segments):
            if s_id not in self.song_to_indices:
                self.song_to_indices[s_id] = []
            self.song_to_indices[s_id].append(idx)

        # If there's only one unique song, we can't do negative from a different song
        if len(self.song_to_indices) < 2:
            print("[Warning] Only one song found! Negative sampling is impossible.")
            print("=> We'll skip building pairs altogether to avoid errors.")
            self.pairs = []

    def __len__(self):
        return len(self.pairs)

    @property
    def input_dim(self):
        if len(self.segments) == 0:
            return 0  # no data
        fpath, _, _ = self.segments[0]
        arr = np.load(fpath)
        return arr.shape[1]

    def __getitem__(self, idx):
        anchor_idx, positive_idx = self.pairs[idx]

        # Load anchor
        fpath_i, song_i, _ = self.segments[anchor_idx]
        arr_i = np.load(fpath_i)  # (T, D)
        T_i = arr_i.shape[0]
        start_i = max(0, T_i - self.overlap_frames)
        anchor = arr_i[start_i:, :]

        # Load positive
        fpath_j, _, _ = self.segments[positive_idx]
        arr_j = np.load(fpath_j)
        T_j = arr_j.shape[0]
        end_j = min(T_j, self.overlap_frames)
        positive = arr_j[:end_j, :]

        # Negative from different song
        # We'll gather all songs except anchor's. Then pick one at random.
        valid_songs = [sid for sid in self.song_to_indices.keys() if sid != song_i]
        # Edge case: if there's no other song => fallback
        if not valid_songs:
            # Return anchor=positive=negative => leads to high loss or skip
            negative = np.zeros_like(anchor)
            # or skip
            # raise ValueError("No other songs for negative sampling!")
        else:
            neg_song = random.choice(valid_songs)
            neg_idx = random.choice(self.song_to_indices[neg_song])
            fpath_neg, _, _ = self.segments[neg_idx]
            arr_neg = np.load(fpath_neg)
            T_n = arr_neg.shape[0]
            end_n = min(T_n, self.overlap_frames)
            negative = arr_neg[:end_n, :]

        # Convert to torch
        anchor   = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()

        return anchor, positive, negative
