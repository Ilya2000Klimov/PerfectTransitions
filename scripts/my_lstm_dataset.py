import os
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class TransitionsDataset(Dataset):
    """
    - Scans a directory of .npy embeddings (one file per segment).
    - Splits them into train/val/test by song ID or index.
    - For each index i, we define:
        anchor   = last overlap_frames of clip i
        positive = first overlap_frames of clip (i+1) (same song)
        negative = first overlap_frames of a random clip from a different song
    - "Two-stage embeddings" => we specifically target the boundary frames
      (e.g. last 80 for anchor, first 80 for positive).
    """

    def __init__(self, embeddings_dir, split="train",
                 overlap_frames=500, train_ratio=0.8, val_ratio=0.1):
        """
        overlap_frames: how many frames we take from the end or start of each clip
        train_ratio, val_ratio: how we split the dataset
        """
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.overlap_frames = overlap_frames

        # 1) Gather all .npy files
        all_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))

        # 2) Optionally parse "song ID" from filename if needed
        #    We assume file naming like: 000145_seg010.npy
        #    Song ID might be "000145", seg index might be "seg010"
        #    We'll store (filepath, song_id, seg_id)
        self.segments = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            # Example: "000145_seg010.npy"
            # We'll parse up to first underscore => "000145", after => "seg010"
            # Adjust as needed for your naming scheme
            if "_" in fname:
                song_id, seg_id = fname.split("_", 1)
                seg_id = seg_id.replace(".npy", "")
            else:
                # fallback: entire fname as song_id
                song_id = fname.replace(".npy", "")
                seg_id = "seg0"

            self.segments.append((fpath, song_id, seg_id))

        # 3) Sort by (song_id, seg_id) so consecutive segments are next to each other
        #    This helps define anchor -> positive pairs
        def seg_sort_key(x):
            # (filepath, "000145", "seg010")
            # parse the numeric part in "seg010"
            fpath, s_id, seg_str = x
            # strip "seg"
            seg_num = 0
            if seg_str.startswith("seg"):
                try:
                    seg_num = int(seg_str[3:])
                except:
                    seg_num = 0
            return (s_id, seg_num)
        self.segments.sort(key=seg_sort_key)

        # 4) Split into train/val/test by index
        N = len(self.segments)
        train_end = int(N * train_ratio)
        val_end   = int(N * (train_ratio + val_ratio))

        if split == "train":
            self.segments = self.segments[:train_end]
        elif split == "val":
            self.segments = self.segments[train_end:val_end]
        elif split == "test":
            self.segments = self.segments[val_end:]
        else:
            raise ValueError(f"Unknown split={split}")

        # 5) Build anchor-positive pairs:
        #    We'll store a list of (anchor_idx, positive_idx) whenever
        #    segments belong to the same song & are consecutive in seg_id.
        self.pairs = []
        for i in range(len(self.segments) - 1):
            fpath_i, song_i, seg_i = self.segments[i]
            fpath_j, song_j, seg_j = self.segments[i+1]
            if song_i == song_j:
                # consecutive segments in the same song => valid anchor-positive
                self.pairs.append((i, i+1))

        # We'll gather all song IDs for negative sampling
        # or we can sample from all segments randomly
        self.all_indices = list(range(len(self.segments)))

    def __len__(self):
        return len(self.pairs)

    @property
    def input_dim(self):
        """
        The dimension of each frame, e.g. 768 for BEATs Large.
        We'll load one file to figure out shape.
        """
        fpath, _, _ = self.segments[0]
        arr = np.load(fpath)
        return arr.shape[1]  # (T, D) => D

    def __getitem__(self, idx):
        """
        Returns (anchor, positive, negative) each shaped [T', D].
        anchor = last overlap_frames from clip i
        positive = first overlap_frames from clip (i+1)
        negative = first overlap_frames from a random *different* clip
        """
        anchor_idx, positive_idx = self.pairs[idx]

        # Load anchor clip
        fpath_i, _, _ = self.segments[anchor_idx]
        arr_i = np.load(fpath_i)  # shape (T, D)
        T_i = arr_i.shape[0]
        # We'll take the last overlap_frames
        start_i = max(0, T_i - self.overlap_frames)
        anchor = arr_i[start_i:, :]  # (overlap_frames, D) or smaller if T_i < overlap_frames

        # Load positive clip
        fpath_j, _, _ = self.segments[positive_idx]
        arr_j = np.load(fpath_j)  # shape (T, D)
        T_j = arr_j.shape[0]
        # We'll take the first overlap_frames
        end_j = min(T_j, self.overlap_frames)
        positive = arr_j[:end_j, :]  # (overlap_frames, D)

        # Negative: choose random index from a different song
        while True:
            neg_idx = random.choice(self.all_indices)
            if neg_idx not in [anchor_idx, positive_idx]:
                # Optionally also ensure it's a different song if you want
                # We skip if same song
                if self.segments[neg_idx][1] != self.segments[anchor_idx][1]:
                    break

        fpath_neg, _, _ = self.segments[neg_idx]
        arr_neg = np.load(fpath_neg)
        T_n = arr_neg.shape[0]
        end_n = min(T_n, self.overlap_frames)
        negative = arr_neg[:end_n, :]

        # Convert to torch tensors
        anchor   = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()

        return anchor, positive, negative
