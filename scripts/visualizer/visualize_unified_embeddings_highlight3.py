#!/usr/bin/env python3
"""
visualize_unified_embeddings_highlight3.py

This script does the following:
1) Loads .npy embeddings from a directory (plus subfolders if needed).
2) Builds a track_id -> genre_id mapping from track_genres.csv and a genre_id -> title from genres.csv.
3) Reduces embeddings to 2D or 3D with t-SNE or PCA.
4) **Randomly picks 3 songs**:
   - Two that share the same genre (colored BLUE, GREEN),
   - One that has a different genre (colored RED),
   - Everything else is gray.
5) Saves the plot to embedding_visualizations/.

Example usage:
  python visualize_unified_embeddings_highlight3.py \
      --embeddings_dir ./../data/embeddings \
      --recursive \
      --method tsne \
      --dim 2 \
      --max_samples 5000 \
      --track_genres_csv ../data/fma_metadata/track_genres.csv \
      --genres_csv ../data/fma_metadata/genres.csv

"""

import os
import glob
import csv
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

###############################################################################
# 1) Read track->genre_id and genre_id->title
###############################################################################
def load_track_genres(track_genres_path):
    """
    Expects a CSV with:
      track_id,genre_id
      14890,3
      8202,4
      ...
    Returns dict track_to_genre: {track_id -> genre_id}
    """
    track_to_genre = {}
    with open(track_genres_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            gid = int(row["genre_id"])
            track_to_genre[tid] = gid
    return track_to_genre

def load_genres(genres_csv_path):
    """
    Expects a CSV with:
      genre_id,#tracks,parent,title,top_level
      1,8693,38,Avant-Garde,38
      ...
    Returns dict genre_id_to_title: {genre_id -> title}
    """
    genre_id_to_title = {}
    with open(genres_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["genre_id"])
            title = row["title"]
            genre_id_to_title[gid] = title
    return genre_id_to_title

###############################################################################
# 2) Arg parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Folder with .npy embeddings. If train/val/test subfolders, use --recursive.")
    parser.add_argument("--recursive", action="store_true",
                        help="Search subfolders for .npy if set.")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "pca"],
                        help="Dim. reduction: tsne or pca.")
    parser.add_argument("--dim", type=int, default=2,
                        choices=[2, 3],
                        help="2D or 3D plot.")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max embeddings to visualize.")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity (ignored if method=pca).")

    parser.add_argument("--track_genres_csv", type=str, default="../data/fma_metadata/track_genres.csv",
                        help="CSV with track_id,genre_id columns.")
    parser.add_argument("--genres_csv", type=str, default="../data/fma_metadata/genres.csv",
                        help="CSV with genre_id, title columns.")
    return parser.parse_args()

def collect_npy_files(embeddings_dir, recursive=False):
    if recursive:
        pattern = os.path.join(embeddings_dir, "**", "*.npy")
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(embeddings_dir, "*.npy")
        files = glob.glob(pattern)
    return sorted(files)

###############################################################################
# 3) Load embeddings + map track_id => genre
###############################################################################
def load_embeddings(files, track_to_genre, genre_id_to_title, max_samples=10000):
    """
    For each .npy file:
      - parse track_id
      - map to genre_id => genre_title
      - average across time dimension => embedding
    Returns:
      embeddings: (N, D)
      track_ids : list of track IDs
      genres    : list of genre titles
    """
    embeddings = []
    track_ids  = []
    genres     = []

    count = 0
    for f in files:
        fname = os.path.basename(f)
        if "_" in fname:
            tid_str = fname.split("_", 1)[0]
        else:
            tid_str = fname.replace(".npy", "")
        track_id = int(tid_str.lstrip("0") or "0")

        # track -> genre_id -> genre_title
        gid = track_to_genre.get(track_id, -1)
        genre_title = genre_id_to_title.get(gid, "Unknown")

        arr = np.load(f)  # (T, D)
        mean_vec = arr.mean(axis=0)

        embeddings.append(mean_vec)
        track_ids.append(track_id)
        genres.append(genre_title)

        count += 1
        if count >= max_samples:
            break

    embeddings = np.array(embeddings)
    return embeddings, track_ids, genres

###############################################################################
# 4) Dimensionality reduction
###############################################################################
def reduce_dimensionality(embeddings, method="tsne", dim=2, perplexity=30):
    if method == "tsne":
        if dim == 2:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:
            reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    else:  # pca
        if dim == 2:
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=3, random_state=42)

    reduced = reducer.fit_transform(embeddings)
    return reduced

###############################################################################
# 5) Randomly pick 3 songs: 2 same genre (blue, green), 1 different genre (red)
###############################################################################
def pick_three_songs(track_ids, genres):
    """
    track_ids: list of ints
    genres   : list of strings (titles)
    Step:
     1) group track_ids by genre
     2) pick a random genre that has >=2 tracks
     3) pick 2 track_ids from that genre
     4) pick a different random genre that has >=1 track
     5) pick 1 track from that second genre
    returns (songA, songB, songC) => 3 track_ids
    """
    from collections import defaultdict

    # Build genre->list of track_ids
    genre_to_tracks = defaultdict(list)
    for tid, g in zip(track_ids, genres):
        genre_to_tracks[g].append(tid)

    # Filter out genres with <2 tracks
    valid_genres = [g for g, lst in genre_to_tracks.items() if len(lst) >= 2]
    if len(valid_genres) < 2:
        # Not enough data to pick 2 songs from same genre + 1 from different
        return None, None, None

    # Pick random genre that has at least 2 tracks
    first_genre = random.choice(valid_genres)
    # pick 2 track_ids from that genre
    track_candidates = genre_to_tracks[first_genre]
    if len(track_candidates) < 2:
        return None, None, None
    selected_two = random.sample(track_candidates, 2)  # 2 songs from same genre

    # pick a different genre
    diff_genres = [g for g in genre_to_tracks if g != first_genre and len(genre_to_tracks[g]) >= 1]
    if not diff_genres:
        return None, None, None
    second_genre = random.choice(diff_genres)
    # pick 1 track from that second genre
    track_candidates2 = genre_to_tracks[second_genre]
    selected_one = random.choice(track_candidates2)

    return selected_two[0], selected_two[1], selected_one

###############################################################################
# 6) Main
###############################################################################
def main():
    args = parse_args()

    # Load track->genre_id
    print(f"[Info] Loading track->genre from {args.track_genres_csv}")
    track_to_genre = {}
    with open(args.track_genres_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            gid = int(row["genre_id"])
            track_to_genre[tid] = gid

    # Load genre_id->title
    print(f"[Info] Loading genre_id->title from {args.genres_csv}")
    genre_id_to_title = {}
    with open(args.genres_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["genre_id"])
            title = row["title"]
            genre_id_to_title[gid] = title

    # Gather .npy files
    files = collect_npy_files(args.embeddings_dir, args.recursive)
    if not files:
        print("[Error] No .npy found.")
        return

    # Build embeddings
    print(f"[Info] Found {len(files)} .npy files. Loading up to {args.max_samples}")
    embeddings, track_ids, genres = load_embeddings(files, track_to_genre, genre_id_to_title, max_samples=args.max_samples)
    print(f"[Info] Using {len(embeddings)} embeddings. Dim={embeddings.shape[1]}")

    # Dim reduce
    print(f"[Info] Reducing to {args.dim}D via {args.method.upper()} (perplexity={args.perplexity if args.method=='tsne' else '--'})")
    reduced = reduce_dimensionality(embeddings, method=args.method, dim=args.dim, perplexity=args.perplexity)
    print("[Info] Reduction done.")

    # Randomly pick 3 songs (2 same genre => blue, green; 1 diff genre => red)
    songA, songB, songC = pick_three_songs(track_ids, genres)
    if not all([songA, songB, songC]):
        print("[Warning] Could not find 3 suitable songs to highlight. All embeddings remain gray.")
        # We'll color everything gray in that case
        highlight_map = {}
    else:
        print(f"[Highlight] Two same genre: {songA}, {songB}.  Different genre: {songC}.")

        # We'll store them in highlight_map => color
        # Blue, green, red
        highlight_map = {songA: "blue", songB: "green", songC: "red"}

    # Build color array
    # if track_id in highlight_map => highlight_map[track_id], else "lightgray"
    colors = []
    for tid in track_ids:
        c = highlight_map.get(tid, "lightgray")
        colors.append(c)

    fig_title = f"{args.method.upper()}({args.dim}D) - highlight 3 tracks"
    # Save logic
    save_dir = "embedding_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_basename = f"{args.method}_{args.dim}D_random_3highlight"
    save_path = os.path.join(save_dir, save_basename + ".png")

    # Plot
    if args.dim == 2:
        plt.figure(figsize=(10,7))
        plt.scatter(reduced[:,0], reduced[:,1], c=colors, alpha=0.7, s=20)
        plt.title(fig_title)
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:,0], reduced[:,1], reduced[:,2], c=colors, alpha=0.7, s=20)
        ax.set_title(fig_title)
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
        ax.set_zlabel("Dim3")
        plt.savefig(save_path, dpi=300)
        plt.show()

    print(f"[✅] Plot saved => {save_path}")

if __name__=="__main__":
    main()
