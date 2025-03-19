#!/usr/bin/env python3
"""
visualize_embeddings_genre.py

Now colors all songs by their genre using two metadata files:
1) track_genres.csv => track_id -> genre_id
2) genres.csv       => genre_id -> title

Usage:
  python visualize_unified_embeddings.py \
      --embeddings_dir /path/to/embeddings  \
      --method tsne \
      --dim 2 \
      --recursive
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

###############################################################################
# 1) Read the CSV metadata to build two dictionaries:
#    track_id -> genre_id, and genre_id -> genre_title
###############################################################################
def load_track_genres(track_genres_path):
    """
    Reads track_genres.csv:
    track_id,genre_id
    e.g. 14890,3
    Returns dict: { track_id:int -> genre_id:int }
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
    Reads genres.csv:
    genre_id,#tracks,parent,title,top_level
    e.g. 3,1752,0,Blues,3
    Returns dict: { genre_id:int -> genre_title:str }
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
# 2) Main arguments and file collecting logic
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

    # Paths to the FMA metadata CSVs
    parser.add_argument("--track_genres_csv", type=str, default="../data/fma_metadata/track_genres.csv",
                        help="CSV containing track_id,genre_id columns.")
    parser.add_argument("--genres_csv", type=str, default="../data/fma_metadata/genres.csv",
                        help="CSV containing genre_id,...,title,... columns.")

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
# 3) Load embeddings, parse track_id from filename, then map to genre
###############################################################################
def load_embeddings_and_genres(files, track_to_genre, genre_id_to_title, max_samples=10000):
    """
    For each .npy file:
      - parse track_id from 'xxxxx_segYYY.npy'
      - map track_id -> genre_id -> genre_title
      - load the embedding (T, D), average across time => (D,)
    Returns:
      embeddings: (N, D)
      genres:     list of genre_title for each embedding
    """
    embeddings = []
    genres = []

    count = 0
    for f in files:
        fname = os.path.basename(f)
        # e.g. "014890_seg006.npy" => track_id=14890
        if "_" in fname:
            tid_str = fname.split("_", 1)[0]
        else:
            tid_str = fname.replace(".npy", "")
        track_id = int(tid_str.lstrip("0") or "0")  # remove leading zeros

        # Look up genre_id, then genre_title
        if track_id in track_to_genre:
            gid = track_to_genre[track_id]
            genre_title = genre_id_to_title.get(gid, "Unknown")
        else:
            genre_title = "Unknown"

        # Load embedding, average across time
        arr = np.load(f)
        mean_vec = arr.mean(axis=0)  # shape(D,)

        embeddings.append(mean_vec)
        genres.append(genre_title)

        count += 1
        if count >= max_samples:
            break

    embeddings = np.array(embeddings)
    return embeddings, genres

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
        from sklearn.decomposition import PCA
        if dim == 2:
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=3, random_state=42)

    reduced = reducer.fit_transform(embeddings)
    return reduced

###############################################################################
# 5) Plot & Save
###############################################################################
def main():
    args = parse_args()

    # 1) Load track->genre_id and genre_id->title
    print("[Info] Loading track->genre from:", args.track_genres_csv)
    track_to_genre = load_track_genres(args.track_genres_csv)
    print("[Info] Loading genre_id->title from:", args.genres_csv)
    genre_id_to_title = load_genres(args.genres_csv)

    # 2) Gather .npy files
    files = collect_npy_files(args.embeddings_dir, args.recursive)
    print(f"[Info] Found {len(files)} .npy files in {args.embeddings_dir}")

    # 3) Build embeddings & genre lists
    embeddings, genres = load_embeddings_and_genres(
        files, track_to_genre, genre_id_to_title, max_samples=args.max_samples
    )
    print(f"[Info] Using {len(embeddings)} embeddings. Dim={embeddings.shape[1]}")

    # 4) Dimensionality reduction
    print(f"[Info] Reducing to {args.dim}D via {args.method.upper()} (perplexity={args.perplexity})")
    reduced = reduce_dimensionality(
        embeddings, method=args.method, dim=args.dim, perplexity=args.perplexity
    )
    print("[Info] Reduction complete.")

    # Create output folder
    save_dir = "embedding_visualizations"
    os.makedirs(save_dir, exist_ok=True)

    # 5) Assign a color to each genre
    import matplotlib.colors as mcolors
    unique_genres = sorted(set(genres))
    palette = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())
    # Build genre->color mapping
    genre_color_map = {}
    for i, g in enumerate(unique_genres):
        genre_color_map[g] = palette[i % len(palette)]

    # Build color array
    colors = [genre_color_map.get(g, "gray") for g in genres]

    # 6) Plot & Save
    fig_title = f"{args.method.upper()}({args.dim}D) - Color by Genre"
    save_basename = f"{args.method}_{args.dim}D_byGenre"
    save_path = os.path.join(save_dir, save_basename + ".png")

    if args.dim == 2:
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7, s=20)
        plt.title(fig_title)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors, alpha=0.7, s=20)

        ax.set_title(fig_title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        plt.savefig(save_path, dpi=300)
        plt.show()

    print(f"[✅] Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
