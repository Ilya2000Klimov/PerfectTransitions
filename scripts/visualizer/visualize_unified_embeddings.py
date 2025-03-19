#!/usr/bin/env python3
"""
visualize_unified_embeddings.py

1) Loads all .npy embeddings from a directory (including subdirectories if needed).
   This can be your entire dataset across train/val/test, or a single folder.
2) Aggregates them into a single space (average across time dimension),
   then applies t-SNE or PCA to reduce to 2D or 3D.
3) Plots all embeddings color-coded by song ID.
4) If user specifies --songA and/or --songB, those songs are highlighted in special colors,
   and all other songs are shown in a more neutral palette.
5) **Now also saves the resulting plot** to a folder named `embedding_visualizations/`.

Example usage:
  python visualize_unified_embeddings.py \
      --embeddings_dir /path/to/embeddings  \
      --method tsne \
      --dim 2 \
      --songA 014890 \
      --songB 008202 \
      --recursive
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Path to the folder with .npy embeddings. If you have train/val/test subfolders, pass the parent directory, and use --recursive if needed.")
    parser.add_argument("--recursive", action="store_true",
                        help="If set, will search all subfolders for *.npy (train/val/test).")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "pca"],
                        help="Dimensionality reduction method: tsne or pca.")
    parser.add_argument("--dim", type=int, default=2,
                        choices=[2, 3],
                        help="Output dimension for visualization: 2 or 3.")
    parser.add_argument("--songA", type=str, default=None,
                        help="First song ID to highlight distinctly, e.g. '014890'.")
    parser.add_argument("--songB", type=str, default=None,
                        help="Second song ID to highlight distinctly, e.g. '008202'.")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max number of embeddings to visualize (for performance).")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity (ignored if method=pca).")
    return parser.parse_args()

def collect_files(embeddings_dir, recursive=False):
    """
    Gathers all .npy files either directly in embeddings_dir or recursively.
    """
    if recursive:
        pattern = os.path.join(embeddings_dir, "**", "*.npy")
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(embeddings_dir, "*.npy")
        files = glob.glob(pattern)
    return sorted(files)

def load_all_embeddings(files, max_samples=10000):
    """
    For each .npy file:
      - parse the song ID from "songID_segXYZ.npy"
      - load the (T, D) array
      - average across time => shape (D,)
    Returns:
      embeddings: (N, D) array
      song_ids  : list of song IDs (strings)
      filenames : list of full paths or basenames
    """
    embeddings = []
    song_ids   = []
    used_files = []

    for f in files:
        fname = os.path.basename(f)
        # parse song ID from "014890_seg006.npy"
        if "_" in fname:
            sid = fname.split("_", 1)[0]
        else:
            sid = fname.replace(".npy", "")

        arr = np.load(f)  # shape (T, D)
        # average across time
        mean_vec = arr.mean(axis=0)

        embeddings.append(mean_vec)
        song_ids.append(sid)
        used_files.append(f)

        if len(embeddings) >= max_samples:
            break

    embeddings = np.array(embeddings)
    return embeddings, song_ids, used_files

def reduce_dimensionality(embeddings, method="tsne", dim=2, perplexity=30):
    """
    Applies t-SNE or PCA to reduce embeddings to 2 or 3 dims.
    """
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

    return reducer.fit_transform(embeddings)

def main():
    args = parse_args()

    print(f"[Info] Gathering files from {args.embeddings_dir} (recursive={args.recursive})...")
    files = collect_files(args.embeddings_dir, recursive=args.recursive)
    if not files:
        print("[Error] Found no .npy files.")
        return

    print(f"[Info] Found {len(files)} .npy files. Loading up to {args.max_samples} embeddings.")
    embeddings, song_ids, used_files = load_all_embeddings(files, max_samples=args.max_samples)
    print(f"[Info] Loaded {len(embeddings)} embeddings.  Dimension={embeddings.shape[1]}")

    # 2) Dimensionality reduction
    print(f"[Info] Reducing to {args.dim}D via {args.method.upper()} (perplexity={args.perplexity if args.method=='tsne' else '--'})")
    reduced = reduce_dimensionality(
        embeddings, method=args.method, dim=args.dim, perplexity=args.perplexity
    )
    print("[Info] Reduction complete.")

    # 3) Build color scheme
    highlight_songs = set()
    if args.songA: highlight_songs.add(args.songA)
    if args.songB: highlight_songs.add(args.songB)

    if highlight_songs:
        # If we highlight 2 songs => those get distinct colors (red, blue).
        # Others => lightgray
        colors = []
        for sid in song_ids:
            if sid == args.songA:
                colors.append("red")
            elif sid == args.songB:
                colors.append("blue")
            else:
                colors.append("lightgray")
    else:
        # color each distinct song ID
        import matplotlib.colors as mcolors
        unique_songs = sorted(list(set(song_ids)))
        palette = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())
        color_map = {song: i for i, song in enumerate(unique_songs)}

        def get_color(idx):
            return palette[idx % len(palette)]

        colors = [get_color(color_map[sid]) for sid in song_ids]

    fig_title = f"{args.method.upper()}({args.dim}D) of All Embeddings"
    if args.songA or args.songB:
        songs_to_show = " & ".join([s for s in [args.songA, args.songB] if s])
        fig_title += f" [Highlighting: {songs_to_show}]"

    # Create a folder to save the plot
    save_dir = "embedding_visualizations"
    os.makedirs(save_dir, exist_ok=True)

    # Construct a filename for saving
    # e.g. "tsne_2D.png" or "pca_3D_highlight_014890_008202.png"
    save_basename = f"{args.method}_{args.dim}D"
    if args.songA or args.songB:
        highlight_str = "_".join([s for s in [args.songA, args.songB] if s])
        save_basename += f"_highlight_{highlight_str}"
    save_path = os.path.join(save_dir, save_basename + ".png")

    # 4) Plot & Save
    if args.dim == 2:
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7, s=20)
        plt.title(fig_title)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:  # dim=3
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                   c=colors, alpha=0.7, s=20)
        ax.set_title(fig_title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

        plt.savefig(save_path, dpi=300)
        plt.show()

    print(f"[✅] Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
