#!/usr/bin/env python3
"""
build_track_genres.py

Creates a CSV "track_genres.csv" with columns: track_id,genre_id
by reading:

1) tracks.csv (the official FMA multi-level CSV)
   - For each track_id, we get track['track', 'genre_top'] => a string like "Rock", "Blues", "Jazz", ...
2) genres.csv
   - Contains lines like: genre_id,#tracks,parent,title,top_level
   - We'll parse "title" => "genre_id" mapping.

We then match the top-level genre string to the "title" in genres.csv
If found, we produce: track_id, genre_id. If not, we mark -1 or skip.

After this, you can feed "track_genres.csv" to scripts like "visualize_unified_embeddings.py"
for coloring embeddings by genre.

Example:
  python build_track_genres.py \
      --tracks_csv ../data/fma_metadata/tracks.csv \
      --genres_csv ../data/fma_metadata/genres.csv \
      --output_csv ../data/fma_metadata/track_genres.csv
"""

import os
import argparse
import pandas as pd
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_csv", type=str, required=True,
                        help="Path to FMA's tracks.csv (multi-level columns).")
    parser.add_argument("--genres_csv", type=str, required=True,
                        help="Path to FMA's genres.csv (with 'genre_id, ... , title').")
    parser.add_argument("--output_csv", type=str, default="track_genres.csv",
                        help="Output CSV name (default: track_genres.csv).")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Read genres.csv -> build: genre_title => genre_id
    print(f"[Info] Loading genres from {args.genres_csv}")
    genre_title_to_id = {}
    with open(args.genres_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row example: {"genre_id":"3", "#tracks":"1752", "parent":"0",
            #               "title":"Blues", "top_level":"3"}
            g_id = int(row["genre_id"])
            g_title = row["title"]
            genre_title_to_id[g_title] = g_id

    print(f"[Info] Found {len(genre_title_to_id)} genre titles in {args.genres_csv}")

    # 2) Read tracks.csv using Pandas, multi-level columns
    #    We'll focus on track['track', 'genre_top'] => string like "Blues", "Rock"
    print(f"[Info] Loading tracks data from {args.tracks_csv}")
    tracks_df = pd.read_csv(args.tracks_csv, index_col=0, header=[0,1])
    # The track_id is the DataFrame index
    # The top-level genre is at tracks_df[("track", "genre_top")]

    # 3) Build a mapping track_id => top_genre_str
    #    Then convert top_genre_str => genre_id if possible
    out_rows = []
    missing_count = 0
    matched_count = 0

    for track_id in tracks_df.index:
        # track_id is an integer like 14890
        # top_genre_str is e.g. "Blues" or "Rock"
        try:
            top_genre_str = tracks_df.loc[track_id, ("track", "genre_top")]
        except KeyError:
            # Possibly no column or data
            top_genre_str = None

        # If it's float('nan') or empty => skip
        if pd.isna(top_genre_str):
            top_genre_str = None

        if top_genre_str and isinstance(top_genre_str, str):
            # match to genre_id from genre_title_to_id
            if top_genre_str in genre_title_to_id:
                g_id = genre_title_to_id[top_genre_str]
                out_rows.append((track_id, g_id))
                matched_count += 1
            else:
                # top_genre_str not found in genres.csv => unknown
                out_rows.append((track_id, -1))
                missing_count += 1
        else:
            # no genre => -1
            out_rows.append((track_id, -1))
            missing_count += 1

    print(f"[Info] Matched {matched_count} tracks to known genres, {missing_count} unknown/missing")

    # 4) Write to output_csv: track_id,genre_id
    out_path = args.output_csv
    print(f"[Info] Writing {len(out_rows)} entries to {out_path}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "genre_id"])
        for (tid, gid) in out_rows:
            writer.writerow([tid, gid])

    print("[Done] build_track_genres.py completed!")

if __name__ == "__main__":
    main()