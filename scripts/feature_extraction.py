import os
import argparse
import numpy as np
import torch

# Import the completed utility functions from my_beats_model
from my_beats_model import load_beats_model, compute_beats_embeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to segmented audio clips.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store .npy embeddings.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path (local dir or HF ID) for Wav2Vec2/BEATs-Large.")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Save checkpoint every N clips.")
    parser.add_argument("--resume_if_checkpoint_exists", type=bool, default=False,
                        help="Whether to resume from a previous checkpoint if found.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beats_model = load_beats_model(args.model_checkpoint)
    # beats_model is a tuple (processor, model)
    # Move model to device
    beats_model = (beats_model[0], beats_model[1].to(device))
    beats_model[1].eval()

    # 2) Prepare file list
    audio_files = sorted([
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
        if f.endswith(".wav") or f.endswith(".flac") or f.endswith(".mp3")
    ])
    print(f"Found {len(audio_files)} audio files in {args.data_dir}")

    # 3) Load "last_processed_index" from checkpoint if resuming
    checkpoint_path = os.path.join(args.output_dir, "feature_extract_state.pt")
    last_processed_index = 0

    if args.resume_if_checkpoint_exists and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        last_processed_index = state.get("last_processed_index", 0)
        print(f"[Resume] Found checkpoint. Resuming from index {last_processed_index}.")

    # Create output dir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 4) Iterate over audio files
    for idx in range(last_processed_index, len(audio_files)):
        audio_path = audio_files[idx]
        clip_id = os.path.splitext(os.path.basename(audio_path))[0]

        embedding_path = os.path.join(args.output_dir, f"{clip_id}.npy")
        
        # Skip if this clip is already processed
        if os.path.exists(embedding_path):
            continue

        # Compute embeddings
        try:
            embedding = compute_beats_embeddings(
                audio_path,
                model_tuple=beats_model,
                device=device,
                sr=16000,
                mono=True
            )  # shape: (T, D)

            # L2 normalize if desired
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            # Avoid dividing by zero if there's a silent clip
            norms[norms == 0] = 1e-9  
            embedding = embedding / norms

            # Save to disk
            np.save(embedding_path, embedding)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

        # 5) Checkpoint progress every N clips
        if (idx + 1) % args.save_freq == 0:
            torch.save({"last_processed_index": idx + 1}, checkpoint_path)
            print(f"[Checkpoint] Saved state at index {idx + 1}")

    # Final checkpoint save
    torch.save({"last_processed_index": len(audio_files)}, checkpoint_path)
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()
