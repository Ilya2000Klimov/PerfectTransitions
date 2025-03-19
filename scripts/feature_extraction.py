import os
import argparse
import numpy as np
import torch
import librosa

import sys
sys.path.append('/data/class/cs175/iklimov/unilm/beats')
from BEATs import BEATs, BEATsConfig

############################################################
# 1) BEATs Model Loader
############################################################
def load_beats_model(checkpoint_path):
    """
    Load a pre-trained or fine-tuned BEATs model from a .pt checkpoint.
    Follows the official GitHub usage:
    https://github.com/microsoft/unilm/tree/master/beats
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # inference mode
    return model

############################################################
# 2) Audio -> Embeddings (Librosa-based)
############################################################
def compute_beats_embeddings(audio_path, model, device, sr=16000, mono=True):
    """
    1) Load an audio file with librosa.
    2) Resample to `sr` if not already.
    3) Convert to mono (if `mono=True`).
    4) Use model.extract_features(...) to get the audio representation.
    5) Return a (T x D) numpy array of embeddings.
    """
    y, source_sr = librosa.load(audio_path, sr=sr, mono=mono)
    print(f"[Debug] Loaded audio from {audio_path} with shape {y.shape} and sample rate {source_sr}")

    if y.ndim == 1:
        waveform = torch.from_numpy(y).unsqueeze(0)  # -> (1, num_samples)
    else:
        waveform = torch.from_numpy(y)

    waveform = waveform.to(device)

    # Create a padding mask [batch=1, time]
    padding_mask = torch.zeros((1, waveform.shape[-1]), dtype=torch.bool, device=device)

    with torch.no_grad():
        print(f"[Debug] Waveform shape: {waveform.shape}, Padding mask shape: {padding_mask.shape}")
        out_tuple = model.extract_features(waveform, padding_mask=padding_mask)
        print(f"[Debug] After model.extract_features(...) => {out_tuple}")

        if isinstance(out_tuple, (list, tuple)) and len(out_tuple) >= 1:
            feats = out_tuple[0]  # shape: (1, T, D)
        else:
            feats = out_tuple

    # Remove batch dimension => (T, D)
    return feats.squeeze(0).cpu().numpy()

############################################################
# 3) Arg Parsing
############################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory of segmented audio files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store .npy embeddings.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the BEATs .pt file.")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Save checkpoint every N clips.")
    parser.add_argument("--resume_if_checkpoint_exists", type=bool, default=False,
                        help="Resume if previous state file found.")
    return parser.parse_args()

############################################################
# 4) Main
############################################################
def main():
    args = parse_args()

    # 1) Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 2) Load BEATs model
    beats_model = load_beats_model(args.model_checkpoint).to(device)
    beats_model.eval()

    # 3) Gather audio files "song_segmentation" style
    audio_files = []
    for root, _, files in os.walk(args.data_dir):
        # Optional: Print a debug message like in 'song_segmentation.py'
        print(f"\n📂 Entering directory: {root}")

        for f in files:
            # If it's a valid audio extension
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                audio_path = os.path.join(root, f)
                audio_files.append(audio_path)

    # Sort them for consistent processing order
    audio_files.sort()
    print(f"[Info] Found {len(audio_files)} audio files in {args.data_dir}")

    # 4) Resume logic
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "feature_extract_state.pt")
    last_processed_index = 0

    if args.resume_if_checkpoint_exists and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        last_processed_index = state.get("last_processed_index", 0)
        print(f"[Resume] Found checkpoint. Resuming from index {last_processed_index}.")

    # 5) Main loop
    for idx in range(last_processed_index, len(audio_files)):
        audio_path = audio_files[idx]
        clip_id = os.path.splitext(os.path.basename(audio_path))[0]
        embedding_path = os.path.join(args.output_dir, f"{clip_id}.npy")

        # Skip if we already processed this clip
        if os.path.exists(embedding_path):
            continue

        # Extract embeddings
        try:
            embedding = compute_beats_embeddings(
                audio_path, model=beats_model, device=device,
                sr=16000, mono=True
            )
            # (Optional) L2 normalization
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embedding = embedding / norms

            # Save the result
            np.save(embedding_path, embedding)
            print(f"[Saved] => {embedding_path}")

        except Exception as e:
            print(f"[Error] {audio_path}: {e}")

        # Checkpoint every N files
        if (idx + 1) % args.save_freq == 0:
            torch.save({"last_processed_index": idx + 1}, checkpoint_path)
            print(f"[Checkpoint] at file index {idx + 1}")

    # 6) Final checkpoint
    torch.save({"last_processed_index": len(audio_files)}, checkpoint_path)
    print("[Done] Feature extraction complete!")

if __name__ == "__main__":
    main()
