import os
import argparse
import numpy as np
import torch
import torchaudio

import sys
sys.path.append('/data/class/cs175/iklimov/unilm/beats')

import BEATs  # Now you can import BEATs correctly
# ------------------------------------------------------------------------
# Official approach to loading the BEATs model
# ------------------------------------------------------------------------
from BEATs import BEATs, BEATsConfig

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
    model.eval()  # set to inference mode
    return model

def compute_beats_embeddings(audio_path, model, device, sr=16000, mono=True):
    """
    1) Load an audio file with torchaudio.
    2) Resample to `sr`.
    3) Convert to mono (if `mono=True`).
    4) Use model.extract_features(...) to get the audio representation.
    5) Return a (T x D) numpy array of embeddings.
    """

    # Load audio
    waveform, source_sr = torchaudio.load(audio_path)

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if source_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=sr)
        waveform = resampler(waveform)

    # waveform shape: [1, num_samples]
    # For BEATs, we need shape: [batch, num_samples], so let's add batch_dim = 1
    waveform = waveform.to(device)  # Move to CPU or GPU
    waveform = waveform.unsqueeze(0)  # (1, 1, num_samples)

    # No padding needed if the entire clip is valid
    # But we do create a boolean mask of shape [1, num_samples]
    padding_mask = torch.zeros((1, waveform.size(-1)), dtype=torch.bool, device=device)

    # Extract features
    with torch.no_grad():
        # extract_features returns a tuple: (features, _) or just (features),
        # depending on the model version. We'll assume it's (features, layer_results)
        # features[0] is the final-layer feature map: shape [1, T, D]
        features_tuple = model.extract_features(waveform, padding_mask=padding_mask)
        # The first element in the returned tuple is the final-layer representation
        # If it's just one element, do: features = features_tuple[0]
        # If it's two elements (features, layer_results), we want features.
        # We'll handle both cases safely:
        if isinstance(features_tuple, (list, tuple)) and len(features_tuple) >= 1:
            features = features_tuple[0]
        else:
            features = features_tuple  # If the model returns just the features

    # features shape: [1, T, D]
    features = features.squeeze(0)  # -> [T, D]
    return features.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory of segmented audio files (wav, flac, mp3).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store .npy embeddings.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the BEATs .pt file.")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Save checkpoint every N clips.")
    parser.add_argument("--resume_if_checkpoint_exists", type=bool, default=False,
                        help="Resume if previous state file found.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 2) Load BEATs model
    beats_model = load_beats_model(args.model_checkpoint)
    beats_model = beats_model.to(device)
    beats_model.eval()

    # 3) Gather audio files
    audio_files = sorted([
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    ])
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
                audio_path,
                model=beats_model,
                device=device,
                sr=16000,
                mono=True
            )  # shape: (T, D)

            # (Optional) L2 normalization
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embedding = embedding / norms

            # Save the result
            np.save(embedding_path, embedding)

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
