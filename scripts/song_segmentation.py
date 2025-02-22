import os
import argparse
import torchaudio
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the folder containing full-length songs.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path where segmented clips will be saved.")
    parser.add_argument("--segment_length", type=float, default=25.0,
                        help="Segment length in seconds (default: 25s).")
    parser.add_argument("--overlap", type=float, default=5.0,
                        help="Overlap between consecutive segments (default: 5s).")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate (default: 16kHz for BEATs).")
    return parser.parse_args()

def segment_audio(audio_path, output_dir, segment_length=25.0, overlap=5.0, target_sr=16000):
    """
    Splits a full-length song into overlapping clips.

    Args:
        audio_path (str): Path to the full song.
        output_dir (str): Where to save segmented clips.
        segment_length (float): Duration of each segment in seconds.
        overlap (float): Overlap duration in seconds.
        target_sr (int): Target sample rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load full song
    waveform, source_sr = torchaudio.load(audio_path)

    # Convert stereo to mono (BEATs expects mono)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if source_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Get song ID from filename
    song_id = os.path.splitext(os.path.basename(audio_path))[0]

    # Compute segment start times
    segment_samples = int(segment_length * target_sr)
    overlap_samples = int(overlap * target_sr)
    step_size = segment_samples - overlap_samples  # Step size for sliding window

    total_samples = waveform.shape[1]  # Total number of samples in the song
    num_segments = (total_samples - overlap_samples) // step_size

    print(f"Processing '{audio_path}' -> {num_segments} segments.")

    # Save each segment
    for i in range(num_segments):
        start_sample = i * step_size
        end_sample = start_sample + segment_samples

        if end_sample > total_samples:  # Ensure we don't exceed audio length
            end_sample = total_samples

        segment = waveform[:, start_sample:end_sample]  # Extract segment

        segment_filename = f"{song_id}_seg{i:03d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)

        torchaudio.save(segment_path, segment, target_sr)
        print(f"Saved: {segment_path}")

def main():
    args = parse_args()

    # Process all songs in input directory
    for file in os.listdir(args.input_dir):
        if file.lower().endswith((".wav", ".mp3", ".flac")):
            song_path = os.path.join(args.input_dir, file)
            segment_audio(song_path, args.output_dir, args.segment_length, args.overlap, args.target_sr)

    print(f"Segmentation complete. Segmented clips saved in {args.output_dir}")

if __name__ == "__main__":
    main()
