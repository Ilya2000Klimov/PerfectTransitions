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
                        help="Segment length in seconds.")
    parser.add_argument("--overlap", type=float, default=5.0,
                        help="Overlap between consecutive segments.")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate (default: 16kHz for BEATs).")
    return parser.parse_args()


def segment_audio(audio_path, output_dir, segment_length=25.0, overlap=5.0, target_sr=16000):
    """
    Splits a full-length song into overlapping clips.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Try loading the audio; if it fails, we'll catch the exception.
    try:
        waveform, source_sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"  ✖ Failed to load '{audio_path}': {e}")
        return  # Skip processing this file

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if source_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Get filename base
    song_id = os.path.splitext(os.path.basename(audio_path))[0]

    # Calculate sample counts
    segment_samples = int(segment_length * target_sr)
    overlap_samples = int(overlap * target_sr)
    step_size = segment_samples - overlap_samples
    total_samples = waveform.shape[1]

    # How many full segments fit?
    num_segments = max(0, (total_samples - overlap_samples) // step_size)
    print(f"  ↳ Processing '{os.path.basename(audio_path)}' ({num_segments} segments)")

    for i in range(num_segments):
        start_sample = i * step_size
        end_sample = start_sample + segment_samples
        if end_sample > total_samples:
            end_sample = total_samples

        segment = waveform[:, start_sample:end_sample]
        segment_filename = f"{song_id}_seg{i:03d}.wav"
        segment_path = os.path.join(output_dir, segment_filename)

        # Save segment
        torchaudio.save(segment_path, segment, target_sr)
        print(f"    Saved: {segment_path}")


def main():
    torchaudio.set_audio_backend("sox_io")
    args = parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    skipped_count = 0
    skipped_dirs = []

    # Iterate over first-level entries in input_dir
    for entry in os.scandir(args.input_dir):
        # Check if this entry is actually a directory (following symlinks)
        if not entry.is_dir(follow_symlinks=True):
            skipped_count += 1
            skipped_dirs.append((entry.name, "Not a directory or symlink to directory"))
            continue

        # Prepare subfolder in the output path
        sub_output_dir = os.path.join(args.output_dir, entry.name)
        os.makedirs(sub_output_dir, exist_ok=True)

        # Get the list of audio files in this directory
        audio_files = [
            f for f in os.listdir(entry.path)
            if f.lower().endswith((".wav", ".mp3", ".flac"))
        ]
        if not audio_files:
            # No audio files to process, skip
            skipped_count += 1
            skipped_dirs.append((entry.name, "No audio files found"))
            continue

        # Process this directory
        print(f"\n📂 Entering directory: {entry.path}")
        for file in audio_files:
            song_path = os.path.join(entry.path, file)
            segment_audio(
                audio_path=song_path,
                output_dir=sub_output_dir,
                segment_length=args.segment_length,
                overlap=args.overlap,
                target_sr=args.target_sr
            )

    # Print skipped-directory summary
    print(f"\nSegmentation complete. Segmented clips saved in '{args.output_dir}'.")
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} directories:")
        for dname, reason in skipped_dirs:
            print(f" - {dname}: {reason}")


if __name__ == "__main__":
    main()
