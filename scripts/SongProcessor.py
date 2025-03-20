import os
def get_all_songs():
    song_dir = os.path.join(os.getcwd(), '../songs')
    extensions = ("*.mp3", "*.wav", "*.flac", "*.m4a")
    audio_files = []

    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for ext in extensions:
            # Use glob to match specific file extensions
            audio_files.extend([os.path.join(dirpath, f) for f in filenames if f.endswith(ext)])
    print(audio_files)