import librosa
import numpy as np
import os
def process_songs(song_dir, save_dir, extensions, song_fixed_length):
    all_files = os.listdir(song_dir)
    all_songs = [song for song in all_files if any(song.endswith(ext) for ext in extensions)]


    song_dir_path = os.path.join(".." , "songs")
    for song in all_songs:
        song_path = os.path.join(song_dir, song)
        y, sr = librosa.load(song_path, sr=None, mono=True)
        if len(y)%2 != 0:
            y = np.pad(y, (0,1), mode='constant')
        song_no_ext, ext = os.path.splitext(song)
        process_song(y, sr, song_fixed_length, save_dir, song_no_ext)

def process_song(y, sr, song_length, save_dir, file_name, n_fft=2048, hop_length=512):

    if len(y) % 2 != 0:
        raise ValueError("Song length is not even")


    #split the song into two parts from the middle
    first_halves = y[:len(y)//2]
    second_halves = y[len(y)//2:]

    #convert them into stft
    first_halves_stft = librosa.stft(first_halves, n_fft=n_fft, hop_length=hop_length)
    second_halves_stft = librosa.stft(second_halves, n_fft=n_fft, hop_length=hop_length)

    #make directories
    first_halves_save_dir = os.path.join(save_dir, "first_halves")
    second_halves_save_dir = os.path.join(save_dir, "second_halves")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(first_halves_save_dir, exist_ok=True)
    os.makedirs(second_halves_save_dir, exist_ok=True)

    np.save(os.path.join(first_halves_save_dir, f"{file_name}_first_halves_stft.npy"), first_halves_stft)
    np.save(os.path.join(second_halves_save_dir, f"{file_name}_second_halves_stft.npy"), second_halves_stft)











