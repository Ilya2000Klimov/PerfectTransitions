import SongExtracter
import os
def main():
    #set directories where songs are stored
    song_tr_dir = os.path.join('..', 'songs_tr')
    song_test_dir = os.path.join('..', 'songs_test')
    stft_save_dir = os.path.join('..', 'stft_tr')
    song_extensions = [".mp3"] #add more if we want to process more types of audio files
    #uses songs that are 30 seconds long
    SongExtracter.process_songs(song_tr_dir,  stft_save_dir, song_extensions, 30)



if __name__ == "__main__":
    main()