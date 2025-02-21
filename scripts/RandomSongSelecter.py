import os
import random
class RandomSongSelecter:
    def __init__(self, directory):
        self.directory = directory  # Initialize song_list as an empty set
        self.song_list = []
        self.get_all_songs() # Assuming this method populates the songs
        self.picked_songs = set()  # Set to track picked songs
        self.remaining_songs = self.song_list.copy()  # Copy song_list to remaining_songs

    def get_all_songs(self):
        song_list = []
        for root, dir, files in os.walk(self.directory):
            for song in files:
                self.song_list.append(os.path.join(root, song))
        return song_list

    def get_random_song(self):
        if not self.remaining_songs:
            print("All song has been randomly chosen")
            return False
        chosen_song = random.choice(self.remaining_songs)
        self.remaining_songs.remove(chosen_song)
        self.picked_songs.add(chosen_song)
        return chosen_song

    #Return a list of all songs in randomized order, should be use for baseline Eval
    def get_random_song_list(self):
        random_song_list = []
        while self.remaining_songs:
            random_song_list.append(self.get_random_song())

        return random_song_list



