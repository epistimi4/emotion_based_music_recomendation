from youtube_api import YouTubeDataAPI
from pytube import YouTube
import pandas as pd
import os

song_counter=0

api_key = 'AIzaSyB_GkfmuxNXFn-oW2DVYlLiJQ74f-VhaNM'
yt = YouTubeDataAPI(api_key)

def find_all_files():
    path = "C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data"
    text_files = [f for f in os.listdir(path) if f.startswith('user')]
    return text_files


def retrieve_song_titles_template1(filename):
    dfs = pd.read_excel(filename, sheet_name='Sheet1')
    return dfs['song-artist']

def retrieve_song_titles_template2(filename):
    return

def get_video_metadata(song):
    searches = yt.search(q=song,
                     max_results=1)
    yt.get_video_metadata_gen(searches[0]['video_id'])
    print(yt.get_video_metadata(searches[0]['video_id']))
    return searches[0]['video_id']

def get_video_id(song):
    searches = yt.search(q=song,
                     max_results=1)
    return searches[0]['video_id']

def download_song(video_id):
    global song_counter
    song_counter = song_counter +1
    url = "https://www.youtube.com/watch?v="+video_id
    try:
        yt = YouTube(url=url)
        t = yt.streams.filter(only_audio=True, file_extension='mp4').all()
        t[0].download("E:\\songs")
    except:
        print(str(song_counter)+" "+url)



text_files = find_all_files()
for file in text_files:
    songs = retrieve_song_titles_template1("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\"+file)
    for song in songs:
        video_id = get_video_id(song)
        download_song(video_id)
#get_video_metadata('My Angel-Stive Morgan')