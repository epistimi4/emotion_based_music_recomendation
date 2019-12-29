from youtube_api import YouTubeDataAPI


api_key = 'AIzaSyB_GkfmuxNXFn-oW2DVYlLiJQ74f-VhaNM'
yt = YouTubeDataAPI(api_key)

def get_video_metadata(song):
    searches = yt.search(q=song,
                     max_results=1)
    yt.get_video_metadata_gen(searches[0]['video_id'])
    print(yt.get_video_metadata(searches[0]['video_id']))
    return ""



get_video_metadata('Martin Garrix & Dua Lipa - Scared to be lonely')