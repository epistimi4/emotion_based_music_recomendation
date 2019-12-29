from pytube import YouTube

yt=YouTube("https://www.youtube.com/watch?v=wdWF5nFuXnQ")
t=yt.streams.filter(only_audio=True).all()
t[0].download("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\")