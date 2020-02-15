import subprocess
import os


def find_all_files():
    path = "E:\\songs"
    music_files = [f for f in os.listdir(path) if f.endswith('mp4')]
    return music_files

def songDoesNotExist(song):
    path = "E:\\song_conv"
    text_files = [f for f in os.listdir(path) if f.startswith(song)]
    return len(text_files) == 0

def convert_to_wav(input_file, output_file):
    command = "C:\\ffmpeg\\bin\\ffmpeg.exe  -loglevel panic -i \""+ input_file + "\" -ab 160k \"" + output_file+"\" "
    print(command)
    subprocess.call(command, shell=True)
    return 0


music_files = find_all_files()
for file in music_files:
    input_file = "E:\\songs\\"+file
    output = "E:\\song_conv\\"+file.partition(".")[0] + ".wav"
    if songDoesNotExist(file.partition(".")[0]):
        convert_to_wav(input_file, output)
