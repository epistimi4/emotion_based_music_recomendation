import numpy as np
#import plotly
#import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


# read machine sound
fs, s = aIO.read_audio_file("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\Oh Child.wav")
duration = len(s) / float(fs)
# extract short term features
[f, fn] = aF.feature_extraction(s, fs, fs * 0.10, fs * 0.10)
#[f, fn] = aF.feature_extraction(s, fs, 1, 1)
print(f)