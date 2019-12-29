import numpy as np
#import plotly
#import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


# read machine sound
fs, s = aIO.read_audio_file("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\Oh Child.mp4")
duration = len(s) / float(fs)
# extract short term features and plot ZCR and Energy
[f, fn] = aF.feature_extraction(s, fs, int(fs * 0.050), int(fs * 0.050))
print(f)