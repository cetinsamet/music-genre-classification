import os
import numpy as np
import matplotlib.pyplot as plt

from librosa import power_to_db
from librosa.core import load, to_mono
from librosa.feature import melspectrogram
from librosa.display import specshow


#datapath = '/Users/cetinsamet/Desktop/git/music-genre-classification/data/'

#for audio in os.listdir(datapath):
#audiopath   = datapath+audio

audiopath   = '../data/a.mp3'
y, sr       = load(audiopath)
y           = to_mono(y)
S           = melspectrogram(y, sr).T

S           = S[:-1*(S.shape[0]%128)]
num_chunk   = S.shape[0]/128
data_chunks = np.split(S, num_chunk)

#print(data_chunks[0])

plt.figure(figsize=(10, 4))
specshow(power_to_db(data_chunks[0]))
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()





