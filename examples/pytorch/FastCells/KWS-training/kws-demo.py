# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from threading import Thread
from queue import Queue
from sys import byteorder
from array import array
from struct import pack
from collections import Counter
import argparse

import pyaudio
import wave
 
import numpy as np
from edgeml_pytorch.graph.rnn import SRNN2
from scipy.io import wavfile
from python_speech_features import fbank
import torch
import time
import os
import pdb
 
from training_config import TrainingConfig
from train_classifier import create_model
 
CLASS_LABELS = {
    1: 'backward', 
    2: 'bed',
    3: 'bird',
    4: 'cat',
    5: 'dog',
    6: 'down',
    7: 'eight',
    8: 'five',
    9: 'follow',
    10: 'forward',
    11: 'four',
    12: 'go',
    13: 'happy',
    14: 'house',
    15: 'learn',
    16: 'left',
    17: 'marvin',
    18: 'nine',
    19: 'no',
    20: 'off',
    21: 'on',
    22: 'one',
    23: 'right',
    24: 'seven',
    25: 'sheila',
    26: 'six',
    27: 'stop',
    28: 'three',
    29: 'tree',
    30: 'two',
    31: 'up',
    32: 'visual',
    33: 'wow',
    34: 'yes',
    35: 'zero'
}

# Audio Recording Parameters
FORMAT = pyaudio.paInt16
RATE = 16000
 
# SRNN Parameters
maxlen = 16000
num_filt = 32
samplerate = 16000
winlen = 0.025
save_file = False
winstep = 0.010

winstepSamples = winstep * samplerate
winlenSamples = winlen * samplerate
numSteps = int(np.ceil((maxlen - winlenSamples)/winstepSamples) + 1)

# Streaming Prediction Parameters
num_windows = 10
majority = 5
stride = int(50 * (samplerate / 1000))
CHUNK_SIZE = stride
queue = Queue(10000000)


def extract_features(audio_data, data_len, num_filters,
                        sample_rate, window_len, window_step):
    """
    Returns MFCC features for input `audio_data`.
    """
    featurized_data = []
    eps = 1e-10
    for sample in audio_data:
        # temp = [num_steps, num_filters]
        temp, _ = fbank(sample, samplerate=sample_rate, winlen=window_len,
                        winstep=window_step, nfilt=num_filters,
                        winfunc=np.hamming)
        temp = np.log(temp + eps)
        featurized_data.append(temp)
    return np.array(featurized_data)
 
class RecordingThread(Thread):
    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=1, rate=RATE,
            input=True, output=True,
            frames_per_buffer=CHUNK_SIZE)
        global queue
        while True:
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            queue.put(snd_data)
        stream.stop_stream()
        stream.close()
        p.terminate()
 
class PredictionThread(Thread):
    def run(self):
        global queue
        global mean
        global std
        global fastgrnn
        global srnn2
        r = array('h')
        count = 0
        prev_class = 0
        srnn_votes = []
        fastgrnn_votes = []       
        while True:
            data = queue.get()
            queue.task_done()
            count += 1
            r.extend(data)
            if count < 21:
                continue
            
            r = r[stride:]
            if save_file:
                data = pack('<' + ('h'*len(r)), *r)
                save(data, 2, 'gen_sounds\cont'+str(count)+'.wav')
            data_np = np.array(r)
            data_np = np.expand_dims(data_np, 0)
            features = extract_features(data_np, numSteps, numFilt, samplerate, winlen, winstep)
            features = (features - mean) / std
            features = np.swapaxes(features, 0, 1)
 
            logits = fastgrnn(torch.FloatTensor(features))            
            _, y = torch.max(logits, dim=1)
            if len(fastgrnn_votes) == num_windows:
                fastgrnn_votes.pop(0)
                fastgrnn_votes.append(y.item())
            else:
                fastgrnn_votes.append(y.item())
            
            if count % 10 == 0:
                class_id = Counter(fastgrnn_votes).most_common(1)[0][0]
                class_freq = Counter(fastgrnn_votes).most_common(1)[0][1]
                if class_id != 0 and class_freq > 7 and prev_class != class_id:
                    try:
                        print('Keyword:', CLASS_LABELS[class_id])
                    except:
                        pass
                prev_class = class_id
 
def save(data, sample_width, path):
    """
    Saves audio `data` to given path. 
    """
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Simple Keyword Spotting Demo")
    parser.add_argument("--config_path", help="Path to config file", type=str)
    parser.add_argument("--model_path", help="Path to trained model", type=str)
    parser.add_argument("--mean_path", help="Path to train dataset mean", type=str)
    parser.add_argument("--std_path", help="Path to train dataset std", type=str)

    args = parser.parse_args()

    # FastGRNN Parameters
    config_path = args.config_path
    fastgrnn_model_path = args.model_path
    fastgrnn_mean_path = args.mean_path
    fastgrnn_std_path = args.std_path
    
    mean = np.load(fastgrnn_mean_path)
    std = np.load(fastgrnn_std_path)

    # Load FastGRNN
    config = TrainingConfig()
    config.load(config_path)
    fastgrnn = create_model(config.model, num_filt, 35)
    fastgrnn.load_state_dict(torch.load(fastgrnn_model_path, map_location=torch.device('cpu')))
    fastgrnn.normalize(None, None)
    
    # Start streaming prediction
    pred = PredictionThread()
    rec = RecordingThread()
    
    pred.start()
    rec.start()

    pred.join()
    rec.join()
