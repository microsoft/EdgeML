# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import librosa
import numpy as np
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_folder', default=None, required=True)
parser.add_argument('--target_folder', default=None, required=True)
parser.add_argument('--fs', type=int, default=16000)
parser.add_argument('--log_rate', type=int, default=1000)
args = parser.parse_args()

source_folder = args.source_folder
target_folder = args.target_folder
fs = args.fs
log_rate = args.log_rate
print(f'Source Folder :: {source_folder}\nTarget Folder :: {target_folder}\nSampling Frequency :: {fs}', flush=True)

source_files = []
target_files = []
list_completed = []

# Get the list of list of wav files from source folder and create target file names (full paths)
for i, f in enumerate(os.listdir(source_folder)):
  if f[-4:].lower() == '.wav':
    source_files.append(os.path.join(source_folder, f))
    target_files.append(os.path.join(target_folder, f))
print(f'Saved all the file paths, Number of files = {len(source_files)}', flush=True)

# Convert the files to args.fs
# Read with librosa and write the mono channel audio using soundfile
print(f'Converting all files to {fs/1000} Khz', flush=True)
for i, file_path in enumerate(source_files): 
  y, sr = librosa.load(file_path, sr=fs, mono=True)
  sf.write(target_files[i], y, sr)
  list_completed.append(target_files[i])
  if i % log_rate == 0:
    print(f'File Number {i+1}, Shape of Audio {y.shape}, Sampling Frequency {sr}', flush=True)

print(f'Number of Files saved {len(list_completed)}')
print('Done', flush=True)
