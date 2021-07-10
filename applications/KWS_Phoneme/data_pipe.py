# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import argparse
import torch.utils.data
import os, glob, random
from collections import Counter
import soundfile as sf
import scipy.signal
import scipy.io.wavfile
import numpy as np
import textgrid
import multiprocessing
from subprocess import call
import librosa
import math
from numpy import random

def synthesize_wave(sigx, snr, wgn_snr, gain, do_rir, args):
    """
    Synth Block - Used to process the input audio.
    The input is convolved with room reverberation recording
    Adds noise in the form of white gaussian noise and regular audio clips (eg:piano, people talking, car engine etc)
    
    Input
        sigx    : input signal to the block
        snr     : signal-to-noise ratio of the input and additive noise (regular audio)
        wg_snr  : signal-to-noise ratio of the input and additive noise (white gaussian noise)
        gain    : gain of the output signal
        do_rir  : boolean flag, if reverbration needs to be incorporated
        args    : args object (contains info about model and training)
    
    Output
        clipped version of the audio post-processing
    """
    beta = np.random.choice([0.1, 0.25, 0.5, 0.75, 1])
    sigx = beta * sigx
    x_power = np.sum(sigx * sigx)
    
    # Do RIR and normalize back to original power
    if do_rir:
        rir_base_path = args.rir_base_path
        rir_fname = random.choice(os.listdir(rir_base_path))
        rir_full_fname = rir_base_path + rir_fname
        rir_sample, fs = sf.read(rir_full_fname)
        if rir_sample.ndim > 1:
            rir_sample = rir_sample[:,0]
        # We cut the tail of the RIR signal at 99% energy
        cum_en = np.cumsum(np.power(rir_sample, 2))
        cum_en = cum_en / cum_en[-1]
        rir_sample = rir_sample[cum_en <= 0.99]

        max_spike = np.argmax(np.abs(rir_sample))
        sigy = scipy.signal.fftconvolve(sigx, rir_sample)[max_spike:]
        sigy = sigy[0:len(sigx)]

        y_power = np.sum(sigy * sigy)
        sigy *= math.sqrt(x_power / y_power)  # normalize so y has same total power

    else:
        sigy = sigx

    y_rmse = math.sqrt(x_power / len(sigy))

    # Only bother with noise addition if the SNR is low enough
    if snr < 50:
        add_sample = get_add_noise(args)
        noise_rmse = math.sqrt(np.sum(add_sample * add_sample) / len(add_sample)) #+ 0.000000000000000001

        if len(add_sample) < len(sigy):
            padded = np.zeros(len(sigy), dtype=np.float32)
            padded[0:len(add_sample)] = add_sample
        else:
            padded = add_sample[0:len(sigy)] 

        add_sample = padded        
        
        noise_scale = y_rmse / noise_rmse * math.pow(10, -snr / 20)
        sigy = sigy + add_sample * noise_scale

    if wgn_snr < 50:
        wgn_samps = np.random.normal(size=(len(sigy))).astype(np.float32)
        noise_scale = y_rmse * math.pow(10, -wgn_snr / 20)
        sigy = sigy + wgn_samps * noise_scale

    # Apply gain & clipping
    return np.clip(sigy * gain, -1.0, 1.0)

def get_add_noise(args):
    """
    Extracts the additive noise file from the defined path
    
    Input
        args: args object (contains info about model and training)
    
    Output
        add_sample: additive noise audio
    """
    additive_base_path = args.additive_base_path
    add_fname = random.choice(os.listdir(additive_base_path))
    add_full_fname = additive_base_path + add_fname
    add_sample, fs = sf.read(add_full_fname)

    return add_sample

def get_ASR_datasets(args):
    """
    Function for preparing the data samples for the phoneme pipeline

    Input
        args: args object (contains info about model and training)
    
    Output
        train_dataset: dataset class used for loading the samples into the training pipeline
    """
    base_path = args.base_path

    # Load the speech data. This code snippet (till line 121) depends on the data format in base_path
    train_textgrid_paths = glob.glob(base_path + 
                                    "/text/train-clean*/*/*/*.TextGrid")
    
    train_wav_paths = [path.replace("text", "audio").replace(".TextGrid", ".wav") 
                        for path in train_textgrid_paths]

    if args.pre_phone_list:
        # If there is a list of phonemes in the dataset, use this flag
        Sy_phoneme = []
        with open(args.phoneme_text_file, "r") as f:
            for line in f.readlines():
                if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
        args.num_phonemes = len(Sy_phoneme)
        print("**************", flush=True)
        print("Phoneme List", flush=True)
        print(Sy_phoneme, flush=True)
        print("**************", flush=True)
        print("**********************", flush=True)
        print("Total Num of Phonemes", flush=True)
        print(len(Sy_phoneme), flush=True)
        print("**********************", flush=True)
    else:
        # No list of phonemes specified. Count from the input dataset
        phoneme_counter = Counter()
        for path in train_textgrid_paths:
            tg = textgrid.TextGrid()
            tg.read(path)
            phoneme_counter.update([phone.mark.rstrip("0123456789") 
                                    for phone in tg.getList("phones")[0] 
                                    if phone.mark not in ['', 'sp', 'spn']])

        # Display and store the phonemes extracted
        Sy_phoneme = list(phoneme_counter)
        args.num_phonemes = len(Sy_phoneme)
        print("**************", flush=True)
        print("Phoneme List", flush=True)
        print(Sy_phoneme, flush=True)
        print("**************", flush=True)
        print("**********************", flush=True)
        print("Total Num of Phonemes", flush=True)
        print(len(Sy_phoneme), flush=True)
        print("**********************", flush=True)
        with open(args.phoneme_text_file, "w") as f:
            for phoneme in Sy_phoneme:
                f.write(phoneme + "\n")

    print("Data Path Prep Done.", flush=True)

    # Create dataset objects
    train_dataset = ASRDataset(train_wav_paths, train_textgrid_paths, Sy_phoneme, args)

    return train_dataset

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, textgrid_paths, Sy_phoneme, args):
        """
        Dataset iterator for the phoneme detection model

        Input
            wav_paths       : list of strings (wav file paths)
            textgrid_paths  : list of strings (textgrid for each wav file)
            Sy_phoneme      : list of strings (all possible phonemes)
            args            : args object (contains info about model and training)
        """
        self.wav_paths = wav_paths
        self.textgrid_paths = textgrid_paths
        self.length_mean = args.pretraining_length_mean
        self.length_var = args.pretraining_length_var
        self.Sy_phoneme = Sy_phoneme
        self.args = args
        # Dataset Loader for the iterator
        self.loader = torch.utils.data.DataLoader(self, batch_size=args.pretraining_batch_size, 
                                                  num_workers=args.workers, shuffle=True, 
                                                  collate_fn=CollateWavsASR())

    def __len__(self):
        """ 
        Number of audio samples available
        """
        return len(self.wav_paths)

    def __getitem__(self, idx):
        """
        Gives one sample from the dataset. Data is read in this snippet. 
        (refer to the collate function for pre-processing)

        Input:
            idx: index for the sample
        
        Output:
            x           : audio sample obtained from the synth block (if used, else input audio) after time-domain clipping
            y_phoneme   : the output phonemes sampled at 30ms
        """
        x, fs = sf.read(self.wav_paths[idx])

        tg = textgrid.TextGrid()
        tg.read(self.textgrid_paths[idx])

        y_phoneme = []
        for phoneme in tg.getList("phones")[0]:
            duration = phoneme.maxTime - phoneme.minTime
            phoneme_index = self.Sy_phoneme.index(phoneme.mark.rstrip("0123456789")) if phoneme.mark.rstrip("0123456789") in self.Sy_phoneme else -1
            if phoneme.mark == '': phoneme_index = -1
            y_phoneme += [phoneme_index] * round(duration * fs)

        # Cut a snippet of length random_length from the audio
        random_length = round(fs * (self.length_mean + self.length_var * torch.randn(1).item()))
        if len(x) <= random_length:
            start = 0
        else:
            start = torch.randint(low=0, high=len(x)-random_length, size=(1,)).item()
        end = start + random_length

        x = x[start:end]

        if np.random.random() < self.args.synth_chance:
            x = synthesize_wave(x, np.random.choice(self.args.snr_samples),
                np.random.choice(self.args.wgn_snr_samples), np.random.choice(self.args.gain_samples),
                np.random.random() < self.args.rir_chance, self.args)

        self.phone_downsample_factor = 160
        y_phoneme = y_phoneme[start:end:self.phone_downsample_factor]

        # feature = librosa.feature.mfcc(x,sr=16000,n_mfcc=80,win_length=25*16,hop_length=10*16)

        return (x, y_phoneme)

class CollateWavsASR:
    def __call__(self, batch):
        """
        Pre-processing and padding, followed by batching the set of inputs

        Input:
            batch: list of tuples (input wav, phoneme labels)
        
        Output:
            feature_tensor      : the melspectogram features of the input audio. The features are padded for batching.
            y_phoneme_tensor    : the phonemes sequences in a tensor format. The phoneme sequences are padded for batching.
        """
        x = []; y_phoneme = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_,y_phoneme_, = batch[index]

            x.append(x_)
            y_phoneme.append(y_phoneme_)
            
        # pad all sequences to have same length and get features
        features=[]
        T = max([len(x_) for x_ in x])
        U_phoneme = max([len(y_phoneme_) for y_phoneme_ in y_phoneme])
        for index in range(batch_size):
            # pad audio to same length for all the audio samples in the batch
            x_pad_length = (T - len(x[index]))
            x[index] = np.pad(x[index], (x_pad_length,0), 'constant', constant_values=(0, 0))

            # Extract Mel-Spectogram from padded audio
            feature = librosa.feature.melspectrogram(y=x[index],sr=16000,n_mels=80,
                                                    win_length=25*16,hop_length=10*16, n_fft=512)
            
            feature = librosa.core.power_to_db(feature)
            # Normalize the features
            max_value = np.max(feature)
            min_value = np.min(feature)
            feature = (feature - min_value) / (max_value - min_value)
            features.append(feature)

            # Pad the labels to same length for all samples in the batch
            y_pad_length = (U_phoneme - len(y_phoneme[index]))
            y_phoneme[index] = np.pad(y_phoneme[index], (y_pad_length,0), 'constant', constant_values=(-1, -1))

        features_tensor = []; y_phoneme_tensor = []
        batch_size = len(batch)
        for index in range(batch_size):
            # x_,y_phoneme_, = batch[index]
            x_ = features[index]
            y_phoneme_ = y_phoneme[index]

            features_tensor.append(torch.tensor(x_).float())
            y_phoneme_tensor.append(torch.tensor(y_phoneme_).long())
            
        features_tensor = torch.stack(features_tensor)
        y_phoneme_tensor = torch.stack(y_phoneme_tensor)

        return (features_tensor,y_phoneme_tensor)

def get_classification_dataset(args):
    """
    Function for preparing the data samples for the classification pipeline

    Input
        args: args object (contains info about model and training)
    
    Output
        train_dataset   : dataset class used for loading the samples into the training pipeline
        test_dataset    : dataset class used for loading the samples into the testing pipeline
    """
    base_path = args.base_path

    # Train Data
    train_wav_paths = []
    train_labels = []

    # data_folder_list = ["google30_train"] or ["google30_azure_tts", "google30_google_tts"]
    data_folder_list = args.train_data_folders
    for data_folder in data_folder_list:
        # For each of the folder, iterate through the words and get the files and the labels
        for (label, word) in enumerate(args.words):
            curr_word_files = glob.glob(base_path + f"/{data_folder}/" + word + "/*.wav")
            train_wav_paths += curr_word_files
            train_labels += [label]*len(curr_word_files)
        print(f"Number of Train Files {len(train_wav_paths)}", flush=True)
    temp = list(zip(train_wav_paths, train_labels))
    random.shuffle(temp)
    train_wav_paths, train_labels = zip(*temp)
    print(f"Train Data Folders Used {data_folder_list}", flush=True)
    # Create dataset objects
    train_dataset = ClassificationDataset(wav_paths=train_wav_paths, labels=train_labels, args=args, is_train=True)

    # Test Data
    test_wav_paths = []
    test_labels = []

    # data_folder_list = ["google30_test"]
    data_folder_list = args.test_data_folders
    for data_folder in data_folder_list:
        # For each of the folder, iterate through the words and get the files and the labels
        for (label, word) in enumerate(args.words):
            curr_word_files = glob.glob(base_path + f"/{data_folder}/" + word + "/*.wav")
            test_wav_paths += curr_word_files
            test_labels += [label]*len(curr_word_files)
        print(f"Number of Test Files {len(test_wav_paths)}", flush=True)
    temp = list(zip(test_wav_paths, test_labels))
    random.shuffle(temp)
    test_wav_paths, test_labels = zip(*temp)
    print(f"Test Data Folders Used {data_folder_list}", flush=True)
    # Create dataset objects
    test_dataset = ClassificationDataset(wav_paths=test_wav_paths, labels=test_labels, args=args, is_train=False)

    return train_dataset, test_dataset

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths, labels, args, is_train=True):
        """
        Dataset iterator for the classifier model

        Input
            wav_paths   : list of strings (wav file paths)
            labels      : list of classification labels for the corresponding audio wav files
            is_train    : boolean flag, if the dataset loader is for the train or test pipeline
            args        : args object (contains info about model and training)
        """
        self.wav_paths = wav_paths
        self.labels = labels
        self.args = args
        self.is_train = is_train
        self.loader = torch.utils.data.DataLoader(self, batch_size=args.pretraining_batch_size, 
                                                  num_workers=args.workers, shuffle=is_train, 
                                                  collate_fn=CollateWavsClassifier())

    def __len__(self):
        """ 
        Number of audio samples available
        """
        return len(self.wav_paths)

    def one_hot_encoder(self, lab):
        """
        Label index to one-hot encoder

        Input:
            lab: label index
        
        Output:
            one_hot: label in the one-hot format
        """
        one_hot = np.zeros(len(self.args.words))
        one_hot[lab]=1
        return one_hot

    def __getitem__(self, idx):
        """
        Gives one sample from the dataset. Data is read in this snippet. (refer to the collate function for pre-processing)

        Input:
            idx: index for the sample
        
        Output:
            x               : audio sample obtained from the synth block (if used, else input audio) after time-domain clipping
            one_hot_label   : one-hot encoded label
            seqlen          : length of the audio file. 
                              This value will be dropped and seqlen after feature extraction will be used. Refer to the collate func
        """
        x, fs = sf.read(self.wav_paths[idx])

        label = self.labels[idx]
        one_hot_label = self.one_hot_encoder(label)
        seqlen = len(x)

        if self.is_train:
            # Use synth only for train files
            if self.args.synth:
                if np.random.random() < self.args.synth_chance:
                    x = synthesize_wave(x, np.random.choice(self.args.snr_samples),
                        np.random.choice(self.args.wgn_snr_samples), np.random.choice(self.args.gain_samples),
                        np.random.random() < self.args.rir_chance, self.args)

        return (x, one_hot_label, seqlen)

class CollateWavsClassifier:
    def __call__(self, batch):
        """
        Pre-processing and padding, followed by batching the set of inputs

        Input:
            batch: list of tuples (input wav, one hot classification label, sequence length)
        
        Output:
            feature_tensor          : the melspectogram features of the input audio. The features are padded for batching.
            one_hot_label_tensor    : the on-hot label in a tensor format
            seqlen_tensor           : the sequence length of the features in a minibatch
        """
        x = []; one_hot_label = []; seqlen = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_,one_hot_label_,_ = batch[index]
            x.append(x_)
            one_hot_label.append(one_hot_label_)
            
        # pad all sequences to have same length and get features
        features=[]
        T = max([len(x_) for x_ in x])
        T = max([T, 48000])
        for index in range(batch_size):
            # pad audio to same length for all the audio samples in the batch
            x_pad_length = (T - len(x[index]))
            x[index] = np.pad(x[index], (x_pad_length,0), 'constant', constant_values=(0, 0))
            
            # Extract Mel-Spectogram from padded audio
            feature = librosa.feature.melspectrogram(y=x[index],sr=16000,n_mels=80,win_length=25*16,hop_length=10*16, n_fft=512)
            feature = librosa.core.power_to_db(feature)
            # Normalize the features
            max_value = np.max(feature)
            min_value = np.min(feature)
            if min_value == max_value:
                feature = feature - min_value
            else:
                feature = (feature - min_value) / (max_value - min_value)           
            features.append(feature)
            seqlen.append(feature.shape[1])

        features_tensor = []; one_hot_label_tensor = []; seqlen_tensor = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_ = features[index]
            one_hot_label_ = one_hot_label[index]
            seqlen_ = seqlen[index]

            features_tensor.append(torch.tensor(x_).float())
            one_hot_label_tensor.append(torch.tensor(one_hot_label_).long())
            seqlen_tensor.append(torch.tensor(seqlen_))
            
        features_tensor = torch.stack(features_tensor)
        one_hot_label_tensor = torch.stack(one_hot_label_tensor)
        seqlen_tensor = torch.stack(seqlen_tensor)

        return (features_tensor,one_hot_label_tensor,seqlen_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help="Path of the speech data folder. The data in this folder should be in accordance to the dataloader code written here.")
    parser.add_argument('--train_data_folders', type=str, default="google30_train", help="List of training folders in base path. Each folder is a dataset in the prescribed format")
    parser.add_argument('--test_data_folders', type=str, default="google30_test", help="List of testing folders in base path. Each folder is a dataset in the prescribed format")
    parser.add_argument('--rir_base_path', type=str, required=True, help="Folder with the reverbration files")
    parser.add_argument('--additive_base_path', type=str, required=True, help="Folder with additive noise files")
    parser.add_argument('--phoneme_text_file', type=str, required=True, help="Text files with pre-fixed phons")
    parser.add_argument('--workers', type=int, default=-1, help="Number of workers. Give -1 for all workers")
    parser.add_argument("--word_model_name", default='google30', help="Name of the word list chosen. Will be used in conjunction with the data loader")
    parser.add_argument('--words', type=str, default="all")
    parser.add_argument("--synth", action='store_true', help="Use Synth block or not")
    parser.add_argument('--pretraining_length_mean', type=int, default=9)
    parser.add_argument('--pretraining_length_var', type=int, default=1)
    parser.add_argument('--pretraining_batch_size', type=int, default=64)
    parser.add_argument('--snr_samples', type=str, default="0,5,10,25,100,100")
    parser.add_argument('--wgn_snr_samples', type=str, default="5,10,15,100,100")
    parser.add_argument('--gain_samples', type=str, default="1.0,0.25,0.5,0.75")
    parser.add_argument('--rir_chance', type=float, default=0.25)
    parser.add_argument('--synth_chance', type=float, default=0.5)
    parser.add_argument('--pre_phone_list', action='store_true')
    args = parser.parse_args()

    # SNRs
    args.snr_samples = [int(samp) for samp in args.snr_samples.split(',')]
    args.wgn_snr_samples = [int(samp) for samp in args.wgn_snr_samples.split(',')]
    args.gain_samples = [float(samp) for samp in args.gain_samples.split(',')]

    # Workers
    if args.workers == -1:
        args.workers = multiprocessing.cpu_count()

    # Words
    if args.word_model_name == 'google30':
        args.words = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", 
            "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", 
            "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"
        ]
    elif args.word_model_name == 'google10':
        args.words = ["yes", "no", "up", "down", "left", "right", "on", "off",
            "stop", "go", "allsilence", "unknown"]
    else:
        raise ValueError('Incorrect Word Model Name')

    # Data Folders
    args.train_data_folders = [folder_idx for folder_idx in args.train_data_folders.split(',')]
    args.test_data_folders  = [folder_idx for folder_idx in args.test_data_folders.split(',')]

    print(args.pre_phone_list, flush=True)
    # get_ASR_datasets(args)
    dset = get_classification_dataset(args)
