# Phoneme based Keyword Spotting(KWS)

# Project Description
There are two major issues in the existing KWS systems (a) they are not robust to heavy background noise and random utterances, and (b) they require collecting a lot of data, hampering the ease of adding a new keyword. Tackling these issues from a different perspective, we propose a new two staged scheme with a model for predicting phonemes which are in turn used for phoneme based keyword classification. 

First we train a phoneme classification model which gives the phoneme transcription of the input speech snippet. For training this phoneme classifier, we use a large public speech dataset like LibriSpeech. The public dataset can be aligned (meaning get the phoneme labels for each speech snippet in the data) using Montreal Forced Aligner. We also add reverberations and additive noise to the speech samples from the public dataset to make the phoneme classifier training robust to various accents, background noise and varied environment. In this project, we predict phonemes at every 10ms which is the standard way. You can find the aligned LibriSpeech dataset we used for training here.

In the second part, we use the predicted phoneme outputs from the phoneme classifier for predicting the input keyword. We train a 1 layer FastGRNN classifier to predict the keyword based on the phoneme transcription as input. Since the phoneme classifier training has been done to account for diverse accent, background noise and environments, the keyword classifier can be trained using a small number of Text-To-Speech(TTS) samples generated using any standard TTS api from cloud services like Azure, Google Cloud or AWS.

This gives two advantages: (a) The phoneme model is trained to account for diverse accents and background noise settings, thus the flexible keyword classifier training requires only a small number of keyword samples, and (b) Empirically this method was able to detect keywords from as far as 9ft of distance. Further, the phoneme model has a small size of around 250k parameters and can fit on a Cortex M7 micro-controller.

# Training the Phoneme Classifier
1) Train a phoneme classification model on some public speech dataset like LibriSpeech
2) Training speech dataset can be labelled using Montreal Force Aligner
3) Speech snippets are convolved with reverberation files, and additive noises from YouTube or other open source are added
4) We also add white gaussian noise of various SNRs

# Training the KWS Model
1) Our method takes as input the speech snippet and passes it through the phoneme classifier
2) Keywords are detected by training a keyword classifier over the detected phonemes
3) For training the keyword classifier, we use Azure and Google Text-To-Speech API to get the training data (keyword snippets)
4) For example, if you want to train a Keyword classifier for the keywords in the Google30 dataset, generate TTS samples from the Azure/Google-Cloud/AWS API for each of the 30 keywords. The TTS samples for each keyword must be stored in a separate folder named according to the keyword. More details about how the generated TTS data should be stored are mentioned below in sample use case for classifier model training.

# Sample Use Cases

## Phoneme Model Training
The following command can be used to instantiate and train the phoneme model.
```
python train_phoneme.py --base_path=/path/to/librispeech_data/ --rir_base_path=/path/to/reverb_files/ --additive_base_path=/path/to/additive_noises/ --snr_samples="0,5,10,25,100,100" --rir_chance=0.5 
```
Some important command line arguments:
1) base_path : Path of the speech data folder. The data in this folder should be in accordance to the dataloader code written here. 
2) rir_base_path, additive_base_path : Path to the reverb and additive noise files
3) snr_samples : List of various SNRs at which the additive noise is to be added.
4) rir_chance : Probability at which reverberation has to be done for each speech sample

## Classifier Model Training
The following command can be used to instantiate and train the classifier model.
```
python train_classifier.py --base_path=/path/to/train_and_test_data_folders/ --train_data_folders=google30_azure_tts,google30_google_tts --test_data_folders=google30_test --phoneme_model_load_ckpt=/path/to/checkpoint/x.pt --rir_base_path=/mnt/reverb_noise_sampled/ --additive_base_path=/mnt/add_noises_sampled/ --synth 
```
Some important command line arguments:

1) base_path : path to train and test data folders
2) train_data_folders, test_data_folders : These folders should have the .wav files for each keyword in a separate subfolder inside according to the dataloader here
3) phoneme_model_load_ckpt : The full path of the checkpoint file that would be used to load the weights to the instantiated phoneme model
4) rir_base_path, additive_base_path : Path to the reverb and additive noise files
5) synth : Boolean flag for specifying if reverberations and noise addition has to be done

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.