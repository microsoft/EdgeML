# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import re
import numpy as np
import torch
# Aux scripts
import kwscnn
import multiprocessing
from data_pipe import get_ASR_datasets, get_classification_dataset

def parseArgs():
    """
    Parse the command line arguments
    Describes the architecture and the hyper-parameters
    """
    parser = argparse.ArgumentParser()
    # Args for Model Traning
    parser.add_argument('--phoneme_model_load_ckpt', type=str, required=True, help="Phoneme checkpoint file to be loaded")
    parser.add_argument('--classifier_model_save_folder', type=str, default='./classifier_model', help="Folder to save the classifier checkpoint")
    parser.add_argument('--classifier_model_load_ckpt', type=str, default=None, help="Classifier checkpoint to be loaded")
    parser.add_argument('--optim', type=str, default='adam', help="Optimizer to be used")
    parser.add_argument('--lr', type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--save_tick', type=int, default=1, help="Number of epochs to wait to save")
    parser.add_argument('--workers', type=int, default=-1, help="Number of workers. Give -1 for all workers")
    parser.add_argument("--gpu", type=str, default='0', help="GPU indices Eg: --gpu=0,1,2,3 for 4 gpus. -1 for CPU")
    parser.add_argument("--word_model_name", default='google30', help="Name of the list of words used")
    parser.add_argument('--words', type=str, help="List of words to be used. This will be assigned in the code. User input will not affect the result")
    parser.add_argument("--is_training", action='store_true', help="True for training")
    parser.add_argument("--synth", action='store_true', help="Use Synth block or not")
    # Args for DataLoader
    parser.add_argument('--base_path', type=str, required=True, help="path to train and test data folders")
    parser.add_argument('--train_data_folders', type=str, default="google30_train", help="List of training folders in base path. Each folder is a dataset in the prescribed format")
    parser.add_argument('--test_data_folders', type=str, default="google30_test", help="List of testing folders in base path. Each folder is a dataset in the prescribed format")
    parser.add_argument('--rir_base_path', type=str, required=True, help="Folder with the reverbration files")
    parser.add_argument('--additive_base_path', type=str, required=True, help="Folder with additive noise files")
    parser.add_argument('--phoneme_text_file', type=str, help="Text files with pre-fixed phons")
    parser.add_argument('--pretraining_length_mean', type=int, default=6, help="Mean of the audio clips lengths")
    parser.add_argument('--pretraining_length_var', type=int, default=1, help="variance of the audio clip lengths")
    parser.add_argument('--pretraining_batch_size', type=int, default=256, help="Batch size for the pipeline")
    parser.add_argument('--snr_samples', type=str, default="-5,0,0,5,10,15,40,100,100", help="SNR values for additive noise files")
    parser.add_argument('--wgn_snr_samples', type=str, default="5,10,20,40,60", help="SNR values for white gaussian noise")
    parser.add_argument('--gain_samples', type=str, default="1.0,0.25,0.5,0.75", help="Gain values for processed signal")
    parser.add_argument('--rir_chance', type=float, default=0.9, help="Probability of performing reverbration")
    parser.add_argument('--synth_chance', type=float, default=0.9, help="Probability of pre-processing the input with reverb and noise")
    parser.add_argument('--pre_phone_list', action='store_true', help="use pre-fixed set of phonemes")
    # Args for Phoneme
    parser.add_argument('--phoneme_cnn_channels', type=int, default=400, help="Number od channels for the CNN layers")
    parser.add_argument('--phoneme_rnn_hidden_size', type=int, default=200, help="Number of RNN hidden states")
    parser.add_argument('--phoneme_rnn_layers', type=int, default=1, help="Number of RNN layers")
    parser.add_argument('--phoneme_rank', type=int, default=50, help="Rank of the CNN layers weights")
    parser.add_argument('--phoneme_fwd_context', type=int, default=15, help="RNN forward window context")
    parser.add_argument('--phoneme_bwd_context', type=int, default=9, help="RNN backward window context")
    parser.add_argument('--phoneme_phoneme_isBi', action='store_true', help="Use Bi-Directional RNN")
    parser.add_argument('--phoneme_num_labels', type=int, default=41, help="Number og phoneme labels")
    # Args for Classifier
    parser.add_argument('--classifier_rnn_hidden_size', type=int, default=100, help="Classifier RNN hidden dimensions")
    parser.add_argument('--classifier_rnn_num_layers', type=int, default=1, help="Classifier RNN number of layers")
    parser.add_argument('--classifier_dropout', type=float, default=0.2, help="Classifier dropout layer probability")
    parser.add_argument('--classifier_islstm', action='store_true', help="Use LSTM in the classifier")
    parser.add_argument('--classifier_isBi', action='store_true', help="Use Bi-Directional RNN in classifier")

    args = parser.parse_args()
    
    # Parse the gain and SNR values to a float format
    args.snr_samples = [int(samp) for samp in args.snr_samples.split(',')]
    args.wgn_snr_samples = [int(samp) for samp in args.wgn_snr_samples.split(',')]
    args.gain_samples = [float(samp) for samp in args.gain_samples.split(',')]

    # Fix the number of workers for the data Loader. If == -1 then use all possible workers
    if args.workers == -1:
        args.workers = multiprocessing.cpu_count()
    
    # Choose the word list to be used. For custom word lists, please add an elif condition
    if args.word_model_name == 'google30':
        args.words = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", 
            "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", 
            "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]
    elif args.word_model_name == 'google10':
        args.words = ["yes", "no", "up", "down", "left", "right", "on", "off",
            "stop", "go", "allsilence", "unknown"]
    else:
        raise ValueError('Incorrect Word Model Name')

    # The data-folder in args.base_path that contain the data
    # Refer to data_pipe.py for loading format
    args.train_data_folders = [folder_idx for folder_idx in args.train_data_folders.split(',')]
    args.test_data_folders  = [folder_idx for folder_idx in args.test_data_folders.split(',')]
    
    print(f"Args : {args}", flush=True)
    return args

def train_classifier_model(args):
    """
    Train the Classifier Model on the designated dataset
    The Dataset loader is defined in data_pipe.py
    Default dataset used is Google30. Change the paths and file reader to change datasets

    args: args object (contains info about model and training)
    """
    # GPU Settings
    gpu_str = str()
    for gpu in args.gpu.split(','):
        gpu_str = gpu_str + str(gpu) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    use_cuda = torch.cuda.is_available() and (args.gpu != -1)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Instantiate Phoneme Model
    phoneme_model = kwscnn.DSCNN_RNN_Block(cnn_channels=args.phoneme_cnn_channels,
                                           rnn_hidden_size=args.phoneme_rnn_hidden_size,
                                           rnn_num_layers=args.phoneme_rnn_layers,
                                           device=device, rank=args.phoneme_rank,
                                           fwd_context=args.phoneme_fwd_context,
                                           bwd_context=args.phoneme_bwd_context,
                                           num_labels=args.phoneme_num_labels)
    
    # Freeze Phoneme Model and Deactivate BatchNorm and Dropout Layers
    for param in phoneme_model.parameters():
        param.requires_grad = False
    phoneme_model.train(False)
    
    # Instantiate Classifier Model
    classifier_model = kwscnn.Binary_Classification_Block(in_size=args.phoneme_num_labels,
                                                          rnn_hidden_size=args.classifier_rnn_hidden_size, 
                                                          rnn_num_layers=args.classifier_rnn_num_layers,
                                                          device=device, islstm=args.classifier_islstm, 
                                                          isBi=args.classifier_isBi, dropout=args.classifier_dropout, 
                                                          num_labels=len(args.words))

    # Transfer to specified device
    phoneme_model.to(device)
    phoneme_model = torch.nn.DataParallel(phoneme_model)
    classifier_model.to(device)
    classifier_model = torch.nn.DataParallel(classifier_model)
    model = {'name': phoneme_model.module.__name__, 'phoneme': phoneme_model,
             'classifier_name': classifier_model.module.__name__, 'classifier': classifier_model}

    # Optimizer
    if args.optim == "adam":
        model['opt'] = torch.optim.Adam(model['classifier'].parameters(), lr=args.lr)
    if args.optim == "sgd":
        model['opt'] = torch.optim.SGD(model['classifier'].parameters(), lr=args.lr)

    # Load the specified phoneme checkpoint. 'phoneme_model_load_ckpt' must point to a checkpoint and not folder
    if args.phoneme_model_load_ckpt is not None:
        if os.path.exists(args.phoneme_model_load_ckpt):
            # Load Checkpoint
            latest_phoneme_ckpt = torch.load(args.phoneme_model_load_ckpt, map_location=device)
            # Load specific state_dicts() and print the latest stats
            print(f"Model Phoneme Location : {args.phoneme_model_load_ckpt}", flush=True)
            model['phoneme'].load_state_dict(latest_phoneme_ckpt['phoneme_state_dict'])
            print(f"Checkpoint Stats : {latest_phoneme_ckpt['train_stats']}", flush=True)
        else:
            raise ValueError("Invalid Phoneme Checkpoint Path")
    else:
        print("No Phoneme Checkpoint Given", flush=True)

    # Load the specified classifier checkpoint. 'classifier_model_load_ckpt' must point to a checkpoint and not folder
    if args.classifier_model_load_ckpt is not None:
        if os.path.exists(args.classifier_model_load_ckpt):
            # Get the number from the classifier checkpoint path
            start_epoch = args.classifier_model_load_ckpt             # Temporarily store the full ckpt path
            start_epoch = start_epoch.split('/')[-1]            # retain only the *.pt from the path (Linux)
            start_epoch = start_epoch.split('\\')[-1]           # retain only the *.pt from the path (Windows)
            start_epoch = int(start_epoch.split('.')[0])        # retain the integers
            # Load Checkpoint
            latest_classifier_ckpt = torch.load(args.classifier_model_load_ckpt, map_location=device)
            # Load specific state_dicts() and print the latest stats
            model['classifier'].load_state_dict(latest_classifier_ckpt['classifier_state_dict'])
            model['opt'].load_state_dict(latest_classifier_ckpt['opt_state_dict'])
            print(f"Checkpoint Stats : {latest_classifier_ckpt['train_stats']}", flush=True)
        else:
            raise ValueError("Invalid Classifier Checkpoint Path")
    else:
        start_epoch = 0

    # Instantiate all Essential Variables and utils
    train_dataset, test_dataset = get_classification_dataset(args)
    train_loader = train_dataset.loader
    test_loader  = test_dataset.loader
    total_batches = len(train_loader)
    output_frame_rate = 3
    save_path = args.classifier_model_save_folder
    os.makedirs(args.classifier_model_save_folder, exist_ok=True)
    # Print for cross-checking
    print(f"Pre Phone List {args.pre_phone_list}", flush=True)
    print(f"Start Epoch : {start_epoch}", flush=True)
    print(f"Device : {device}", flush=True)
    print(f"Output Frame Rate (multiple of 10ms): {output_frame_rate}", flush=True)
    print(f"Number of Batches: {total_batches}", flush=True)
    print(f"Synth: {args.synth}", flush=True)
    print(f"Words: {args.words}", flush=True)
    print(f"Optimizer : {model['opt']}", flush=True)

    # Train Loop
    for epoch in range(start_epoch + 1, args.epochs):
        model['train_stats'] = {'loss': 0, 'correct': 0, 'total': 0}
        model['classifier'].train(True)
        for train_features, train_label, train_seqlen in train_loader:
            train_seqlen_classifier = train_seqlen.clone() / output_frame_rate
            train_features = train_features.to(device)
            train_label = train_label.to(device)
            train_seqlen_classifier = train_seqlen_classifier.to(device)
            model['opt'].zero_grad()

            # Data-padding for bricking
            train_features = train_features.permute((0, 2, 1))  # NCL to NLC
            mod_len = train_features.shape[1]
            pad_len_mod = (output_frame_rate - mod_len % output_frame_rate) % output_frame_rate
            pad_len_feature = pad_len_mod
            pad_data = torch.zeros(train_features.shape[0], pad_len_feature,
                                   train_features.shape[2]).to(device)
            train_features = torch.cat((train_features, pad_data), dim=1)

            assert (train_features.shape[1]) % output_frame_rate == 0

            # Get the posterior predictions and trim the labels to the same length as the predictions
            train_features = train_features.permute((0, 2, 1))  # NLC to NCL
            train_posteriors = model['phoneme'](train_features)
            train_posteriors = model['classifier'](train_posteriors, train_seqlen_classifier)
            N, L, C = train_posteriors.shape

            # Permute and ready the final and pred labels values
            train_flat_posteriors = train_posteriors.reshape((-1, C))  # to [NL] x C

            # Loss and backward step
            train_label = train_label.type(torch.float32)
            loss_classifier_model = torch.nn.functional.binary_cross_entropy_with_logits(train_flat_posteriors, train_label)
            loss_classifier_model.backward()
            torch.nn.utils.clip_grad_norm_(model['classifier'].parameters(), 10.0)
            model['opt'].step()

            # Stats
            model['train_stats']['loss'] += loss_classifier_model.detach()
            _, train_idx_pred = torch.max(train_flat_posteriors, dim=1)
            _, train_idx_label = torch.max(train_label, dim=1)
            model['train_stats']['correct'] += float(np.sum((train_idx_pred == train_idx_label).detach().cpu().numpy()))
            model['train_stats']['total'] += train_idx_label.shape[0]

        if epoch % args.save_tick == 0:
            # Save the model
            torch.save({'classifier_state_dict': model['classifier'].state_dict(),
                        'opt_state_dict': model['opt'].state_dict(), 'train_stats' : model['train_stats']}, 
                        os.path.join(save_path, f'{epoch}.pt'))

        avg_ce = model['train_stats']['loss'].cpu() / total_batches
        train_accuracy = 100.0 * model['train_stats']['correct'] / model['train_stats']['total']
        print(f"Summary for Epoch {epoch} for Model {model['classifier_name']}; Loss: {avg_ce}", flush=True)
        print(f"TRAIN => Accuracy: {train_accuracy}; Correct: {model['train_stats']['correct']}; Total: {model['train_stats']['total']}", flush=True)

        model['test_stats'] = {'correct': 0,'total': 0}
        model['classifier'].eval()
        with torch.no_grad():
            for test_features, test_label, test_seqlen in test_loader:
                test_seqlen_classifier = test_seqlen.clone() / output_frame_rate
                test_features = test_features.to(device)
                test_label = test_label.to(device)
                test_seqlen_classifier = test_seqlen_classifier.to(device)

                # Data-padding for bricking
                test_features = test_features.permute((0, 2, 1))  # NCL to NLC
                mod_len = test_features.shape[1]
                pad_len_mod = (output_frame_rate - mod_len % output_frame_rate) % output_frame_rate
                pad_len_feature = pad_len_mod
                pad_data = torch.zeros(test_features.shape[0], pad_len_feature,
                                       test_features.shape[2]).to(device)
                test_features = torch.cat((test_features, pad_data), dim=1)

                assert (test_features.shape[1]) % output_frame_rate == 0

                # Get the posterior predictions and trim the labels to the same length as the predictions
                test_features = test_features.permute((0, 2, 1))  # NLC to NCL
                test_posteriors = model['phoneme'](test_features)
                test_posteriors = model['classifier'](test_posteriors, test_seqlen_classifier)
                N, L, C = test_posteriors.shape

                # Permute and ready the final and pred labels values
                test_flat_posteriors = test_posteriors.reshape((-1, C))  # to [NL] x C

                # Stats
                _, test_idx_pred = torch.max(test_flat_posteriors, dim=1)
                _, test_idx_label = torch.max(test_label, dim=1)
                model['test_stats']['correct'] += float(np.sum((test_idx_pred == test_idx_label).detach().cpu().numpy()))
                model['test_stats']['total'] += test_idx_label.shape[0]

            test_accuracy = 100.0 * model['test_stats']['correct'] / model['test_stats']['total']
            print(f"TEST => Accuracy: {test_accuracy}; Correct: {model['test_stats']['correct']}; Total: {model['test_stats']['total']}", flush=True)
    return

if __name__ == '__main__':
    args = parseArgs()
    train_classifier_model(args)    