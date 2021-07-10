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
from data_pipe import get_ASR_datasets

def parseArgs():
    """
    Parse the command line arguments
    Describes the architecture and the hyper-parameters
    """
    parser = argparse.ArgumentParser()
    # Args for Model Traning
    parser.add_argument('--phoneme_model_save_folder', type=str, default='./phoneme_model', help="Folder to save the checkpoint")
    parser.add_argument('--phoneme_model_load_ckpt', type=str, default=None, help="Checkpoint file to be loaded")
    parser.add_argument('--optim', type=str, default='adam', help="Optimizer to be used")
    parser.add_argument('--lr', type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--save_tick', type=int, default=1, help="Number of epochs to wait to save")
    parser.add_argument('--workers', type=int, default=-1, help="Number of workers. Give -1 for all workers")
    parser.add_argument("--gpu", type=str, default='0', help="GPU indices Eg: --gpu=0,1,2,3 for 4 gpus. -1 for CPU")

    # Args for DataLoader
    parser.add_argument('--base_path', type=str, required=True, help="Path of the speech data folder. The data in this folder should be in accordance to the dataloader code written here.")
    parser.add_argument('--rir_base_path', type=str, required=True, help="Folder with the reverbration files")
    parser.add_argument('--additive_base_path', type=str, required=True, help="Folder with additive noise files")
    parser.add_argument('--phoneme_text_file', type=str, required=True, help="Text files with pre-fixed phons")
    parser.add_argument('--pretraining_length_mean', type=int, default=6, help="Mean of the audio clips lengths")
    parser.add_argument('--pretraining_length_var', type=int, default=1, help="variance of the audio clip lengths")
    parser.add_argument('--pretraining_batch_size', type=int, default=256, help="Batch size for the pipeline")
    parser.add_argument('--snr_samples', type=str, default="0,5,10,25,100,100", help="SNR values for additive noise files")
    parser.add_argument('--wgn_snr_samples', type=str, default="5,10,15,100,100", help="SNR values for white gaussian noise")
    parser.add_argument('--gain_samples', type=str, default="1.0,0.25,0.5,0.75", help="Gain values for the processed signal")
    parser.add_argument('--rir_chance', type=float, default=0.25, help="Probability of performing reverbration")
    parser.add_argument('--synth_chance', type=float, default=0.5, help="Probability of pre-processing the signal with noise and reverb")
    parser.add_argument('--pre_phone_list', action='store_true', help="Use pre-fixed list of phonemes")
    # Args for Phoneme
    parser.add_argument('--phoneme_cnn_channels', type=int, default=400, help="Number od channels for the CNN layers")
    parser.add_argument('--phoneme_rnn_hidden_size', type=int, default=200, help="Number of RNN hidden states")
    parser.add_argument('--phoneme_rnn_layers', type=int, default=1, help="Number of RNN layers")
    parser.add_argument('--phoneme_rank', type=int, default=50, help="Rank of the CNN layers weights")
    parser.add_argument('--phoneme_fwd_context', type=int, default=15, help="RNN forward window context")
    parser.add_argument('--phoneme_bwd_context', type=int, default=9, help="RNN backward window context")
    parser.add_argument('--phoneme_phoneme_isBi', action='store_true', help="Use Bi-Directional RNN")
    parser.add_argument('--phoneme_num_labels', type=int, default=41, help="Number og phoneme labels")

    args = parser.parse_args()
    
    # Parse the gain and SNR values to a float format
    args.snr_samples = [int(samp) for samp in args.snr_samples.split(',')]
    args.wgn_snr_samples = [int(samp) for samp in args.wgn_snr_samples.split(',')]
    args.gain_samples = [float(samp) for samp in args.gain_samples.split(',')]
    
    # Fix the number of workers for the data Loader. If == -1 then use all possible workers
    if args.workers == -1:
        args.workers = multiprocessing.cpu_count()
    
    print(f"Args : {args}", flush=True)
    return args

def train_phoneme_model(args):
    """
    Train the Phoneme Model on the designated dataset
    The Dataset loader is defined in data_pipe.py
    Default dataset used is LibriSpeeech. Change the paths and file reader to change datasets

    args: args object (contains info about model and training)
    """
    # GPU Settings
    gpu_str = str()
    for gpu in args.gpu.split(','):
        gpu_str = gpu_str + str(gpu) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    use_cuda = torch.cuda.is_available() and (args.gpu != -1)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Instantiate model
    phoneme_model = kwscnn.DSCNN_RNN_Block(cnn_channels=args.phoneme_cnn_channels,
                                           rnn_hidden_size=args.phoneme_rnn_hidden_size, 
                                           rnn_num_layers=args.phoneme_rnn_layers,
                                           device=device, rank=args.phoneme_rank, 
                                           fwd_context=args.phoneme_fwd_context,
                                           bwd_context=args.phoneme_bwd_context,
                                           num_labels=args.phoneme_num_labels)

    # Transfer to specified device
    phoneme_model.to(device)
    phoneme_model = torch.nn.DataParallel(phoneme_model)
    model = {'name': phoneme_model.module.__name__, 'phoneme': phoneme_model}
    

    # Optimizer
    if args.optim == "adam":
        model['opt'] = torch.optim.Adam(model['phoneme'].parameters(), lr=args.lr)
    if args.optim == "sgd":
        model['opt'] = torch.optim.SGD(model['phoneme'].parameters(), lr=args.lr)

    # Load the specified checkpoint. 'phoneme_model_load_ckpt' must point to a checkpoint and not folder
    if args.phoneme_model_load_ckpt is not None:
        if os.path.exists(args.phoneme_model_load_ckpt):
            # Get the number from the phoneme checkpoint path
            start_epoch = args.phoneme_model_load_ckpt                # Temporarily store the full ckpt path
            start_epoch = start_epoch.split('/')[-1]            # retain only the *.pt from the path (Linux)
            start_epoch = start_epoch.split('\\')[-1]           # retain only the *.pt from the path (Windows)
            start_epoch = int(start_epoch.split('.')[0])        # retain the integers
            # Load Checkpoint
            latest_ckpt = torch.load(args.phoneme_model_load_ckpt, map_location=device)
            # Load specific state_dicts() and print the latest stats
            model['phoneme'].load_state_dict(latest_ckpt['phoneme_state_dict'])
            model['opt'].load_state_dict(latest_ckpt['opt_state_dict'])
            print(f"Checkpoint Stats : {latest_ckpt['train_stats']}", flush=True)
        else:
            raise ValueError("Invalid Checkpoint Path")
    else:
        start_epoch = 0

    # Instantiate dataloaders, essential variables and save folders
    train_dataset = get_ASR_datasets(args)
    train_loader = train_dataset.loader
    total_batches = len(train_loader)
    output_frame_rate = 3
    save_path = args.phoneme_model_save_folder
    os.makedirs(args.phoneme_model_save_folder, exist_ok=True)

    print(f"Pre Phone List {args.pre_phone_list}", flush=True)
    print(f"Start Epoch : {start_epoch}", flush=True)
    print(f"Device : {device}", flush=True)
    print(f"Output Frame Rate (multiple of 10ms): {output_frame_rate}", flush=True)
    print(f"Number of Batches: {total_batches}", flush=True)

    # Train Loop
    for epoch in range(start_epoch + 1, args.epochs):
        model['train_stats'] = {'loss': 0, 'predstd': 0, 'correct': 0, 'valid': 0}
        for features, label in train_loader:
            features = features.to(device)
            label = label.to(device)
            model['opt'].zero_grad()

            # Data-padding for bricking
            features = features.permute((0, 2, 1))  # NCL to NLC
            mod_len = features.shape[1]
            pad_len_mod = (output_frame_rate - mod_len % output_frame_rate) % output_frame_rate
            pad_len_feature = pad_len_mod
            pad_data = torch.zeros(features.shape[0], pad_len_feature,
                                   features.shape[2]).to(device)
            features = torch.cat((features, pad_data), dim=1)

            assert (features.shape[1]) % output_frame_rate == 0
            # Augmenting the label accordingly
            pad_len_label = pad_len_feature
            pad_data = torch.ones(label.shape[0], pad_len_label).to(device) * (-1)
            pad_data = pad_data.type(torch.long)
            label = torch.cat((label, pad_data), dim=1)

            # Get the posterior predictions and trim the labels to the same length as the predictions
            features = features.permute((0, 2, 1))  # NLC to NCL
            posteriors = model['phoneme'](features)
            N, C, L = posteriors.shape
            trim_label = label[:, ::output_frame_rate]  # 30ms frame_rate
            trim_label = trim_label[:, :L]

            # Permute and ready the final and pred labels values
            flat_posteriors = posteriors.permute((0, 2, 1))  # TO NLC
            flat_posteriors = flat_posteriors.reshape((-1, C))  # to [NL] x C
            flat_labels = trim_label.reshape((-1))

            _, idx = torch.max(flat_posteriors, dim=1)
            correct_count = (idx == flat_labels).detach().sum()
            valid_count = (flat_labels >= 0).detach().sum()

            # Loss and backward step
            loss_phoneme_model = torch.nn.functional.cross_entropy(flat_posteriors, 
                                                                   flat_labels, ignore_index=-1)
            loss_phoneme_model.backward()
            torch.nn.utils.clip_grad_norm_(model['phoneme'].parameters(), 10.0)
            model['opt'].step()

            # Stats
            pred_std = idx.to(torch.float32).std()

            model['train_stats']['loss'] += loss_phoneme_model.detach()
            model['train_stats']['correct'] += correct_count
            model['train_stats']['predstd'] += pred_std
            model['train_stats']['valid'] += valid_count

        if epoch % args.save_tick == 0:
            # Save the model
            torch.save({'phoneme_state_dict': model['phoneme'].state_dict(),
                        'opt_state_dict': model['opt'].state_dict(),
                        'train_stats' : model['train_stats']}, os.path.join(save_path, f'{epoch}.pt'))

        valid_frames = model['train_stats']['valid'].cpu()
        correct_frames = model['train_stats']['correct'].cpu().to(torch.float32)
        epoch_prestd = model['train_stats']['predstd'].cpu()

        avg_ce = model['train_stats']['loss'].cpu() / total_batches
        avg_err = 100 - 100.0 * (correct_frames / valid_frames)

        print(f"Summary for Epoch {epoch} for Model {model['name']}", flush=True)
        print(f"CE: {avg_ce}, ERR: {avg_err}, FRAMES {correct_frames} / {valid_frames}, PREDSTD: {epoch_prestd / total_batches}", flush=True)
    return

if __name__ == '__main__':
    args = parseArgs()
    train_phoneme_model(args)    