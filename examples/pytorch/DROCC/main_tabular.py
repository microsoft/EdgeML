from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import numpy as np
from edgeml_pytorch.trainer.drocc_trainer import DROCCTrainer


class MLP(nn.Module):
    """
    Multi-layer perceptron with single hidden layer.
    """
    def __init__(self,
                 input_dim=2,
                 num_classes=1, 
                 num_hidden_nodes=20):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_hidden_nodes = num_hidden_nodes
        activ = nn.ReLU(True)
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.input_dim, self.num_hidden_nodes)),
            ('relu1', activ)]))
        self.size_final = self.num_hidden_nodes

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_final, self.num_classes))]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.labels[idx]), torch.tensor([0])

def load_data(path):
    train_data = np.load(os.path.join(path, 'train_data.npy'), allow_pickle = True)
    train_lab = np.ones((train_data.shape[0])) #All positive labelled data points collected
    test_data = np.load(os.path.join(path, 'test_data.npy'), allow_pickle = True)
    test_lab = np.load(os.path.join(path, 'test_labels.npy'), allow_pickle = True)

    ## preprocessing 
    mean=np.mean(train_data,0)
    std=np.std(train_data,0)
    train_data=(train_data-mean)/ (std + 1e-4)
    num_features = train_data.shape[1]

    test_data = (test_data - mean)/(std + 1e-4)
    print(train_data.shape, train_lab.shape, test_data.shape, test_lab.shape)
    return CustomDataset(train_data, train_lab), CustomDataset(test_data, 1 - test_lab), num_features

def main():
    train_dataset, test_dataset, num_features = load_data(args.data_path)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    
    model = MLP(input_dim=num_features, num_hidden_nodes=args.hd, num_classes=1).to(device)
    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")
    
    # Training the model
    trainer = DROCCTrainer(model, optimizer, args.inp_lamda, args.inp_radius, args.gamma, device)
    
    # Restore from checkpoint 
    if args.restore == 1:
        if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
    
    trainer.train(train_loader, test_loader, args.lr, args.epochs, metric='F1', ascent_step_size=0.001)

    trainer.save(args.model_dir)

if __name__ == '__main__':
    torch.set_printoptions(precision=5)
    
    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=50, metavar='N',
                        help='Number of gradient ascent steps')                        
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Num hidden nodes')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')                        
    parser.add_argument('--mom', type=float, default=0.99, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not')
    parser.add_argument('--inp_radius', type=float, default=0.2, metavar='N',
                        help='radius of the ball in input space')
    parser.add_argument('--inp_lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss for input layer')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--restore', type=int, default=1, metavar='N',
                        help='load model ')
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',
                        help='r to gamma * r projection')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    args = parser.parse_args()

    # settings
    #Checkpoint store path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    main()
