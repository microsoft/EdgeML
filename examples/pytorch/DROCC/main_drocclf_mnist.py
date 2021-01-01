from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from data_process_scripts.process_mnist import MNIST_Dataset
from edgeml_pytorch.trainer.drocclf_trainer import DROCCLFTrainer, cal_precision_recall

class MNIST_LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, 1, bias=False)

    def forward(self, x):
        x = x.view(x.shape[0],1,28,28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):

    """Adjust learning rate during training.

    Parameters
    ----------
    epoch: Current training epoch.
    total_epochs: Total number of epochs for training.
    only_ce_epochs: Number of epochs for initial pretraining.
    learning_rate: Initial learning rate for training.
    """
    #We dont want to consider the only ce 
    #based epochs for the lr scheduler
    epoch = epoch - only_ce_epochs
    drocc_epochs = total_epochs - only_ce_epochs
    # lr = learning_rate
    if epoch <= drocc_epochs:
        lr = learning_rate * 0.01
    if epoch <= 0.80 * drocc_epochs:
        lr = learning_rate * 0.1
    if epoch <= 0.40 * drocc_epochs:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

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

def get_close_negs(test_loader):
    # Getting the Close Negs data for MNIST Digit 0 by
    # randomly masking 50% of the digit 0 test image pixels
    batch_idx = -1
    for data, target, _ in test_loader:
        batch_idx += 1
        data, target = data.to(device), target.to(device)
        data = data.to(torch.float)
        target = target.to(torch.float)
        #In our case , label of digit 0 is 1
        data_0 = data[target==1]
        aug1 = data_0.clone()
        indices = np.random.choice(np.arange(torch.numel(aug1)), replace=False,
                           size=int(torch.numel(aug1) * 0.4))
        aug1[np.unravel_index(indices, np.shape(aug1))] = torch.min(data)
        if batch_idx==0:
            close_neg_data = aug1
        else:
            close_neg_data = torch.cat((close_neg_data,aug1), dim=0)

    close_neg_data = close_neg_data.detach().cpu().numpy()
    close_neg_labels = np.zeros((close_neg_data.shape[0]))
    
    return CustomDataset(close_neg_data, close_neg_labels)

def main():
    #Load digit 0 and digit 1 data from MNIST
    dataset = MNIST_Dataset("data")
    train_loader, test_loader = dataset.loaders(batch_size=args.batch_size)
    closeneg_test_data = get_close_negs(test_loader)
    closeneg_test_loader = DataLoader(closeneg_test_data, args.batch_size, shuffle=True)

    model = MNIST_LeNet().to(device)
    model = nn.DataParallel(model)

    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")

    trainer = DROCCLFTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)

    if args.eval==0:
        # Training the model        
        trainer.train(train_loader, test_loader, closeneg_test_loader, args.lr, adjust_learning_rate, args.epochs,
            ascent_step_size=args.ascent_step_size, only_ce_epochs = args.only_ce_epochs)

        trainer.save(args.model_dir)

    else:
        if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
        _, pos_scores, far_neg_scores  = trainer.test(test_loader, get_auc=False)
        _, _, close_neg_scores  = trainer.test(closeneg_test_loader, get_auc=False)
        
        precision_fpr03, recall_fpr03 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.03)
        precision_fpr05, recall_fpr05 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.05)
        print('Test Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(
            precision_fpr03, recall_fpr03))
        print('Test Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(
            precision_fpr05, recall_fpr05))

if __name__ == '__main__':
    torch.set_printoptions(precision=5)
    
    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                    help='CIFAR10 normal class index')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=50, metavar='N',
                        help='Number of gradient ascent steps')                        
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')                        
    parser.add_argument('--mom', type=float, default=0.99, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=0.2, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',
                        help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    args = parser. parse_args()

    # settings
    #Checkpoint store path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    main()
