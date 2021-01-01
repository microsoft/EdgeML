import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def cal_precision_recall(positive_scores, far_neg_scores, close_neg_scores, fpr):
    """
    Computes the precision and recall for the given false positive rate.
    """
    #combine the far and close negative scores
    all_neg_scores = np.concatenate((far_neg_scores, close_neg_scores), axis = 0)
    num_neg = all_neg_scores.shape[0]
    idx = int((1-fpr) * num_neg)
    #sort scores in ascending order
    all_neg_scores.sort()
    thresh = all_neg_scores[idx]
    tp = np.sum(positive_scores > thresh)
    recall = tp/positive_scores.shape[0]
    fp = int(fpr * num_neg)
    precision = tp/(tp+fp)
    return precision, recall


def normalize_grads(grad):
    """
    Utility function to normalize the gradients.
    grad: (batch, -1)
    """
    # make sum equal to the size of second dim
    grad_norm = torch.sum(torch.abs(grad), dim=1)
    grad_norm = torch.unsqueeze(grad_norm, dim = 1)
    grad_norm = grad_norm.repeat(1, grad.shape[1])
    grad = grad/grad_norm * grad.shape[1]
    return grad

def compute_mahalanobis_distance(grad, diff, radius, device, gamma):
    """
    Compute the mahalanobis distance.
    grad: (batch,-1)
    diff: (batch,-1)
    """
    mhlnbs_dis = torch.sqrt(torch.sum(grad*diff**2, dim=1))
    #Categorize the batches based on mahalanobis distance
    #lamda = 1 : mahalanobis distance < radius
    #lamda = 2 : mahalanobis distance > gamma * radius
    lamda = torch.zeros((grad.shape[0],1))
    lamda[mhlnbs_dis < radius] = 1
    lamda[mhlnbs_dis > (gamma * radius)] = 2
    return lamda, mhlnbs_dis


# The following are utitlity functions for checking the conditions in
# Proposition 1 in https://arxiv.org/abs/2002.12718

def check_left_part1(lam, grad, diff, radius, device):
    #Part 1 condition value
    n1 = diff**2 * lam**2 * grad**2
    d1 = (1 + lam * grad)**2 + 1e-10
    term = n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_left_part2(nu, grad, diff, radius, device, gamma):
    #Part 2 condition value
    n1 = diff**2 * grad**2
    d1 = (nu + grad)**2 + 1e-10
    term = n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_right_part1(lam, grad, diff, radius, device):
    #Check if 'such that' condition is true in proposition 1 part 1
    n1 = grad
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum > radius**2:
        return check_left_part1(lam, grad, diff, radius, device)
    else:
        return np.inf

def check_right_part2(nu, grad, diff, radius, device, gamma):
    #Check if 'such that' condition is true in proposition 1 part 2
    n1 = grad*nu**2
    d1 = (nu + grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum < (gamma*radius)**2:
        return check_left_part2(nu, grad, diff, radius, device, gamma)
    else:
        # return torch.tensor(float('inf'))
        return np.inf

def range_lamda_lower(grad):
    #Gridsearch range for lamda
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    lam = -1 / lam + eps*0.0001
    return lam

def range_nu_upper(grad, mhlnbs_dis, radius, gamma):
    #Gridsearch range for nu
    alpha = (gamma*radius)/mhlnbs_dis
    max_sigma, _ = torch.max(grad, dim=1)
    nu = (alpha/(1-alpha))*max_sigma
    return nu

def optim_solver(grad, diff, radius, device, gamma=2):
    """
    Solver for the optimization problem presented in Proposition 1 in
    https://arxiv.org/abs/2002.12718
    """
    lamda, mhlnbs_dis = compute_mahalanobis_distance(grad, diff, radius, device, gamma)
    lamda_lower_limit = range_lamda_lower(grad).detach().cpu().numpy()
    nu_upper_limit = range_nu_upper(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()
    
    #num of values of lamda and nu samples in the allowed range
    num_rand_samples = 40 
    final_lamda =  torch.zeros((grad.shape[0],1))
    
    #Solve optim for each example in the batch
    for idx in range(lamda.shape[0]):
        #Optim corresponding to mahalanobis dis < radius
        if lamda[idx] == 1:
            min_left = np.inf
            best_lam = 0
            for k in range(num_rand_samples):
                val = np.random.uniform(low = lamda_lower_limit[idx], high = 0)
                left_val = check_right_part1(val, grad[idx], diff[idx], radius, device)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = best_lam
        
        #Optim corresponding to mahalanobis dis > gamma * radius
        elif lamda[idx] == 2:
            min_left = np.inf
            best_lam = np.inf
            for k in range(num_rand_samples):
                val = np.random.uniform(low = 0, high = nu_upper_limit[idx])
                left_val = check_right_part2(val, grad[idx], diff[idx], radius, device, gamma)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val
            
            final_lamda[idx] = 1.0/best_lam       

        else:
            final_lamda[idx] = 0

    final_lamda = final_lamda.to(device)
    for j in range(diff.shape[0]):
        diff[j,:] = diff[j,:]/(1+final_lamda[j]*grad[j,:])

    return diff

def get_gradients(model, device, data, target):
    """
    Utility function to compute the gradients of the model on the
    given data.
    """
    total_train_pts = len(data)
    data = data.to(torch.float)
    target = target.to(torch.float)
    target = torch.squeeze(target)

    #Extract the logits for cross entropy loss
    data_copy = data
    data_copy = data_copy.detach().requires_grad_()
    # logits = model(data_copy)
    logits = model(data_copy)
    logits = torch.squeeze(logits, dim = 1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)
    
    grad = torch.autograd.grad(ce_loss, data_copy)[0]

    return torch.abs(grad)

#trainer class for DROCC
class DROCCLFTrainer:
    """
    Trainer class that implements the DROCC-LF algorithm proposed for 
    one-class classification with limited negative data presented in    
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        """Initialize the DROCC-LF Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """     
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

    def train(self, train_loader, val_loader, closeneg_val_loader, learning_rate, lr_scheduler, total_epochs, 
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50):
        """Trains the model on the given training dataset with periodic 
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset with far negatives.
        closeneg_val_loader: Dataloader object for the validation dataset with close negatives.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial 
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial 
                          generation of negative points.
        """
        best_recall_fpr03 = -np.inf
        best_precision_fpr03 = -np.inf
        best_recall_fpr05 = -np.inf
        best_precision_fpr05 = -np.inf
        best_model = None
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        for epoch in range(total_epochs): 
            #Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)
            
            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_ce_loss = 0  #Cross entropy Loss
            
            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                self.optimizer.zero_grad()
                
                # Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 1]
                    target = torch.ones(data.shape[0]).to(self.device)
                    gradients = get_gradients(self.model, self.device, data, target)
                    # AdvLoss 
                    adv_loss = self.one_class_adv_loss(data, gradients)
                    epoch_adv_loss += adv_loss

                    loss = ce_loss + adv_loss * self.lamda
                else: 
                    # If only CE based training has to be done
                    loss = ce_loss
                
                # Backprop
                loss.backward()
                self.optimizer.step()
                    
            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss

            #normal val loader has the positive data and the far negative data
            auc, pos_scores, far_neg_scores  = self.test(val_loader, get_auc=True)
            _, _, close_neg_scores  = self.test(closeneg_val_loader, get_auc=False)
            
            precision_fpr03 , recall_fpr03 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.03)
            precision_fpr05 , recall_fpr05 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.05)
            if recall_fpr03 > best_recall_fpr03:
                best_recall_fpr03 = recall_fpr03
                best_precision_fpr03 = precision_fpr03
                best_recall_fpr05 = recall_fpr05
                best_precision_fpr05 = precision_fpr05
                best_model = copy.deepcopy(self.model)
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item()))
            print('Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(
                precision_fpr03, recall_fpr03))
            print('Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(
                precision_fpr05, recall_fpr05))
        self.model = copy.deepcopy(best_model)
        print('\nBest test Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(
            best_precision_fpr03, best_recall_fpr03
        ))
        print('\nBest test Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(
            best_precision_fpr05, best_recall_fpr05
        ))

    def test(self, test_loader, get_auc = True):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        """        
        label_score = []
        batch_idx = -1
        for data, target, _ in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = sigmoid_logits
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        pos_scores = scores[labels==1]
        neg_scores = scores[labels==0]
        auc = -1
        if get_auc:
            auc = roc_auc_score(labels, scores)
        return auc, pos_scores, neg_scores
        
    
    def one_class_adv_loss(self, x_train_data, gradients):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r) 
            classified as +ve (label=0). This is done by maximizing 
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R 
            (set N_i(r) with mahalanobis distance as a distance measure),
            by solving the optimization problem
        4) Pass the calculated adversarial points through the model, 
            and calculate the CE loss wrt target class 0
        
        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        gradients: gradients of the model for the given data.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)
                
                logits = self.model(x_adv_sampled)         
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm 
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 5==0:
                # Project the normal points to the set N_i(r) based on mahalanobis distance
                h = x_adv_sampled - x_train_data
                h_flat = torch.reshape(h, (h.shape[0], -1))
                gradients_flat = torch.reshape(gradients, (gradients.shape[0], -1))
                #Normalize the gradients 
                gradients_normalized = normalize_grads(gradients_flat)
                #Solve the non-convex 1D optimization
                h_flat = optim_solver(gradients_normalized, h_flat, self.radius, self.device, self.gamma)
                h = torch.reshape(h_flat, h.shape)
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path):
        torch.save(self.model.state_dict(),os.path.join(path, 'model.pt'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
