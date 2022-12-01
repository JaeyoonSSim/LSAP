import numpy as np
import pandas as pd
import torch
import argparse
import sys
import os
import random
import wandb
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.utility import *
from utils.loader import *
from utils.utility import *
from utils.metric import *
from utils.approximate import *
from models.exact import *
from models.asap import *
from sklearn import metrics

### Trainer for 'Ours'
class Our_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, adj_size):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        self.laplacian = laplacian
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj_size = adj_size

        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        if torch.cuda.is_available():
            self.t = self.t.to(device)

    ### Scale regularization of loss function
    def t_loss(self):
        t_one = torch.abs(self.t)
        t_zero = torch.zeros_like(self.t)

        t_l = torch.where(self.t < self.t_loss_threshold, t_one, t_zero)

        return self.t_lambda * torch.sum(t_l)

    def t_deriv(self):
        t_one = self.t_lambda * torch.ones_like(self.t)
        t_zero = torch.zeros_like(self.t)

        t_de = torch.where(self.t < self.t_loss_threshold, -t_one, t_zero)

        return t_de

    def fir_deriv(self, output, feature, label, heat_kernel, heat_kernel_grad):
        y_oh = torch.zeros_like(output)
        y_oh.scatter_(1, label.reshape(-1, 1), 1)
        dl_ds = (torch.exp(output) - y_oh) / output.shape[0] # (# node, # class)
        
        dl_dl2 = torch.mul(dl_ds, self.model.gcn.rdp2) @ self.model.gcn.gc2.weight.T
        
        dl_first = torch.mul((dl_dl2 @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad)
        backward = torch.matmul(self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1))

        dl_second_tmp = torch.mul(dl_dl2, self.model.gcn.rdp)
        dl_second = torch.matmul(torch.mul(dl_second_tmp @ backward, heat_kernel_grad), heat_kernel.swapaxes(0, 1))

        dl_dt = dl_first + dl_second

        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=1)
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1))]).to(self.device)

        dl_dt += self.t_deriv()
        now_lr = self.t_lr * dl_dt

        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold

        self.t = self.t - now_lr # Update t

        #if self.args.use_t_local == 1:
        #    print(f't:{self.t[0].item()}', end=' ')
        #else:
        #    print(f't:{self.t.item():.4f}', end=' ')

    ### Train
    def train(self):
        hk_list = []
        w_c_list = []
        
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()

            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

            heat_kernel, heat_kernel_grad = compute_heat_kernel(self.eigenvalue, self.eigenvector, self.t, self.args.hk_threshold) # Use heat kernel instead of adjacency matrix

            # Use heat kernel instead of adjacency matrix
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)

            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train]) + self.t_loss()
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())
            
            if epoch > 2:
                if lt[-1] > lt[-2] * 5 and self.args.data == 'pubmed':
                    print('Epoch: ', best_epoch)
                    break
                elif lt[-1] > lt[-2] * 3 and (self.args.data == 'cora' or self.args.data == 'citeseer'):
                    print('Epoch: ', best_epoch)
                    break
                elif lt[-1] > lt[-2] * 4 and (self.args.data == 'amazon-photo' or self.args.data == 'amazon-com'):
                    print('Epoch: ', best_epoch)
                    break
            
            with torch.no_grad():
                self.fir_deriv(output, self.X, self.y, heat_kernel, heat_kernel_grad)
            
            #w_c = hk_filtering(self.X, self.eigenvalue, self.eigenvector, heat_kernel, self.device)
            #w_c_list.append(w_c.detach().cpu().numpy())
            #hk_list.append(heat_kernel.detach().cpu().numpy())
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:# or epoch < 400:
                    if lv[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_tr: {loss_train.item():.5f} acc_tr: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} acc_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    print('Epoch: ', best_epoch)
                    break
                
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
        
            self.model.eval()
            
        return self.t
        

    ### Test
    def test(self):
        self.model.eval()

        heat_kernel, heat_kernel_grad = compute_heat_kernel(self.eigenvalue, self.eigenvector, self.t, self.args.hk_threshold) # Use heat kernel instead of adjacency matrix

        # Use heat kernel instead of adjacency matrix
        output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
        
        loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test]) + self.t_loss()
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")




### Trainer for approximation version of 'Ours'
class Our_APPROX_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, adj_size, poly):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        self.laplacian = laplacian
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj_size = adj_size
        self.poly = poly

        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        self.m_che = args.m_chebyshev
        self.m_her = args.m_hermite
        self.m_lag = args.m_laguerre
        
        if torch.cuda.is_available():
            self.t = self.t.to(device)

    ### Scale regularization of loss function
    def t_loss(self):
        t_one = torch.abs(self.t)
        t_zero = torch.zeros_like(self.t)

        t_l = torch.where(self.t < self.t_loss_threshold, t_one, t_zero)

        return self.t_lambda * torch.sum(t_l)

    def t_deriv(self):
        t_one = self.t_lambda * torch.ones_like(self.t)
        t_zero = torch.zeros_like(self.t)

        t_de = torch.where(self.t < self.t_loss_threshold, -t_one, t_zero)

        return t_de

    def fir_deriv(self, output, feature, label, heat_kernel, heat_kernel_grad):
        y_real = torch.zeros_like(output) # (# node, # class)
        y_real.scatter_(1, label.reshape(-1, 1), 1) # (# node, # class)

        dl_dh = (torch.exp(output) - y_real) / output.shape[0] # (# node, # class)

        dl_dc_tmp = torch.mul(dl_dh, self.model.gcn.rdp2) @ self.model.gcn.gc2.weight.T # (# node, # hidden unit)

        dl_dc_first = torch.mul((dl_dc_tmp @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad) # (# node, # node)
        
        backward = torch.matmul(self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)) # (# hidden unit, # node)
        dl_dc_second = torch.matmul(torch.mul(torch.mul(dl_dc_tmp, self.model.gcn.rdp) @ backward, heat_kernel_grad), heat_kernel.swapaxes(0, 1)) # (# node, # node)

        dl_dt = dl_dc_first + dl_dc_second # (# node, # node)

        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=1) # (# node)
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1))]).to(self.device) # (1)

        dl_dt += self.t_deriv()
        now_lr = self.t_lr * dl_dt

        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold

        self.t = self.t - now_lr # Update t

    ### Train
    def train(self):
        w_c_list = []
        hk_list = []
        
        lt = [] # List of train loss
        at = [] # List of train accuracy

        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0
        
        # Compute orthogonal polynomials
        if self.poly == 0: # Chebyshev
            eigmax = self.eigenvalue[-1]
            T_n = compute_Tn(self.laplacian, self.m_che, eigmax, self.device)
        elif self.poly == 1: # Hermite
            H_n = compute_Hn(self.laplacian, self.m_her, self.device)
        elif self.poly == 2: # Laguerre
            L_n = compute_Ln(self.laplacian, self.m_lag, self.device)
            
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            
            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

            if self.poly == 0: # Chebyshev
                heat_kernel, heat_kernel_grad = compute_heat_kernel_chebyshev(T_n, self.m_che, self.t, eigmax, self.device, self.args.hk_threshold)
            elif self.poly == 1: # Hermite
                heat_kernel, heat_kernel_grad = compute_heat_kernel_hermite(H_n, self.m_her, self.t, self.device, self.args.hk_threshold)
            elif self.poly == 2: # Laguerre
                heat_kernel, heat_kernel_grad = compute_heat_kernel_laguerre(L_n, self.m_lag, self.t, self.device, self.args.hk_threshold)
            
            # Use heat kernel instead of adjacency matrix
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
            
            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train]) + self.t_loss()
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())
            
            #if epoch > 2:
            #    if lt[-1] > lt[-2] * 5 and self.args.data == 'pubmed':
            #        print('Epoch: ', best_epoch)
            #        break
            #    elif lt[-1] > lt[-2] * 3 and (self.args.data == 'cora' or self.args.data == 'citeseer'):
            #        print('Epoch: ', best_epoch)
            #        break
            #    elif lt[-1] > lt[-2] * 4 and (self.args.data == 'amazon-photo' or self.args.data == 'amazon-com'):
            #        print('Epoch: ', best_epoch)
            #        break
            
            if epoch > 2:
                if lt[-1] > lt[-2] * 4 and self.args.polynomial == 1:
                    print('Epoch: ', best_epoch)
                    break
            
            with torch.no_grad():
                self.fir_deriv(output, self.X, self.y, heat_kernel, heat_kernel_grad)
            
            #w_c = approx_hk_filtering(self.X, self.eigenvalue, self.eigenvector, heat_kernel, self.device)
            #w_c_list.append(w_c.detach().cpu().numpy())
            #hk_list.append(heat_kernel.detach().cpu().numpy())    
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:# or epoch < 400:
                    if lv[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1
                    
                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_tr: {loss_train.item():.5f} acc_tr: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} acc_val: {accuracy_val.item():.5f}")

                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    print('Epoch: ', best_epoch)
                    break
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
            
            self.model.eval()
        
        return self.t

    ### Test
    def test(self):
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        if self.args.polynomial == 0: # Chebyshev
            eigmax = self.eigenvalue[-1]
            T_n = compute_Tn(self.laplacian, self.m_che, eigmax, self.device)
            heat_kernel, _ = compute_heat_kernel_chebyshev(T_n, self.m_che, self.t, eigmax, self.device, self.args.hk_threshold)
        elif self.args.polynomial == 1: # Hermite
            H_n = compute_Hn(self.laplacian, self.m_her, self.device)
            heat_kernel, _ = compute_heat_kernel_hermite(H_n, self.m_her, self.t, self.device, self.args.hk_threshold)
        elif self.args.polynomial == 2: # Laguerre
            L_n = compute_Ln(self.laplacian, self.m_lag, self.device)
            heat_kernel, _ = compute_heat_kernel_laguerre(L_n, self.m_lag, self.t, self.device, self.args.hk_threshold)

        with torch.no_grad():
            # Use heat kernel instead of adjacency matrix
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
            
            loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test]) + self.t_loss()
            accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

#os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"

def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', type=str, default='cora', help='Type of dataset (cora / citeseer / pubmed)')
    ### Condition
    parser.add_argument('--seed_num', type=int, default=42, help='Number of random seed')
    parser.add_argument('--device_num', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--model', type=str, default='ours', help='Models to use')
    parser.add_argument('--val_mode', type=int, default=1, help='Validation mode')
    ### Experiment
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--hidden_units', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 loss on parameters')
    ### Experiment for ours
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_lr', type=float, default=1, help='t learning rate')
    parser.add_argument('--t_loss_threshold', type=float, default=0.01, help='t loss threshold')
    parser.add_argument('--t_lambda', type=float, default=1, help='t lambda of loss function')
    parser.add_argument('--t_threshold', type=float, default=0.1, help='t threshold')
    parser.add_argument('--hk_threshold', type=float, default=1e-5, help='Heat kernel threshold')
    ### Experiment for ours_approx
    parser.add_argument('--polynomial', type=int, default=2, help='Which polynomial is used (0:Chebyshev, 1:Hermite, 2:Laguerre)')
    parser.add_argument('--m_chebyshev', type=int, default=20, help='Expansion degree of Chebyshev')
    parser.add_argument('--m_hermite', type=int, default=30, help='Expansion degree of Hermite')
    parser.add_argument('--m_laguerre', type=int, default=20, help='Expansion degree of Laguerre')
    ### Etc
    parser.add_argument('--num_head_attentions', type=int, default=16, help='Number of head attentions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky relu')
    args = parser.parse_args()
    
    return args

### Control the randomness of all experiments
def set_randomness(seed_num):
    torch.manual_seed(seed_num) # Pytorch randomness
    np.random.seed(seed_num) # Numpy randomness
    random.seed(seed_num) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num) # Current GPU randomness
        torch.cuda.manual_seed_all(seed_num) # Multi GPU randomness

### Main function
def main():
    args = get_args()
    set_randomness(args.seed_num)
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    device3 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    dataset = args.data
    
    """
    A: adjacency (# nodes, # nodes)
    X: feature (# nodes, # features)
    y: label (# nodes)
    idx_train: 140 / 120 / 60
    idx_val: 500 / 500 / 500
    idx_test: 1000 / 1000 / 1000
    """
    if dataset =='cora' or dataset == 'citeseer' or dataset == 'pubmed':
        A, X, y, idx_train, idx_val, idx_test = load_data_CCP(dataset)
    elif dataset == 'amazon-com' or dataset == 'amazon-photo' or dataset == 'coauthor-cs' or dataset == 'coauthor-phy':
        A, X, y, idx_train, idx_val, idx_test = load_data_AAC(dataset)
    print(idx_train.shape, idx_val.shape, idx_test.shape)
    
    model_ours = Exact2(adj_dim=A.shape[0],
                    in_dim=X.shape[1],
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(y).item() + 1,
                    dropout=args.dropout_rate)
    optimizer_ours = optim.Adam(model_ours.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    
    model_ours_approx1 = ASAP2(adj_dim=A.shape[0],
                            in_dim=X.shape[1],
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(y).item() + 1,
                            dropout=args.dropout_rate)
    optimizer_ours_approx1 = optim.Adam(model_ours_approx1.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
    
    model_ours_approx2 = ASAP2(adj_dim=A.shape[0],
                            in_dim=X.shape[1],
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(y).item() + 1,
                            dropout=args.dropout_rate)
    optimizer_ours_approx2 = optim.Adam(model_ours_approx2.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
    
    model_ours_approx3 = ASAP2(adj_dim=A.shape[0],
                            in_dim=X.shape[1],
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(y).item() + 1,
                            dropout=args.dropout_rate)
    optimizer_ours_approx3 = optim.Adam(model_ours_approx3.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

    eigenvalue, eigenvector, laplacian = compute_eigen_decomposition(A)
    
    if torch.cuda.is_available():
        if args.model != 'svm':
            model_ours = model_ours.to(device0)
            model_ours_approx1 = model_ours_approx1.to(device1)
            model_ours_approx2 = model_ours_approx2.to(device2)
            model_ours_approx3 = model_ours_approx3.to(device3)
            A = A.to(device0)
            X = X.to(device0)
            y = y.to(device0)
            
            A1 = A.to(device1)
            X1 = X.to(device1)
            y1 = y.to(device1)
            
            A2 = A.to(device2)
            X2 = X.to(device2)
            y2 = y.to(device2)
            
            A3 = A.to(device3)
            X3 = X.to(device3)
            y3 = y.to(device3)
        
        eigenvalue = eigenvalue.to(device0)
        eigenvector = eigenvector.to(device0)
        laplacian = laplacian.to(device0)
        
        idx_train = idx_train.to(device0)
        idx_val = idx_val.to(device0)
        idx_test = idx_test.to(device0)
        
        eigenvalue1 = eigenvalue.to(device1)
        eigenvector1 = eigenvector.to(device1)
        laplacian1 = laplacian.to(device1)
        
        idx_train1 = idx_train.to(device1)
        idx_val1 = idx_val.to(device1)
        idx_test1 = idx_test.to(device1)
        
        eigenvalue2 = eigenvalue.to(device2)
        eigenvector2 = eigenvector.to(device2)
        laplacian2 = laplacian.to(device2)
        
        idx_train2 = idx_train.to(device2)
        idx_val2 = idx_val.to(device2)
        idx_test2 = idx_test.to(device2)
        
        eigenvalue3 = eigenvalue.to(device3)
        eigenvector3 = eigenvector.to(device3)
        laplacian3 = laplacian.to(device3)
        
        idx_train3 = idx_train.to(device3)
        idx_val3 = idx_val.to(device3)
        idx_test3 = idx_test.to(device3)
        

    #trainer_ours = Our_Trainer(args, device0, model_ours, optimizer_ours, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    #t_exact = trainer_ours.train().detach().cpu()
    #
    #trainer_ours_approx0 = Our_APPROX_Trainer(args, device1, model_ours_approx1, optimizer_ours_approx1, A1, X1, y1, eigenvalue1, eigenvector1, laplacian1, idx_train1, idx_val1, idx_test1, A1.shape[0], 0)
    #t_che = trainer_ours_approx0.train().detach().cpu()
    #
    #trainer_ours_approx1 = Our_APPROX_Trainer(args, device2, model_ours_approx2, optimizer_ours_approx2, A2, X2, y2, eigenvalue2, eigenvector2, laplacian2, idx_train2, idx_val2, idx_test2, A2.shape[0], 1)
    #t_her = trainer_ours_approx1.train().detach().cpu()
    #
    #trainer_ours_approx2 = Our_APPROX_Trainer(args, device3, model_ours_approx3, optimizer_ours_approx3, A3, X3, y3, eigenvalue3, eigenvector3, laplacian3, idx_train3, idx_val3, idx_test3, A3.shape[0], 2)
    #t_lag = trainer_ours_approx2.train().detach().cpu()
    
    ### Exact
    t = torch.empty(X.shape[0]).fill_(2.).to(device0)
    t_lambda = torch.mul(t, eigenvalue)
    g = torch.exp(-t_lambda)
    
    ### GapHeat-C
    fir = torch.zeros_like(laplacian).to(device0)
    sec = torch.eye(laplacian.shape[1]).to(device0)
    T_n = [sec.detach().cpu()]

    T_0 = (laplacian @ sec * 2) / eigenvalue[-1] - sec - fir
    T_0 = 0.5 * (T_0.T + T_0)
    fir, sec = sec, T_0

    ### Chebyshev polynomial T_n
    for n in range(1, args.m_chebyshev + 1):
        T_n.append(sec.detach().cpu())
        fir, sec = sec, chebyshev_recurrence(sec, fir, laplacian, eigenvalue[-1])
    
    #T_n = compute_Tn(laplacian, args.m_chebyshev, eigenvalue[-1], device0)

    che = torch.zeros_like(T_n[0])

    t = t.reshape(-1, 1) # (160, 1)
    z = t * eigenvalue[-1] / 2 # (160, 1)
    ez = torch.exp(-z) # (160, 1)
    deg = torch.arange(0, args.m_chebyshev + 1).to(device0) # (31)
    ivd = iv(deg.cpu(), z.cpu()).to(device0) # (160, 31)

    coeff = 2 * ((-1) ** deg) * ez * ivd # (160, 31)
    
    ### Chebyshev polynomial expansion of the solution to heat diffusion 
    for n in range(0, args.m_chebyshev + 1):
        coef = coeff[:,n].unsqueeze(dim=-1).detach().cpu()
        che += coef * T_n[n]
    
    che[che < args.hk_threshold] = 0
    che = torch.sum(che, axis=0)
    
    t_lambda = t_lambda.detach().cpu()
    eigenvalue = eigenvalue.detach().cpu()
    g = g.detach().cpu()
    print(T_n, len(T_n))
    
    plt.figure(figsize=(20,10))
    #ax1 = plt.subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.8)
    #plt.title('Pubmed', fontweight='bold', fontsize=22, pad=20)
    plt.plot(g, linewidth=3, linestyle='solid',  color='blue', label='Exact')
    plt.plot(che, linewidth=3, linestyle='solid',  color='orange', label='GapHeat-C')
    #plt.plot(y,  linestyle='dashed',  color='black', label='Initial scale')
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    #plt.xlabel('λ', fontsize=15, fontweight='bold')
    plt.ylabel('Kernel', fontsize=15, fontweight='bold')
    plt.grid(True)
    plt.legend(fontsize=15, frameon=True, shadow=True)
    #plt.legend(fontsize=10, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1), frameon=True, shadow=True)
    #ax2 = plt.subplot(2, 2, 2)
    ##plt.subplots_adjust(wspace=0.8)
    ##plt.title('Pubmed', fontweight='bold', fontsize=22, pad=20)
    #plt.plot(T_n, linestyle='solid',  color='orange', label='GapHeat-C')
    #plt.xticks(fontsize=12, fontweight='bold')
    #plt.yticks(fontsize=12, fontweight='bold')
    ##plt.ylabel('Scale')
    #plt.grid(True)
    #plt.legend(fontsize=10, frameon=True, shadow=True)
    #
    #ax1 = plt.subplot(2, 2, 3)
    ##plt.subplots_adjust(wspace=0.8)
    ##plt.title('Pubmed', fontweight='bold', fontsize=22, pad=20)
    #plt.plot(t_her, linestyle='solid',  color='green', label='GapHeat-H')
    #plt.plot(y,  linestyle='dashed',  color='black', label='Initial scale')
    #plt.xticks(fontsize=12, fontweight='bold')
    #plt.yticks(fontsize=12, fontweight='bold')
    ##plt.ylabel('Scale')
    #plt.grid(True)
    #plt.legend(fontsize=10, frameon=True, shadow=True)
    #
    #ax1 = plt.subplot(2, 2, 4)
    ##plt.subplots_adjust(wspace=0.8)
    ##plt.title('Pubmed', fontweight='bold', fontsize=22, pad=20)
    #plt.plot(t_lag, linestyle='solid',  color='red', label='GapHeat-L')
    #plt.plot(y,  linestyle='dashed',  color='black', label='Initial scale')
    #plt.xticks(fontsize=12, fontweight='bold')
    #plt.yticks(fontsize=12, fontweight='bold')
    ##plt.ylabel('Scale')
    #plt.grid(True)
    #plt.legend(fontsize=10, frameon=True, shadow=True)
    ##plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(args.data + '_scale.png')


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}") 