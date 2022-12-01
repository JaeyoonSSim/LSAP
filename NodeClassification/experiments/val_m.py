import numpy as np
import pandas as pd
import torch
import argparse
import sys
import os
import math
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

### Trainer for Ours
class Our_Trainer:
    def __init__(self, args, device, dataset):
        self.args = args
        self.device = device
        self.dataset = dataset
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        if self.dataset =='cora' or self.dataset == 'citeseer' or self.dataset == 'pubmed':
            self.A, self.X, self.y, self.idx_train, self.idx_val, self.idx_test = load_data_CCP(self.dataset)
        elif self.dataset == 'amazon-com' or self.dataset == 'amazon-photo' or self.dataset == 'coauthor-cs' or self.dataset == 'coauthor-phy':
            self.A, self.X, self.y, self.idx_train, self.idx_val, self.idx_test = load_data_AAC(self.dataset)
        print(self.idx_train.shape, self.idx_val.shape, self.idx_test.shape)
            
        self.model = ASAP2(adj_dim=self.A.shape[0],
                        in_dim=self.X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(self.y).item() + 1,
                        dropout=args.dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
        
        self.adj_size = self.A.shape[0]
        
        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(self.adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
            
        if torch.cuda.is_available():
            self.t = self.t.to(self.device)
            
        self.timings = []
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        
    def get_time(self):
        #self.timings=np.zeros((1,1))
        
        self.starter.record()
        
        self.eigenvalue, self.eigenvector, self.laplacian = compute_eigen_decomposition(self.A)
        
        if torch.cuda.is_available():
            if self.args.model != 'svm':
                self.model = self.model.to(self.device)
                self.A = self.A.to(self.device)
                self.X = self.X.to(self.device)
                self.y = self.y.to(self.device)
            
            self.eigenvalue = self.eigenvalue.to(self.device)
            self.eigenvector = self.eigenvector.to(self.device)
            self.laplacian = self.laplacian.to(self.device)
            
            self.idx_train = self.idx_train.to(self.device)
            self.idx_val = self.idx_val.to(self.device)
            self.idx_test = self.idx_test.to(self.device)
        

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
        scale_list = []
        heat_kernel_list = []
        
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
            self.get_time()
            
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
            
            self.optimizer.step() # Updates the parameters
            
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender)
            self.timings.append(curr_time)
            #print(curr_time)
            
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
            
            ### Delete
            del heat_kernel, heat_kernel_grad
            del output, loss_train, accuracy_train, loss_val, accuracy_val
            del curr_time
        
        #del self.timings[0:5]
        #print(len(self.timings))
        mean = sum(self.timings)/len(self.timings)
        vsum = 0
        for val in self.timings:
            vsum = vsum + (val - mean)**2
        variance = vsum / len(self.timings)
        #print(mean)
        std = math.sqrt(variance)
        #print(std)
        
        return mean, std
        

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

        return accuracy_test.item()



### Trainer for approximation version of 'Ours'
class Our_APPROX_Trainer:
    def __init__(self, args, device, dataset, poly, m):
        self.args = args
        self.device = device
        self.dataset = dataset
        self.poly = poly

        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        if self.poly == 0:
            self.m_che = m
        elif self.poly == 1:
            self.m_her = m
        elif self.poly == 2:
            self.m_lag = m
        
        if self.dataset =='cora' or self.dataset == 'citeseer' or self.dataset == 'pubmed':
            self.A, self.X, self.y, self.idx_train, self.idx_val, self.idx_test = load_data_CCP(self.dataset)
        elif self.dataset == 'amazon-com' or self.dataset == 'amazon-photo' or self.dataset == 'coauthor-cs' or self.dataset == 'coauthor-phy':
            self.A, self.X, self.y, self.idx_train, self.idx_val, self.idx_test = load_data_AAC(self.dataset)
        print(self.idx_train.shape, self.idx_val.shape, self.idx_test.shape)

        self.model = ASAP2(adj_dim=self.A.shape[0],
                        in_dim=self.X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(self.y).item() + 1,
                        dropout=args.dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
        
        self.eigenvalue, self.eigenvector, self.laplacian = compute_eigen_decomposition(self.A)
        
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #timings=np.zeros((1,1))
        self.timings = []
        
        if torch.cuda.is_available():
            if self.args.model != 'svm':
                self.model = self.model.to(self.device)
                self.A = self.A.to(self.device)
                self.X = self.X.to(self.device)
                self.y = self.y.to(self.device)
            
            self.eigenvalue = self.eigenvalue.to(self.device)
            self.eigenvector = self.eigenvector.to(self.device)
            self.laplacian = self.laplacian.to(self.device)
            
            self.idx_train = self.idx_train.to(self.device)
            self.idx_val = self.idx_val.to(self.device)
            self.idx_test = self.idx_test.to(self.device)
        
        self.adj_size = self.A.shape[0]
        
        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(self.adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        if torch.cuda.is_available():
            self.t = self.t.to(self.device)

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
        
        scale_list = []
        heat_kernel_list = []
        
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
            self.starter.record()
            
            self.model.train()
            
            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
        
            # Compute orthogonal polynomials
            if self.poly == 0: # Chebyshev
                eigmax = self.eigenvalue[-1]
                T_n = compute_Tn(self.laplacian, self.m_che, eigmax, self.device)
            elif self.poly == 1: # Hermite
                H_n = compute_Hn(self.laplacian, self.m_her, self.device)
            elif self.poly == 2: # Laguerre
                L_n = compute_Ln(self.laplacian, self.m_lag, self.device)

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
            
            self.optimizer.step() # Updates the parameters
            
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender)
            self.timings.append(curr_time)
            #print(curr_time)
            
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
            ### Delete
            if self.poly == 0: # Chebyshev
                del T_n
            elif self.poly == 1: # Hermite
                del H_n 
            elif self.poly == 2: # Laguerre
                del L_n
            del heat_kernel, heat_kernel_grad
            del output, loss_train, accuracy_train, loss_val, accuracy_val
            del curr_time
            
        #del self.timings[0:5]
        #print(len(self.timings))
        mean = sum(self.timings)/len(self.timings)
        vsum = 0
        for val in self.timings:
            vsum = vsum + (val - mean)**2
        variance = vsum / len(self.timings)
        #print(mean)
        std = math.sqrt(variance)
        #print(std)
        
        return mean, std

    ### Test
    def test(self):
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        if self.poly == 0: # Chebyshev
            eigmax = self.eigenvalue[-1]
            T_n = compute_Tn(self.laplacian, self.m_che, eigmax, self.device)
            heat_kernel, _ = compute_heat_kernel_chebyshev(T_n, self.m_che, self.t, eigmax, self.device, self.args.hk_threshold)
        elif self.poly == 1: # Hermite
            H_n = compute_Hn(self.laplacian, self.m_her, self.device)
            heat_kernel, _ = compute_heat_kernel_hermite(H_n, self.m_her, self.t, self.device, self.args.hk_threshold)
        elif self.poly == 2: # Laguerre
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

        return accuracy_test.item()
    
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
    parser.add_argument('--hidden_units', type=int, default=32, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 loss on parameters')
    ### Experiment for ours
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_lr', type=float, default=10, help='t learning rate')
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
    device4 = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    device5 = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    device6 = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device7 = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    
    dataset = args.data

    poly = 2
    
    trainer_ours0 = Our_Trainer(args, device0, dataset)
    _, _ = trainer_ours0.train()
    acc_exa = trainer_ours0.test()
    
    trainer_ours_approx1 = Our_APPROX_Trainer(args, device1, dataset, poly, 10)
    _, _ = trainer_ours_approx1.train()
    acc_m1 = trainer_ours_approx1.test()
    
    trainer_ours_approx2 = Our_APPROX_Trainer(args, device2, dataset, poly, 20)
    _, _ = trainer_ours_approx2.train()
    acc_m2 = trainer_ours_approx2.test()
    
    trainer_ours_approx3 = Our_APPROX_Trainer(args, device3, dataset, poly, 30)
    _, _ = trainer_ours_approx3.train()
    acc_m3 = trainer_ours_approx3.test()
    
    trainer_ours_approx4 = Our_APPROX_Trainer(args, device4, dataset, poly, 40)
    _, _ = trainer_ours_approx4.train()
    acc_m4 = trainer_ours_approx4.test()
    
    trainer_ours_approx5 = Our_APPROX_Trainer(args, device5, dataset, poly, 50)
    _, _ = trainer_ours_approx5.train()
    acc_m5 = trainer_ours_approx5.test()
    
    trainer_ours_approx6 = Our_APPROX_Trainer(args, device6, dataset, poly, 60)
    _, _ = trainer_ours_approx6.train()
    acc_m6 = trainer_ours_approx6.test()
    
    trainer_ours_approx7 = Our_APPROX_Trainer(args, device7, dataset, poly, 70)
    _, _ = trainer_ours_approx7.train()
    acc_m7 = trainer_ours_approx7.test()
    
    out = []
    out.append(abs(acc_exa-acc_m1))
    out.append(abs(acc_exa-acc_m2))
    out.append(abs(acc_exa-acc_m3))
    out.append(abs(acc_exa-acc_m4))
    out.append(abs(acc_exa-acc_m5))
    out.append(abs(acc_exa-acc_m6))
    out.append(abs(acc_exa-acc_m7))
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    
    #plt.rc('font',family='Times New Roman')
    plt.plot([10,20,30,40,50,60,70], out, linewidth=2, zorder = 2)
    #plt.title('Cora', fontweight='bold', fontsize=40, pad=10, fontname='Times New Roman')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    #plt.ylabel('ms', fontsize=30, fontweight='bold')
    plt.grid(True, axis='y', zorder=1)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(args.data + '_m2.png')


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")