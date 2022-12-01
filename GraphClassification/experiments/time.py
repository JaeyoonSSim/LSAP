import numpy as np
import pandas as pd
import random
import torch
import argparse
import time
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

from models.exact import *
from models.asap import *
from utils.utility import *
from utils.metric import *
from utils.approximate import *
#wandb.init(project="ADNI1", entity="jaeyoonsim")

from torch.linalg import eigh
from os import walk
from models.gdc import gdc

### Preprocess the feature file by sorting and removing unnecessary columns
def preprocess_feature_table(args, filename):
    feature = pd.read_excel(args.data_path + '/' + filename, engine='openpyxl')
    
    ### Sorting
    if 'Subject' in feature.columns: # If there is subject name in feature file(DT)
        subjects_num = feature['Subject'].copy()
        
        for i, subject in enumerate(subjects_num): # Leave only the subject name number(I00000 -> 00000)
            subjects_num[i] = int(subject[1:])
        
        feature['subject_num'] = subjects_num # Add a column which is subject name with only number
        feature = feature.sort_values(by=['PTID', 'subject_num'], kind='stable') # Sort by PTID first and subject name second in ascending order
        feature.drop(['subject_num'], axis=1, inplace=True) # Remove this column after sorting
    else: # Else(Amyloid, FDG, Tau)
        feature = feature.sort_values(by=['PTID'], kind='stable') # Sort by PTID in ascending order
    
    feature.reset_index(drop=True, inplace=True) # Set index in current order
    
    ### Find the number of inspections per person and display the number of inspections in the PTID corresponding to each subject
    inspection_num = feature.groupby('PTID').size().to_list() # Total number of inspections per person
    PTID_list = list(feature['PTID']) # List of 'PTID'

    inspection_index = []
    for inspection in inspection_num: # Find the number of inspections by index
        for i in range(inspection):
            inspection_index.append(i) # Append from 0 to the number of inspections per person
    
    inspection_index_str = list(map(str, inspection_index)) # Change the type of inspection index (int -> str)
    
    ID_list_tmp = list(zip(PTID_list, inspection_index_str)) # ('PTID', 'inspection_index')
    ID_list = list(map(lambda x:x[0] + '_' + x[1], ID_list_tmp)) # 'PTID + inspection_index'
    
    if filename == 'DT_thickness.xlsx':
        feature.insert(1, 'ID', ID_list)
        feature.insert(3, 'VISCODE', inspection_index)
    elif filename == 'Amyloid_SUVR.xlsx' or filename == 'FDG_SUVR.xlsx' or filename == 'Tau_SUVR.xlsx':
        feature.drop(['SCAN', 'EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTETHCAT', 'PTMARRY', 'APOE4'], 
                    axis=1, inplace=True) # Remove unnecessary columns
        feature.insert(0, 'ID', ID_list)
        feature.insert(2, 'VISCODE', inspection_index)
    
    return feature


### Get dictionary for subject with ROI features
def get_roi_feature(args, features_list, labels_dict):
    all_features = [] # List for storing preprocessed tables
    all_features_PTID = [] # List for storing PITD of preprocessed tables
    
    for feature in features_list:
        feature = preprocess_feature_table(args, feature) # Preprocess feature file
        
        PTID_list = list(feature['PTID']) # List of 'PTID'
        DX_list = list(feature['DX']) # List of 'DX'
        
        ### Remove subjects for people with different labels
        DX_dict = {}
        for ptid, dx in zip(PTID_list, DX_list): # Mark to '-1' when the same person has a different label
            if ptid in DX_dict and DX_dict[ptid] != dx:
                DX_dict[ptid] = -1
            else:
                DX_dict[ptid] = dx
        
        for i, ptid in enumerate(PTID_list): # Remove person with different labels
            if DX_dict[ptid] == -1:
                feature.drop(labels=i, inplace=True)
        
        feature.sort_index(inplace=True)
        
        all_features.append(feature)
        all_features_PTID.append(list(feature['PTID']))
    
    ### Find duplicate people from feature files 
    PTID_common_set = set()
    for i, ptid_list in enumerate(all_features_PTID): # Select only duplicate people from the feature files used as input
        if i == 0:
            PTID_common_set = set(ptid_list)
        else:
            PTID_common_set = PTID_common_set & set(ptid_list)
    
    PTID_common_list = list(PTID_common_set) # Change the common people set to list

    ### Select the maximum number of inspections in duplicate people
    all_features_num = len(all_features) # The number of feature files
    
    common_inspection = [[] for _ in range(all_features_num)] # List storing the number of checks for duplicate people present in each feature table
    common_inspection_max = [] # List storing the maximum number of checks of duplicate people present in each feature table
    
    for ptid in PTID_common_list: # For each duplicated person
        max_count = 0
        
        for i, feature in enumerate(all_features): # For each feature file
            PTID_inspection_num = list(feature['PTID']).count(ptid) # Number of checks for duplicate people present in each feature table
            common_inspection[i].append(PTID_inspection_num)
            max_count = max(max_count, PTID_inspection_num) # Maximum number of duplicate checks present in each feature table

        common_inspection_max.append(max_count)
        
    common_feature = [] # List with tables recreated from the original feature file with duplicate information from people
    common_inspection_dict = [] # List that stores a dictionary that stores the number of inspections of people present in the feature file
    common_inspection_max_dict = {} # Dictionary that stores the maximum number of inspection of people 
    
    for i, feature in enumerate(all_features): # For each feature file
        common_feature.append(feature[feature['PTID'].isin(PTID_common_list)])
        common_inspection_dict.append({ptid:inspection for ptid, inspection in zip(PTID_common_list, common_inspection[i])})

    common_inspection_max_dict = {ptid:inspection for ptid, inspection in zip(PTID_common_list, common_inspection_max)}
    
    ### Oversampling - the feature file that is insufficient adds the previous inspection result as it is depending on the maximum number of inspections
    for ptid in PTID_common_list:
        for i, inspection in enumerate(common_inspection_dict):
            if inspection[ptid] < common_inspection_max_dict[ptid]:
                last_inspection_index = inspection[ptid] - 1
                copy_object_id = ptid + '_' + str(last_inspection_index)
                
                for inspection_index in range(inspection[ptid], common_inspection_max_dict[ptid]):
                    new_id = ptid + '_' + str(inspection_index)
                    copy_object = common_feature[i].loc[common_feature[i].ID == copy_object_id].copy()
                    copy_object['ID'] = new_id
                    common_feature[i] = pd.concat([common_feature[i], copy_object], axis=0)
    
    for i in range(all_features_num):
        common_feature[i].set_index('ID', inplace=True) # Set the index through the newly assigned ID
        common_feature[i].sort_index(inplace=True) # Sort by ID index

    ### Leave only the necessary columns in the feature file lastly
    subject = None
    for feature in common_feature:
        if 'Subject' in feature.columns:
            subject = feature['Subject']
            feature.drop(['Subject'], axis=1, inplace=True) # Remove 'Subject' column from DT

    common_feature[0].insert(0, "Subject", subject) # Add 'Subject' column to feature file used
    
    labels = common_feature[0]['DX'].map(labels_dict) # Number labels according to the desired task
    common_feature[0].drop(['DX'], axis=1, inplace=True) # Remove column 'DX' from original feature file

    ### Leave only feature files to be used in practice
    if args.use_all_features == 0:
        while len(common_feature) > 1:
            common_feature.pop()
    
    ### Create a dictionary with ROI feature and label as value and subject as key
    subject_map = {}
    for i, subject in enumerate(common_feature[0]['Subject']):
        if labels[i] is not np.nan:
            subject_map[subject] = [torch.stack([torch.tensor(x.iloc[i, 3:], dtype=torch.float) for x in common_feature]), labels[i]]

    return subject_map


### Load the graph data (edge information)
def load_graph_data(args, features_list, subject_map, model, p):
    _, files = walk(args.data_path)
    
    if args.use_all_features == 0:
        while len(features_list) > 1:
            features_list.pop()
    
    adjacencies, features, labels, eigenvalues, eigenvectors, laplacians = list([] for _ in range(6))
    threshold = 10
    
    f_count = 0
    l_count = 0
    i_count = 0
    
    for i, file in enumerate(files[2]):
        index = file.rfind('_') + 1 # Find index corresponding to '_' from file name
        subject_name = file[index:] # Get subject name from file name
        
        if subject_name in subject_map: # If subject name we have is in subject_map
            path = args.data_path + args.adjacency_path # Adjacency matrices path
            
            A = torch.FloatTensor(np.genfromtxt(f"{path}/{file}")) # Get adjacency matrix (# of ROI features, # of ROI features)
            A = 0.5 * (A.transpose(0, 1) + A) # Make symmetric adjacency matrix
            A[A < threshold] = 0 # 0 if each value is less than the threshold
            A[A != 0] = 1 # 1 if each value is not 0
            
            subject_info = subject_map[subject_name]
            
            if not np.isnan(subject_info[1]): # If subject has label(no nan)
                D = torch.diag(torch.sum(A, axis=1)) # Get degree matrix
                
                if torch.sum(D != 0).item() == A.shape[0]: # If all the diagonal entries exist in the degree matrix
                    L = D - A # Get laplacian matrix (L = D - A)
                    
                    D_inverse_sqrt = torch.linalg.inv(torch.sqrt(D)) # Get inverse square root degree matrix
                    L_normalized = torch.matmul(torch.matmul(D_inverse_sqrt, L), D_inverse_sqrt) # Get normalized laplacian matrix
                    
                    eigenvalue, eigenvector = eigh(L_normalized) # Get eigenvalues and eigenvectors
                    
                    if args.use_feature == 1: # If feature is used
                        X = torch.FloatTensor(subject_info[0]).reshape(len(features_list), -1).T # Use ROI feature
                    else: 
                        X = torch.sum(A, axis=1).reshape(-1, 1) # Use degree as feature 

                    X = normalize_feature(X) # Normalize the feature matrix

                    A += torch.eye(A.shape[0]) # A + I
                    A = normalize_adjacency(A) # Normalize the adjacency matrix

                    if args.model == 'svm': # For 'svm' library
                        A = A.reshape(1, -1)
                        X = torch.cat((A, X.reshape(1, -1)), dim=1).reshape(-1)
                        X = X.reshape(-1)
                    
                    if args.model == 'gdc': # For 'gdc'
                        A = gdc(A)
                    
                    y = torch.LongTensor([subject_info[1]]) # Class label
                    
                    adjacencies.append(A)
                    features.append(X)
                    labels.append(y)
                    
                    if args.model == 'graphheat' or model == 0 or model == 1:
                        eigenvalues.append(eigenvalue)
                        eigenvectors.append(eigenvector)
                    if model == 1:
                        laplacians.append(L_normalized)
                    
                else:
                    print(f"{file:<20} is not invertible")
                    i_count += 1
            else:
                print(f"{file:<20} label is not found")
                l_count += 1
        else:
            print(f"{file:<20} features are not found")
            f_count += 1
    
    print(f"Features are not found: {f_count} / Label is not found: {l_count} / Degree is not invertible: {i_count}")

    ### Change lists of data to tensor
    adjacencies = torch.stack(adjacencies) # Shape : (# subjects, # ROI features, # ROI features)
    features = torch.stack(features) # Shape : (# subjects, # ROI features, # used features)
    labels = torch.stack(labels).reshape(-1) # Shape : (# subjects)

    if args.model == 'graphheat' or model == 0 or model == 1:
        eigenvalues = torch.stack(eigenvalues)
        eigenvectors = torch.stack(eigenvectors)
    if model == 1:
        laplacians = torch.stack(laplacians)
    
    return features_list, adjacencies, features, labels, eigenvalues, eigenvectors, laplacians


### Trainer for 'Ours'
class Our_Trainer:
    def __init__(self, args, device, network, train_loader, valid_loader, test_loader, adj_size):
        self.args = args
        self.device = device
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.adj_size = adj_size
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)

        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        if torch.cuda.is_available():
            self.network = self.network.to(device)
            self.t = self.t.to(device)
        
        self.timings = []
        self.timings_hk = []
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.starter_hk, self.ender_hk = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        

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
        y_oh = torch.zeros_like(output) # (# sample, # label)
        y_oh.scatter_(1, label.reshape(-1, 1), 1)
        dl_ds = (torch.exp(output) - y_oh) / output.shape[0]
        
        ds_dro0 = torch.mul(dl_ds, self.network.linrdp2) @ self.network.linear2.weight
        ds_dro1 = torch.mul(ds_dro0, self.network.linrdp)
        
        #ds_dro1 = torch.mul(dl_ds @ self.network.linear2.weight,  self.network.linrdp)
        dl_dro = torch.matmul(ds_dro1, self.network.linear.weight).reshape(-1, heat_kernel.shape[-2], self.args.hidden_units)
        
        if self.args.layer_num == 1:
            dl_dc = torch.mul(dl_dro, self.network.gcn.rdp) @ self.network.gcn.gc1.weight.T
            dl_dt = torch.mul((dl_dc @ self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
        elif self.args.layer_num == 2:
            dl_dl2 = torch.mul(dl_dro, self.network.gcn.rdp2) @ self.network.gcn.gc2.weight.T

            dl_first = torch.mul((dl_dl2 @ self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
            backward = torch.matmul(self.network.gcn.gc1.weight.T, feature.swapaxes(1, 2))

            dl_second_tmp = torch.mul(dl_dl2, self.network.gcn.rdp)
            dl_second = torch.matmul(torch.mul(dl_second_tmp @ backward, heat_kernel_grad), heat_kernel.swapaxes(1, 2))

            dl_dt = dl_first + dl_second
        elif self.args.layer_num == 3:
            dl_dc = torch.mul(dl_dro, self.network.gcn.rdp3) @ self.network.gcn.gc3.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.network.gcn.f2.swapaxes(1, 2)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.network.gcn.rdp2) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
                            heat_kernel.swapaxes(1, 2))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.network.gcn.rdp2) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.rdp) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
                            heat_kernel.swapaxes(1, 2))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third

        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=(0, 2))
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1, 2))]).to(self.device)
            
        dl_dt += self.t_deriv() # Add regularizer on t
        now_lr = self.t_lr * dl_dt

        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold

        self.t = self.t - now_lr # Update t

        if self.args.use_t_local == 1:
            print(f't:{self.t[0].item()}', end=' ')
        else:
            print(f't:{self.t.item():.4f}', end=' ')

    ### Train
    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        
        scale_list_max = []
        scale_list_min = []

        for iter in range(1, self.args.epochs + 1):
            self.network.train()
            
            for adjacency, feature, label, eigenvalue, eigenvector in self.train_loader:
                self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

                heat_kernel, heat_kernel_grad = compute_heat_kernel(eigenvalue, eigenvector, self.t) # Use heat kernel instead of adjacency matrix

                # Use heat kernel instead of adjacency matrix
                output = self.network.forward(feature, heat_kernel) # Shape: (# of samples, # of labels)

                loss_train = F.nll_loss(output, label) + self.t_loss()
                accuracy_train = compute_accuracy(output, label)
                loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
                
                if iter % 100 == 0:
                    print(f"Epoch [{iter} / {self.args.epochs}] loss_train: {loss_train.item():.5f} accuracy_train: {accuracy_train.item():.5f}")

                with torch.no_grad():
                    self.fir_deriv(output, feature, label, heat_kernel, heat_kernel_grad)
                
                scale_list_max.append(self.t[18].cpu().detach().numpy().tolist())
                scale_list_min.append(self.t[145].cpu().detach().numpy().tolist())
                
                
                self.optimizer.step() # Updates the parameters
                
                #wandb.log({"loss_train": loss_train.item(),
                #           "accuracy_train": accuracy_train.item()})

                lt.append(loss_train.item())
                at.append(accuracy_train.item())
            
            self.network.eval()
        
        return scale_list_max, scale_list_min

    ### Test
    def test(self):
        tl, ta, tac, tpr, tsp, tse, f1s, ts = [[] for _ in range(8)]

        self.network.eval()

        for adjacency, feature, label, eigenvalue, eigenvector in self.test_loader:
            heat_kernel, heat_kernel_grad = compute_heat_kernel(eigenvalue, eigenvector, self.t)

            output = self.network.forward(feature, heat_kernel) # Shape: (# of samples, # of labels)
            
            loss_test = F.nll_loss(output, label) + self.t_loss()
            accuracy_test = compute_accuracy(output, label)

            print("Prediction Labels >")
            print(output.max(1)[1])
            print("Real Labels >")
            print(label)
            
            print(f"Test set results: loss_test: {loss_test.item():.5f} accuracy_test: {accuracy_test.item():.5f}")

            ac, pr, sp, se, f1 = confusion(output, label)
            print(f"Confusion - Accuracy: {ac:.10f} Precision: {pr:.10f} Specificity: {sp:.10f} Sensitivity: {se:.10f} F1 score: {f1:.10f}")

            tl.append(loss_test.item())
            ta.append(accuracy_test.item())
            tac.append(ac.item())
            tpr.append(pr.item())
            tsp.append(sp.item())
            tse.append(se.item())
            f1s.append(f1.item())
            
            for i in range(len(self.t)):
                ts.append(self.t[i].item())

        return np.mean(tl), np.mean(ta), np.mean(tac), np.mean(tpr), np.mean(tsp), np.mean(tse), np.mean(f1s), np.array(ts)
    

### Trainer for approximation version of 'Ours'
class Our_APPROX_Trainer:
    def __init__(self, args, poly, device, network, train_loader, valid_loader, test_loader, adj_size):
        self.args = args
        self.device = device
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.adj_size = adj_size # 160
        self.optimizer = optim.Adam(self.network.parameters(), 
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)

        if args.use_t_local == 1: # Local scale 
            self.t = torch.empty(adj_size).fill_(2.) # (160)
        else: # Global scale 
            self.t = torch.tensor([2.]) # (1)
        
        self.t_lr = self.args.t_lr # 1
        self.t_loss_threshold = self.args.t_loss_threshold # 0.01
        self.t_lambda = self.args.t_lambda # 1
        self.t_threshold = self.args.t_threshold # 0.1
        
        self.m_che = self.args.m_chebyshev # 30
        self.m_her = self.args.m_hermite # 60
        self.m_lag = self.args.m_laguerre # 30
        
        self.poly = poly
        
        if torch.cuda.is_available():
            self.network = self.network.to(device)
            self.t = self.t.to(device)
            
        self.timings = []
        self.timings_hk = []
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.starter_hk, self.ender_hk = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        

    ### Scale regularization of loss function
    def t_loss(self):
        t_one = torch.abs(self.t)
        t_zero = torch.zeros_like(self.t)

        t_loss_tmp = torch.where(self.t < self.t_loss_threshold, t_one, t_zero)
        t_loss_final = self.t_lambda * torch.sum(t_loss_tmp) # λ|s|

        return t_loss_final

    def t_deriv(self):
        t_one = self.t_lambda * torch.ones_like(self.t)
        t_zero = torch.zeros_like(self.t)

        t_de = torch.where(self.t < self.t_loss_threshold, -t_one, t_zero)

        return t_de

    def fir_deriv(self, output, feature, label, heat_kernel, heat_kernel_grad):
        y_oh = torch.zeros_like(output) # (# sample, # label)
        y_oh.scatter_(1, label.reshape(-1, 1), 1) # (# sample, # label)

        dl_ds = (torch.exp(output) - y_oh) / output.shape[0] # (# sample, # label)

        ds_dro0 = torch.mul(dl_ds, self.network.linrdp2) @ self.network.linear2.weight
        ds_dro1 = torch.mul(ds_dro0, self.network.linrdp)
        
        #ds_dro1 = torch.mul(dl_ds @ self.network.linear2.weight,  self.network.linrdp)
        dl_dro = torch.matmul(ds_dro1, self.network.linear.weight).reshape(-1, heat_kernel.shape[-2], self.args.hidden_units)
        
        if self.args.layer_num == 1:
            dl_dc = torch.mul(dl_dro, self.network.gcn.rdp) @ self.network.gcn.gc1.weight.T
            dl_dt = torch.mul((dl_dc @ self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
        elif self.args.layer_num == 2:
            dl_dl2 = torch.mul(dl_dro, self.network.gcn.rdp2) @ self.network.gcn.gc2.weight.T

            dl_first = torch.mul((dl_dl2 @ self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
            backward = torch.matmul(self.network.gcn.gc1.weight.T, feature.swapaxes(1, 2))

            dl_second_tmp = torch.mul(dl_dl2, self.network.gcn.rdp)
            dl_second = torch.matmul(torch.mul(dl_second_tmp @ backward, heat_kernel_grad), heat_kernel.swapaxes(1, 2))

            dl_dt = dl_first + dl_second
        elif self.args.layer_num == 3:
            dl_dc = torch.mul(dl_dro, self.network.gcn.rdp3) @ self.network.gcn.gc3.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.network.gcn.f2.swapaxes(1, 2)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.network.gcn.rdp2) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
                            heat_kernel.swapaxes(1, 2))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.network.gcn.rdp2) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.rdp) 
                            @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
                            heat_kernel.swapaxes(1, 2))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third

        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=(0, 2)) # 160
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1, 2))]).to(self.device) # 1

        dl_dt += self.t_deriv() # 160 / 1
        now_lr = self.t_lr * dl_dt # 160 / 1
        
        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold
        
        self.t = self.t - now_lr # Update t 

        if self.args.use_t_local == 1:
            print(f't:{self.t[0].item()}', end=' ')
        else:
            print(f't:{self.t.item():.4f}', end=' ')

    ### Train
    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        scale_list_max = []
        scale_list_min = []

        for iter in range(1, self.args.epochs + 1):
            self.network.train()
            
            for adjacency, feature, label, eigenvalue, eigenvector, laplacian in self.train_loader:
                self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

                self.starter.record()
                self.starter_hk.record()
                
                if self.poly == 0: # Chebyshev
                    b = eigenvalue[:,eigenvalue.shape[1]-1] # Maximum value of eigenvalue
                    heat_kernel, heat_kernel_grad = compute_heat_kernel_chebyshev(laplacian, self.m_che, self.t, b, self.device)
                elif self.poly == 1: # Hermite
                    heat_kernel, heat_kernel_grad = compute_heat_kernel_hermite(laplacian, self.m_her, self.t, self.device)
                elif self.poly == 2: # Laguerre
                    heat_kernel, heat_kernel_grad = compute_heat_kernel_laguerre(laplacian, self.m_lag, self.t, self.device)

                self.ender_hk.record()
                torch.cuda.synchronize()
                curr_time_hk = self.starter_hk.elapsed_time(self.ender_hk)
                self.timings_hk.append(curr_time_hk)
                
                # Use heat kernel instead of adjacency matrix
                output = self.network.forward(feature, heat_kernel) # Shape: (# of samples, # of labels)

                loss_train = F.nll_loss(output, label) + self.t_loss()
                accuracy_train = compute_accuracy(output, label)
                loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
                
                if iter % 100 == 0:
                    print(f"Epoch [{iter} / {self.args.epochs}] loss_train: {loss_train.item():.5f} accuracy_train: {accuracy_train.item():.5f}")

                with torch.no_grad():
                    self.fir_deriv(output, feature, label, heat_kernel, heat_kernel_grad)
                
                scale_list_max.append(self.t[18].cpu().detach().numpy().tolist())
                scale_list_min.append(self.t[145].cpu().detach().numpy().tolist())
                
                self.optimizer.step() # Updates the parameters
                
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender)
                self.timings.append(curr_time)
                
                #wandb.log({"loss_train": loss_train.item(),
                #           "accuracy_train": accuracy_train.item()})

                lt.append(loss_train.item())
                at.append(accuracy_train.item())
            
            self.network.eval()
        
        return self.timings_hk[0], self.timings[0]

    ### Test
    def test(self):
        tl, ta, tac, tpr, tsp, tse, f1s, ts = [[] for _ in range(8)]

        self.network.eval()

        for adjacency, feature, label, eigenvalue, eigenvector, laplacian in self.test_loader:
            
            if self.args.polynomial == 0: # Chebyshev
                eigmax = eigenvalue[:,eigenvalue.shape[1]-1]
                heat_kernel, _ = compute_heat_kernel_chebyshev(laplacian, self.m_che, self.t, eigmax, self.device)
            elif self.args.polynomial == 1: # Hermite
                heat_kernel, _ = compute_heat_kernel_hermite(laplacian, self.m_her, self.t, self.device)
            elif self.args.polynomial == 2: # Laguerre
                heat_kernel, _ = compute_heat_kernel_laguerre(laplacian, self.m_lag, self.t, self.device)

            output = self.network.forward(feature, heat_kernel) # Shape: (# of samples, # of labels)
            
            loss_test = F.nll_loss(output, label) + self.t_loss()
            accuracy_test = compute_accuracy(output, label)

            print("Prediction Labels >")
            print(output.max(1)[1])
            print("Real Labels >")
            print(label)
            
            print(f"Test set results: loss_test: {loss_test.item():.5f} accuracy_test: {accuracy_test.item():.5f}")

            ac, pr, sp, se, f1 = confusion(output, label)
            print(f"Confusion - Accuracy: {ac:.10f} Precision: {pr:.10f} Specificity: {sp:.10f} Sensitivity: {se:.10f} F1 score: {f1:.10f}")

            tl.append(loss_test.item())
            ta.append(accuracy_test.item())
            tac.append(ac.item())
            tpr.append(pr.item())
            tsp.append(sp.item())
            tse.append(se.item())
            f1s.append(f1.item())
            
            for i in range(len(self.t)):
                ts.append(self.t[i].item())

        return np.mean(tl), np.mean(ta), np.mean(tac), np.mean(tpr), np.mean(tsp), np.mean(tse), np.mean(f1s), np.array(ts)


def main1(args, device):
    timings = []    
    timings_hk = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter_hk, ender_hk = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    ### Select the feature files and labels
    features_list, labels_dict = set_classification_conditions(args)
    
    ### Get dictionary for subject with ROI features
    subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

    starter.record()
    starter_hk.record()
    used_features, adjacencies0, features0, labels0, eigenvalues0, eigenvectors0, _ = load_graph_data(args, features_list, subject_map, 0, 'e')
    
    print(features0.shape)
    
    ### K-fold cross validation 
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)
    
    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(adjacencies0, labels0):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        adjacencies0 = adjacencies0.to(device) # Shape: (# subjects, # ROI feature, # ROI features)
        features0 = features0.to(device) # Shape: (# subjects, # ROI features, # used features)
        labels0 = labels0.to(device) # Shape: (# subjects)
    
        eigenvalues0 = eigenvalues0.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors0 = eigenvectors0.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)

    if args.model != 'svm':
        num_ROI_features = features0.shape[1]
        num_used_features = features0.shape[2]
        
    for i, idx_pair in enumerate(idx_pairs):
        if i == 1 :
            break
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        ### Build data loader
        idx_train, idx_test = idx_pair

        data_train_e = TensorDataset(adjacencies0[idx_train], features0[idx_train], labels0[idx_train], eigenvalues0[idx_train], eigenvectors0[idx_train])
        data_test_e = TensorDataset(adjacencies0[idx_test], features0[idx_test], labels0[idx_test], eigenvalues0[idx_test], eigenvectors0[idx_test])

        data_loader_train_e = DataLoader(data_train_e, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
        data_loader_test_e = DataLoader(data_test_e, batch_size=idx_test.shape[0], shuffle=True) # Full-batch

        network = GAPHeat2(adj_dim=num_ROI_features,
                            in_dim=num_used_features,
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(labels0).item() + 1,
                            dropout=args.dropout_rate)


        optimizer = optim.Adam(network.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        if args.use_t_local == 1: # Local scale
            t = torch.empty(adjacencies0[0].shape[0]).fill_(2.)
        else: # Global scale
            t = torch.tensor([2.]) 
        
        t_lr = args.t_lr
        t_loss_threshold = args.t_loss_threshold
        t_lambda = args.t_lambda
        t_threshold = args.t_threshold
        
        if torch.cuda.is_available():
            network = network.to(device)
            t = t.to(device)
        

        ### Scale regularization of loss function
        def t_loss(t):
            t_one = torch.abs(t)
            t_zero = torch.zeros_like(t)

            t_l = torch.where(t < t_loss_threshold, t_one, t_zero)

            return t_lambda * torch.sum(t_l)

        def t_deriv(t):
            t_one = t_lambda * torch.ones_like(t)
            t_zero = torch.zeros_like(t)

            t_de = torch.where(t < t_loss_threshold, -t_one, t_zero)

            return t_de

        def fir_deriv(output, feature, label, heat_kernel, heat_kernel_grad, t):
            y_oh = torch.zeros_like(output) # (# sample, # label)
            y_oh.scatter_(1, label.reshape(-1, 1), 1)
            dl_ds = (torch.exp(output) - y_oh) / output.shape[0]
            
            ds_dro0 = torch.mul(dl_ds, network.linrdp2) @ network.linear2.weight
            ds_dro1 = torch.mul(ds_dro0, network.linrdp)
            
            #ds_dro1 = torch.mul(dl_ds @ self.network.linear2.weight,  self.network.linrdp)
            dl_dro = torch.matmul(ds_dro1, network.linear.weight).reshape(-1, heat_kernel.shape[-2], args.hidden_units)
            
            if args.layer_num == 1:
                dl_dc = torch.mul(dl_dro, network.gcn.rdp) @ network.gcn.gc1.weight.T
                dl_dt = torch.mul((dl_dc @ network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
            elif args.layer_num == 2:
                dl_dl2 = torch.mul(dl_dro, network.gcn.rdp2) @ network.gcn.gc2.weight.T

                dl_first = torch.mul((dl_dl2 @ network.gcn.f.swapaxes(1, 2)), heat_kernel_grad)
                backward = torch.matmul(network.gcn.gc1.weight.T, feature.swapaxes(1, 2))

                dl_second_tmp = torch.mul(dl_dl2, network.gcn.rdp)
                dl_second = torch.matmul(torch.mul(dl_second_tmp @ backward, heat_kernel_grad), heat_kernel.swapaxes(1, 2))

                dl_dt = dl_first + dl_second
            # elif self.args.layer_num == 3:
            #     dl_dc = torch.mul(dl_dro, self.network.gcn.rdp3) @ self.network.gcn.gc3.weight.T
            #     dl_dc_first = torch.mul((dl_dc @ self.network.gcn.f2.swapaxes(1, 2)), heat_kernel_grad)
            #     dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.network.gcn.rdp2) 
            #                     @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
            #                     heat_kernel.swapaxes(1, 2))
            #     dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.network.gcn.rdp2) 
            #                     @ self.network.gcn.gc2.weight.T, self.network.gcn.rdp) 
            #                     @ self.network.gcn.gc2.weight.T, self.network.gcn.f.swapaxes(1, 2)), heat_kernel_grad), 
            #                     heat_kernel.swapaxes(1, 2))
            #     dl_dt = dl_dc_first + dl_dc_second + dl_dc_third

            if args.use_t_local == 1:
                dl_dt = torch.sum(dl_dt, dim=(0, 2))
            else:
                dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1, 2))]).to(device)
                
            dl_dt += t_deriv(t) # Add regularizer on t
            now_lr = t_lr * dl_dt

            now_lr[now_lr > t_threshold] =t_threshold
            now_lr[now_lr < -t_threshold] = -t_threshold

            t = t - now_lr # Update t

            if args.use_t_local == 1:
                print(f't:{t[0].item()}', end=' ')
            else:
                print(f't:{t.item():.4f}', end=' ')

        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        
        scale_list_max = []
        scale_list_min = []

        for iter in range(1, args.epochs + 1):
            network.train()
            
            for adjacency, feature, label, eigenvalue, eigenvector in data_loader_train_e:
                optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

                heat_kernel, heat_kernel_grad = compute_heat_kernel(eigenvalue, eigenvector, t) # Use heat kernel instead of adjacency matrix
                
                ender_hk.record()
                torch.cuda.synchronize()
                curr_time_hk = starter_hk.elapsed_time(ender_hk)
                timings_hk.append(curr_time_hk)
                    
                # Use heat kernel instead of adjacency matrix
                output = network.forward(feature, heat_kernel) # Shape: (# of samples, # of labels)

                loss_train = F.nll_loss(output, label) + t_loss(t)
                accuracy_train = compute_accuracy(output, label)
                loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
                
                if iter % 100 == 0:
                    print(f"Epoch [{iter} / {args.epochs}] loss_train: {loss_train.item():.5f} accuracy_train: {accuracy_train.item():.5f}")

                with torch.no_grad():
                    fir_deriv(output, feature, label, heat_kernel, heat_kernel_grad, t)
                
                # scale_list_max.append(t[18].cpu().detach().numpy().tolist())
                # scale_list_min.append(t[145].cpu().detach().numpy().tolist())
                
                
                optimizer.step() # Updates the parameters
                
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
                
                #wandb.log({"loss_train": loss_train.item(),
                #           "accuracy_train": accuracy_train.item()})

                lt.append(loss_train.item())
                at.append(accuracy_train.item())
            
            network.eval()
            
        return timings_hk[0], timings[0]








def main2(args, device):
    ### Select the feature files and labels
    features_list, labels_dict = set_classification_conditions(args)
    
    ### Get dictionary for subject with ROI features
    subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

    used_features, adjacencies_c, features_c, labels_c, eigenvalues_c, eigenvectors_c, laplacians_c = load_graph_data(args, features_list, subject_map, 1, 'c')
    
    print(features_c.shape)

    ### K-fold cross validation 
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)
    
    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(adjacencies_c, labels_c):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        adjacencies_c = adjacencies_c.to(device) # Shape: (# subjects, # ROI feature, # ROI features)
        features_c = features_c.to(device) # Shape: (# subjects, # ROI features, # used features)
        labels_c = labels_c.to(device) # Shape: (# subjects)
        eigenvalues_c = eigenvalues_c.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors_c = eigenvectors_c.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        laplacians_c = laplacians_c.to(device)

    
    if args.model != 'svm':
        num_ROI_features = features_c.shape[1]
        num_used_features = features_c.shape[2]
        
    for i, idx_pair in enumerate(idx_pairs):
        if i == 1 :
            break
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        ### Build data loader
        idx_train, idx_test = idx_pair

        data_train_c = TensorDataset(adjacencies_c[idx_train], features_c[idx_train], labels_c[idx_train], eigenvalues_c[idx_train], eigenvectors_c[idx_train], laplacians_c[idx_train])
        data_test_c = TensorDataset(adjacencies_c[idx_test], features_c[idx_test], labels_c[idx_test], eigenvalues_c[idx_test], eigenvectors_c[idx_test], laplacians_c[idx_test])

        data_loader_train_c = DataLoader(data_train_c, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
        data_loader_test_c = DataLoader(data_test_c, batch_size=idx_test.shape[0], shuffle=True) # Full-batch

        model_c = GAPHeat2(adj_dim=num_ROI_features,
                            in_dim=num_used_features,
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(labels_c).item() + 1,
                            dropout=args.dropout_rate)

        trainer_chl0 = Our_APPROX_Trainer(args, 0, device, model_c, data_loader_train_c, data_loader_test_c, data_loader_test_c, adjacencies_c[0].shape[0])
        hk, tt = trainer_chl0.train()

    return hk, tt
        




def main3(args, device):
    ### Select the feature files and labels
    features_list, labels_dict = set_classification_conditions(args)
    
    ### Get dictionary for subject with ROI features
    subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

    used_features, adjacencies_h, features_h, labels_h, eigenvalues_h, eigenvectors_h, laplacians_h = load_graph_data(args, features_list, subject_map, 1, 'h')
    
    print(features_h.shape)

    ### K-fold cross validation 
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)
    
    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(adjacencies_h, labels_h):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        adjacencies_h = adjacencies_h.to(device) # Shape: (# subjects, # ROI feature, # ROI features)
        features_h = features_h.to(device) # Shape: (# subjects, # ROI features, # used features)
        labels_h = labels_h.to(device) # Shape: (# subjects)
        eigenvalues_h = eigenvalues_h.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors_h = eigenvectors_h.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        laplacians_h = laplacians_h.to(device)

    if args.model != 'svm':
        num_ROI_features = features_h.shape[1]
        num_used_features = features_h.shape[2]
        
    for i, idx_pair in enumerate(idx_pairs):
        if i == 1 :
            break
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        ### Build data loader
        idx_train, idx_test = idx_pair

        data_train_h = TensorDataset(adjacencies_h[idx_train], features_h[idx_train], labels_h[idx_train], eigenvalues_h[idx_train], eigenvectors_h[idx_train], laplacians_h[idx_train])
        data_test_h = TensorDataset(adjacencies_h[idx_test], features_h[idx_test], labels_h[idx_test], eigenvalues_h[idx_test], eigenvectors_h[idx_test], laplacians_h[idx_test])
        
        data_loader_train_h = DataLoader(data_train_h, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
        data_loader_test_h = DataLoader(data_test_h, batch_size=idx_test.shape[0], shuffle=True) # Full-batch

        model_h = GAPHeat2(adj_dim=num_ROI_features,
                            in_dim=num_used_features,
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(labels_h).item() + 1,
                            dropout=args.dropout_rate)

        trainer_chl1 = Our_APPROX_Trainer(args, 1, device, model_h, data_loader_train_h, data_loader_test_h, data_loader_test_h, adjacencies_h[0].shape[0])
        hk, tt = trainer_chl1.train()

    return hk, tt






def main4(args, device):
    ### Select the feature files and labels
    features_list, labels_dict = set_classification_conditions(args)
    
    ### Get dictionary for subject with ROI features
    subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

    used_features, adjacencies_l, features_l, labels_l, eigenvalues_l, eigenvectors_l, laplacians_l = load_graph_data(args, features_list, subject_map, 1, 'l')

    print(features_l.shape)
    
    ### K-fold cross validation 
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)
    
    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(adjacencies_l, labels_l):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        adjacencies_l = adjacencies_l.to(device) # Shape: (# subjects, # ROI feature, # ROI features)
        features_l = features_l.to(device) # Shape: (# subjects, # ROI features, # used features)
        labels_l = labels_l.to(device) # Shape: (# subjects)
        eigenvalues_l = eigenvalues_l.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors_l = eigenvectors_l.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        laplacians_l = laplacians_l.to(device)
    
    if args.model != 'svm':
        num_ROI_features = features_l.shape[1]
        num_used_features = features_l.shape[2]
    
    for i, idx_pair in enumerate(idx_pairs):
        if i == 1 :
            break
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        ### Build data loader
        idx_train, idx_test = idx_pair

        data_train_l = TensorDataset(adjacencies_l[idx_train], features_l[idx_train], labels_l[idx_train], eigenvalues_l[idx_train], eigenvectors_l[idx_train], laplacians_l[idx_train])
        data_test_l = TensorDataset(adjacencies_l[idx_test], features_l[idx_test], labels_l[idx_test], eigenvalues_l[idx_test], eigenvectors_l[idx_test], laplacians_l[idx_test])
        
        data_loader_train_l = DataLoader(data_train_l, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
        data_loader_test_l = DataLoader(data_test_l, batch_size=idx_test.shape[0], shuffle=True) # Full-batch

        model_l = GAPHeat2(adj_dim=num_ROI_features,
                            in_dim=num_used_features,
                            hid_dim=args.hidden_units,
                            out_dim=torch.max(labels_l).item() + 1,
                            dropout=args.dropout_rate)

        trainer_chl2 = Our_APPROX_Trainer(args, 2, device, model_l, data_loader_train_l, data_loader_test_l, data_loader_test_l, adjacencies_l[0].shape[0])
        hk, tt = trainer_chl2.train()
    
    return hk, tt



### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', default='adni', help='Type of dataset')
    parser.add_argument('--data_path', default='./data/adni_2022')
    parser.add_argument('--adjacency_path', default='/matrices2326')
    ### Condition
    parser.add_argument('--seed_num', type=int, default=100, help='Number of random seed')
    parser.add_argument('--device_num', type=int, default=4, help='Which gpu to use')
    parser.add_argument('--features', type=int, default=10, help='Features to use')
    parser.add_argument('--labels', type=int, default=5, help="Labels to use")
    parser.add_argument('--model', type=str, default='ours_approx', help='Models to use')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    parser.add_argument('--use_feature', type=int, default=1, help='Whether features are used (0:degree / 1:feature)')
    parser.add_argument('--use_all_features', type=int, default=0, help='Whether all features are used')
    ### Experiment
    parser.add_argument('--split_num', type=int, default=5, help="Number of splits for k-fold")
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
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
    #wandb.config.update(args)
    set_randomness(args.seed_num)
    device = torch.device('cuda:' + str(args.device_num) if torch.cuda.is_available() else 'cpu')
    
    hk1, t1 = main1(args, device)
    print(hk1, t1)
    hk2, t2 = main2(args, device)
    print(hk2, t2)
    hk3, t3 = main3(args, device)
    print(hk3, t3)
    hk4, t4 = main4(args, device)
    print(hk4, t4)
    
    all_hk=[]
    all_time = []
    all_time.append(t4)
    all_time.append(t3)
    all_time.append(t2)
    all_time.append(t1)
    all_hk.append(hk4)
    all_hk.append(hk3)
    all_hk.append(hk2)
    all_hk.append(hk1)
    
    print(all_time)
    print(all_hk)
    
    
    
    name = ['L', 'H', 'C', 'E']
    
    colors = ['#d62728','#2ca02c','#ff7f0e','#1f77b4']
    colors_black = ['#000000','#000000','#000000','#000000']
    y = np.arange(4)
    #plt.rc('font',family='Times New Roman')
    plt.barh(y, all_time, color=colors, height=0.5, zorder=2)
    plt.barh(y, all_hk, color= colors_black, height=0.25, zorder=3)
    #plt.barh(y, all_time, width=2, color=colors, edgecolor='lightgray', zorder = 2)
    #plt.title('Cora', fontweight='bold', fontsize=40, pad=10, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(y, name, fontsize=12, fontweight='bold')
    plt.xlabel('time (ms)', fontsize=20, fontweight='bold')
    plt.grid(True, axis='x', zorder=1)
    plt.tight_layout()
    
    plt.show()
    #plt.savefig(args.data + '_time.png')


    #     avl.append(losses)
    #     ava.append(accuracies)
    #     avac.append(cf_accuracies)
    #     avpr.append(cf_precisions)
    #     avsp.append(cf_specificities)
    #     avse.append(cf_sensitivities)
    #     avf1s.append(cf_f1score)
    #     if args.model == 'ours' or args.model == 'ours_approx':
    #         ts.append(t)
        
    # class_info = labels.tolist()
    # cnt = Counter(class_info)

    # ### Show results
    # print("--------------- Result ---------------")
    # if args.data == 'adni':
    #     print(f"Used features:        {used_features}")
    # print(f"Label distribution:   {cnt}")
    # print(f"{args.split_num}-Fold test loss:     {avl}")
    # print(f"{args.split_num}-Fold test accuracy: {ava}")
    # print("---------- Confusion Matrix ----------")
    # #print(f"{args.split_num}-Fold accuracy:      {avac}")
    # print(f"{args.split_num}-Fold precision:     {avpr}")
    # #print(f"{args.split_num}-Fold specificity:   {avsp}")
    # print(f"{args.split_num}=Fold sensitivity:   {avse}")
    # #print(f"{args.split_num}=Fold f1 score:      {avf1s}")
    # print("-------------- Mean, Std --------------")
    # #print(f"Mean: {np.mean(avl):.10f} {np.mean(ava):.10f} {np.mean(avac):.10f} {np.mean(avpr):.10f} {np.mean(avsp):.10f} {np.mean(avse):.10f} {np.mean(avf1s):.10f}")
    # #print(f"Std:  {np.std(avl):.10f} {np.std(ava):.10f} {np.std(avac):.10f} {np.std(avpr):.10f} {np.std(avsp):.10f} {np.std(avse):.10f} {np.mean(avf1s):.10f}")
    # print(f"Mean:  {np.mean(ava):.10f} {np.mean(avpr):.10f} {np.mean(avse):.10f}")
    # print(f"Std:   {np.std(ava):.10f}  {np.std(avpr):.10f}  {np.std(avse):.10f}")
    # if args.model == 'ours' or args.model == 'ours_approx':
    #     print("t: ", *(np.mean(ts, axis=0)))

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")