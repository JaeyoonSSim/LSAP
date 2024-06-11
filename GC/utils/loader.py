import torch
import numpy as np
import pandas as pd
import sys
import os

from torch.linalg import eigh
from os import walk
from models.gdc import gdc
from .utility import *
from torch.utils.data import TensorDataset, DataLoader

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
def load_graph_data(args, features_list, subject_map):
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
                    
                    eigenvalues.append(eigenvalue)
                    eigenvectors.append(eigenvector)
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
    eigenvalues = torch.stack(eigenvalues)
    eigenvectors = torch.stack(eigenvectors)
    laplacians = torch.stack(laplacians)
    print(features.shape)
    print(labels.shape)
    print(adjacencies.shape)
    
    return features_list, adjacencies, features, labels, eigenvalues, eigenvectors, laplacians


def build_data_loader(args, idx_pair, adjacencies, features, labels, eigenvalues, eigenvectors, laplacians, P_n):
    idx_train, idx_test = idx_pair

    if args.model == 'svm' or args.model == 'mlp' or args.model == 'gcn' or args.model == 'gat' or args.model == 'gdc':  
        data_train = TensorDataset(adjacencies[idx_train], features[idx_train], labels[idx_train])
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test])
    elif args.model == 'graphheat' or args.model == 'exact':
        data_train = TensorDataset(adjacencies[idx_train], features[idx_train], labels[idx_train], eigenvalues[idx_train], eigenvectors[idx_train])
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test], eigenvalues[idx_test], eigenvectors[idx_test])
    elif args.model == 'asap':
        data_train = TensorDataset(adjacencies[idx_train], features[idx_train], labels[idx_train], eigenvalues[idx_train], eigenvectors[idx_train], laplacians[idx_train], P_n[idx_train])
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test], eigenvalues[idx_test], eigenvectors[idx_test], laplacians[idx_test], P_n[idx_test])
    
    data_loader_train = DataLoader(data_train, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
    data_loader_test = DataLoader(data_test, batch_size=idx_test.shape[0], shuffle=True) # Full-batch
    
    return data_loader_train, data_loader_test