import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from models.exact import *
from GraphClassification.models.lsap import *
from models.gcn import DDNet
from models.gat import GAT
from models.mlp import MLP
from models.gdc import gdc
from sklearn.svm import SVC

def select_model(args, num_ROI_features, num_used_features, adjacencies, labels):
    if args.model == 'svm':
        model = SVC(kernel='linear')
    elif args.model == 'mlp':
        model = MLP(in_feats = num_ROI_features,
                    hid_feats = args.hidden_units,
                    out_feats = torch.max(labels).item() + 1)
    elif args.model == 'exact':
        if args.layer_num == 1:
            model = Exact1(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = Exact2(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 3:
            model = Exact3(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = Exact4(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
    elif args.model == 'asap':
        if args.layer_num == 1:
            model = LSAP1(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = LSAP2(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 3:
            model = LSAP3(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = LSAP4(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
    elif args.model == 'gcn' or args.model == 'gdc' or args.model == 'graphheat':
        """
        ajd_dim: # ROI features (edges) 
        in_dim: # used features (nodes)
        hid_dim: # hidden units (weights)
        out_dim: # labels (classes)
        """
        model = DDNet(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif args.model == 'gat':
        """
        nfeat: # used features (nodes)
        nhid: # hidden units (weights)
        nclass: # labels (classes)
        """
        model = GAT(nfeat=num_used_features,
                    nhid=args.hidden_units,
                    nclass=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate,
                    alpha=args.alpha,
                    adj_sz=adjacencies[0].shape[0],
                    nheads=args.num_head_attentions)
    
    return model
        
def select_optimizer(args, model):
    if args.model != 'svm':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = None
    
    return optimizer

def select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, adjacencies):
    if args.model == 'svm':
        trainer = SVM_Trainer(args, device, model, data_loader_train, data_loader_test)
    elif args.model == 'mlp':
        trainer = MLP_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test)
    elif args.model == 'gcn' or args.model == 'gat' or args.model == 'gdc':
        trainer = GNN_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test)
    elif args.model == 'graphheat':
        trainer = GraphHeat_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    elif args.model == 'exact':
        trainer = Exact_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    elif args.model == 'asap':
        trainer = LSAP_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    
    return trainer