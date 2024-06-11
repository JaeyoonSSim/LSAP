import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from models.exact import *
from models.asap import *
from models.gcn import ANS
from models.gat import GAT
from models.mlp import MLP
from models.gdc import gdc
from sklearn.svm import SVC

def select_model(args, A, X, y):
    if args.model == 'exact':
        if args.layer_num == 1:
            model = Exact1(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = Exact2(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
            # model = GAPHeat2_w(adj_dim=A.shape[0],
            #             in_dim=X.shape[1],
            #             hid_dim=args.hidden_units,
            #             out_dim=torch.max(y).item() + 1,
            #             dropout=args.dropout_rate,
            #             adj_size = A.shape[0],
            #             args = args)
        elif args.layer_num == 3:
            model = Exact3(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = Exact4(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
    elif args.model == 'asap':
        if args.layer_num == 1:
            model = ASAP1(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = ASAP2(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
            # model = GAPHeat2_w(adj_dim=A.shape[0],
            #             in_dim=X.shape[1],
            #             hid_dim=args.hidden_units,
            #             out_dim=torch.max(y).item() + 1,
            #             dropout=args.dropout_rate,
            #             adj_size = A.shape[0],
            #             args = args)
        elif args.layer_num == 3:
            model = ASAP3(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = ASAP4(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
    elif args.model == 'graphheat' or args.model == 'gcn' or args.model == 'gdc':
        model = ANS(adj_dim=A.shape[0],
                    in_dim=X.shape[1],
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(y).item() + 1,
                    dropout=args.dropout_rate)
    elif args.model == 'gat':
        model = GAT(nfeat=X.shape[1],
                    nhid=args.hidden_units,
                    nclass=torch.max(y).item() + 1,
                    dropout=args.dropout_rate,
                    alpha=args.alpha,
                    adj_sz=A[0].shape[0],
                    nheads=args.num_head_attentions)
    elif args.model == 'mlp':
        model = MLP(in_feats=X.shape[1],
                    hid_feats=args.hidden_units,
                    out_feats=torch.max(y).item() + 1)
    elif args.model == 'svm':
        model = SVC(kernel='linear')
    
    return model
        
def select_optimizer(args, model):
    if args.model != 'svm':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = None
    
    return optimizer

def select_trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test):
    if args.model == 'exact':
        trainer = Exact_Trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    elif args.model == 'asap':
        trainer = ASAP_Trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    elif args.model == 'graphheat':
        trainer = GraphHeat_Trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    elif args.model == 'gat' or args.model == 'gcn' or args.model == 'gdc':
        A = 0.5 * (A.transpose(0,1) + A)
        A[A != 0] = 1
        if args.model == 'gdc':
            A = gdc(A, device, 0.2, 1e-5)
        trainer = GNN_Trainer(args, device, model, optimizer, A, X, y, idx_train, idx_val, idx_test)
    elif args.model == 'mlp':
        trainer = MLP_Trainer(args, model, optimizer, X, y, idx_train, idx_val, idx_test)
    elif args.model == 'svm':
        trainer = SVM_Trainer(args, model, X, y, idx_train, idx_val, idx_test)
    
    return trainer