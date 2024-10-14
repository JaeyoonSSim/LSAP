import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from models.exact import *
from models.lsap import *

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
    elif args.model == 'lsap':
        if args.layer_num == 1:
            model = LSAP1(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = LSAP2(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 3:
            model = LSAP3(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = LSAP4(adj_dim=A.shape[0],
                        in_dim=X.shape[1],
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(y).item() + 1,
                        dropout=args.dropout_rate)
    else:
        raise ValueError
    
    return model
        
def select_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return optimizer

def select_trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test):
    if args.model == 'exact':
        trainer = EXACT_Trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    elif args.model == 'lsap':
        trainer = LSAP_Trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, A.shape[0])
    else:
        raise ValueError

    return trainer