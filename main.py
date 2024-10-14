import numpy as np
import torch
import argparse
import random
import time

from utils.utility import *
from utils.loader import *
from utils.model import *
from utils.train import *

def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', type=str, default='cora', help='Type of dataset (cora / citeseer / pubmed / amazon-com / amazon-photo / coauthor-cs)')
    ### Condition
    parser.add_argument('--seed_num', type=int, default=42, help='Number of random seed')
    parser.add_argument('--device_num', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--model', type=str, default='lsap', help='Models to use')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    parser.add_argument('--val_mode', type=int, default=1, help='Validation mode')
    ### Experiment
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--hidden_units', type=int, default=32, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learing rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 loss on parameters')
    ### Experiment for "Exact" and "LSAP"
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_lr', type=float, default=1, help='t learning rate')
    parser.add_argument('--t_loss_threshold', type=float, default=0.01, help='t loss threshold')
    parser.add_argument('--t_lambda', type=float, default=1, help='t lambda of loss function')
    parser.add_argument('--t_threshold', type=float, default=0.1, help='t threshold')
    parser.add_argument('--hk_threshold', type=float, default=1e-5, help='Heat kernel threshold')
    ### Experiment for "LSAP"
    parser.add_argument('--polynomial', type=int, default=2, help='Which polynomial is used (0:Chebyshev / 1:Hermite / 2:Laguerre)')
    parser.add_argument('--m_chebyshev', type=int, default=20, help='Expansion degree of Chebyshev')
    parser.add_argument('--m_hermite', type=int, default=30, help='Expansion degree of Hermite')
    parser.add_argument('--m_laguerre', type=int, default=20, help='Expansion degree of Laguerre')
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
    device = torch.device('cuda:' + str(args.device_num) if torch.cuda.is_available() else 'cpu')
    
    """
    A: adjacency (# nodes, # nodes)
    X: feature (# nodes, # features)
    y: label (# nodes)
    < CORA, CITESEER, PUBMED >
    idx_train: 140 / 120 / 60
    idx_val: 500 / 500 / 500
    idx_test: 1000 / 1000 / 1000
    """
    dataset = args.data
    if dataset =='cora' or dataset == 'citeseer' or dataset == 'pubmed':
        A, X, y, idx_train, idx_val, idx_test = load_data_CCP(dataset)
    # elif dataset == 'amazon-com' or dataset == 'amazon-photo' or dataset == 'coauthor-cs':
    #     A, X, y, idx_train, idx_val, idx_test = load_data_AAC(dataset)
    
    print(idx_train.shape, idx_val.shape, idx_test.shape)
    
    model = select_model(args, A, X, y) 
    optimizer = select_optimizer(args, model)
    eigenvalue, eigenvector, laplacian = compute_eigen_decomposition(A)
    
    if torch.cuda.is_available() and args.model != 'svm':
        model = model.to(device)
        A = A.to(device)
        X = X.to(device)
        y = y.to(device)
        
        eigenvalue = eigenvalue.to(device)
        eigenvector = eigenvector.to(device)
        laplacian = laplacian.to(device)
        
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    
    trainer = select_trainer(args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test)
    trainer.train()
    _ = trainer.test()

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")