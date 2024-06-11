import numpy as np
import random
import torch
import argparse
import time

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from utils.train import *
from utils.loader import *
from utils.utility import *
from utils.model import *

#wandb.init(project="ADNI1", entity="jaeyoonsim")

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', default='adni', help='Type of dataset')
    parser.add_argument('--data_path', default='./data/adni_2022')
    parser.add_argument('--adjacency_path', default='/matrices2326')
    ### Condition
    parser.add_argument('--seed_num', type=int, default=100, help='Number of random seed')
    parser.add_argument('--device_num', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--features', type=int, default=2, help='Features to use')
    parser.add_argument('--labels', type=int, default=3, help="Labels to use")
    parser.add_argument('--model', type=str, default='ours', help='Models to use')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    parser.add_argument('--use_feature', type=int, default=1, help='Whether X are used (0:degree / 1:feature)')
    parser.add_argument('--use_all_features', type=int, default=0, help='Whether all X are used')
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
    
    if args.data == 'adni':
        ### Select the feature files and labels
        features_list, labels_dict = set_classification_conditions(args)
        
        ### Get dictionary for subject with ROI features
        subject_map = get_roi_feature(args, features_list, labels_dict) # {Subject : [ROI features, label]}

        ### Load the graph data
        used_features, A, X, y, eigenvalues, eigenvectors, laplacians = load_graph_data(args, features_list, subject_map)
        
        print(X.shape)
        
    ### K-fold cross validation
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)

    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(A, y):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        A = A.to(device) # Shape: (# subjects, # ROI feature, # ROI X)
        X = X.to(device) # Shape: (# subjects, # ROI X, # used X)
        y = y.to(device) # Shape: (# subjects)
    
        eigenvalues = eigenvalues.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors = eigenvectors.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        laplacians = laplacians.to(device)
    
    ### Compute polynomial
    if args.model == 'asap':
        if args.polynomial == 0: # Chebyshev
            b = eigenvalues[:,eigenvalues.shape[1]-1] 
            P_n = compute_Tn(laplacians, args.m_chebyshev, b, device)
        elif args.polynomial == 1: # Hermite
            P_n = compute_Hn(laplacians, args.m_hermite, device)
        elif args.polynomial == 2: # Laguerre
            P_n = compute_Ln(laplacians, args.m_laguerre, device)
    else:
        P_n = None
    
    if args.model != 'svm':
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2]
    else:
        num_ROI_features = None
        num_used_features = None

    for i, idx_pair in enumerate(idx_pairs):
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        ### Build data loader
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors, laplacians, P_n)

        ### Select the model to use
        model = select_model(args, num_ROI_features, num_used_features, A, y)
        optimizer = select_optimizer(args, model)
        trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A)
        #wandb.watch(model)
        
        ### Train and test
        trainer.train()
        losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, cf_sensitivities, cf_f1score, t = trainer.test()

        avl.append(losses)
        ava.append(accuracies)
        avac.append(cf_accuracies)
        avpr.append(cf_precisions)
        avsp.append(cf_specificities)
        avse.append(cf_sensitivities)
        avf1s.append(cf_f1score)
        ts.append(t)

    class_info = y.tolist()
    cnt = Counter(class_info)

    ### Show results
    print("--------------- Result ---------------")
    if args.data == 'adni':
        print(f"Used X:        {used_features}")
    print(f"Label distribution:   {cnt}")
    print(f"{args.split_num}-Fold test loss:     {avl}")
    print(f"{args.split_num}-Fold test accuracy: {ava}")
    print("---------- Confusion Matrix ----------")
    #print(f"{args.split_num}-Fold accuracy:      {avac}")
    print(f"{args.split_num}-Fold precision:     {avpr}")
    #print(f"{args.split_num}-Fold specificity:   {avsp}")
    print(f"{args.split_num}=Fold sensitivity:   {avse}")
    #print(f"{args.split_num}=Fold f1 score:      {avf1s}")
    print("-------------- Mean, Std --------------")
    #print(f"Mean: {np.mean(avl):.10f} {np.mean(ava):.10f} {np.mean(avac):.10f} {np.mean(avpr):.10f} {np.mean(avsp):.10f} {np.mean(avse):.10f} {np.mean(avf1s):.10f}")
    #print(f"Std:  {np.std(avl):.10f} {np.std(ava):.10f} {np.std(avac):.10f} {np.std(avpr):.10f} {np.std(avsp):.10f} {np.std(avse):.10f} {np.mean(avf1s):.10f}")
    print(f"Mean:  {np.mean(ava):.10f} {np.mean(avpr):.10f} {np.mean(avse):.10f}")
    print(f"Std:   {np.std(ava):.10f}  {np.std(avpr):.10f}  {np.std(avse):.10f}")
    if args.model == 'exact' or args.model == 'asap':
        print("t: ", *(np.mean(ts, axis=0)))

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")