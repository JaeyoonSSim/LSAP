import datetime
import torch
import dgl
import dgl.nn
import sys
import pickle as pkl
import networkx as nx

from utils.utility import *

def load_data_CCP(dataset_str = 'cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/planetoid/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :] # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()
    
    A = torch.FloatTensor(adj.to_dense())
    X = torch.FloatTensor(np.array(features.todense()))
    Y = torch.LongTensor(np.argmax(labels, -1))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return A, X, Y, idx_train, idx_val, idx_test

def load_data_AAC(g_data = 'amazon-com'):
    if g_data == 'weekday':
        startdate = datetime.date(1980, 1, 1)
        enddate = datetime.date(2020, 1, 1)
        delta = datetime.timedelta(days=1)
        fmt = '%Y%m%d'
        node_features, node_labels = [], []
        while startdate < enddate:
            node_labels.append(startdate.weekday())
            node_features.append([float(c) for c in startdate.strftime(fmt)])
            startdate += delta
        node_features = torch.tensor(node_features)
        node_labels = torch.tensor(node_labels, dtype=int)
        n_nodes = node_features.shape[0]
        _, adj = build_knn_graph(
            node_features, node_features, ignore_self=True)
        src = torch.arange(adj.shape[0]).repeat(adj.shape[1])
        dst = torch.cat([adj[:, i] for i in range(adj.shape[1])], dim=0)
        graph = dgl.graph((src.cpu(), dst.cpu()))
        graph.ndata['feat'] = node_features.cpu()
        graph.ndata['label'] = node_labels.cpu()
    else:
        graph = (
            dgl.data.CoraGraphDataset()[0] if g_data == 'cora'
            else dgl.data.CiteseerGraphDataset()[0] if g_data == 'citeseer'
            else dgl.data.PubmedGraphDataset()[0] if g_data == 'pubmed'
            else dgl.data.CoauthorCSDataset()[0] if g_data == 'coauthor-cs'
            else dgl.data.CoauthorPhysicsDataset()[0] if g_data == 'coauthor-phy'
            else dgl.data.RedditDataset()[0] if g_data == 'reddit'
            else dgl.data.AmazonCoBuyComputerDataset()[0] if g_data == 'amazon-com'
            else dgl.data.AmazonCoBuyPhotoDataset()[0] if g_data == 'amazon-photo'
            else None
        )

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    src, dst = graph.edges()

    n_nodes = node_features.shape[0]
    n_edges = src.shape[0]
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)
    print('nodes: %d' % n_nodes)
    print('features: %d' % n_features)
    print('classes: %d' % n_labels)
    print('edges: %d' % ((n_edges - (src == dst).sum().item())))
    print('degree: %.2f' % ((1 if g_data == 'weekday' else 2) * n_edges / n_nodes))
    print('subgraphs:', count_subgraphs(src, dst, n_nodes))
    print('intra_rate: %.2f%%' % (
        100 * (node_labels[src] == node_labels[dst]).sum().float() / n_edges))
    
    A = spnorm(graph.adj() + speye(n_nodes), eps=0)
    A = A.to_dense()
    X = node_features
    Y = node_labels
    
    idx = torch.randperm(n_nodes)
    val_num = test_num = int(n_nodes * (1 - 0.1 * 6) / 2)
    
    train_mask = torch.zeros(n_nodes, dtype=bool)
    valid_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)

    train_mask[idx[val_num + test_num:]] = True
    valid_mask[idx[:val_num]] = True
    test_mask[idx[val_num:val_num + test_num]] = True
    
    idx_train = idx[val_num + test_num:]
    idx_val = idx[:val_num]
    idx_test = idx[val_num:val_num + test_num]

    return A, X, Y, idx_train, idx_val, idx_test