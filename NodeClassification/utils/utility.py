import numpy as np
import scipy.sparse as sp
import torch

from torch.linalg import eigh

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    
    return labels_onehot

def speye(n):
    return torch.sparse_coo_tensor(
        torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)

def spnorm(A, eps=1e-5):
    D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -0.5
    indices = A._indices()
    return torch.sparse_coo_tensor(indices, D[indices[0]] * D[indices[1]])

def dot(x, y):
    return (x.unsqueeze(-2) @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def count_subgraphs(src, dst, n):
    val = torch.arange(n)
    for _ in range(100):
        idx = val[src] < val[dst]
        val[src[idx]] = val[dst[idx]]
    return val.unique().shape[0]

def build_knn_graph(x, base, k=4, b=512, ignore_self=False):
    n = x.shape[0]
    weight = torch.zeros((n, k))
    adj = torch.zeros((n, k), dtype=int)
    for i in range(0, n, b):
        knn = (
            (x[i:i+b].unsqueeze(1) - base.unsqueeze(0))
            .norm(dim=2)
            .topk(k + int(ignore_self), largest=False))
        val = knn.values[:, 1:] if ignore_self else knn.values
        idx = knn.indices[:, 1:] if ignore_self else knn.indices
        val = torch.softmax(-val, dim=-1)
        weight[i:i+b] = val
        adj[i:i+b] = idx
    return weight, adj

def compute_eigen_decomposition(A):
    A = 0.5 * (A.transpose(0,1) + A)
    A[A != 0] = 1
    
    D = torch.diag(torch.sum(A, axis=1))
    L = D - A
    
    D_inverse_sqrt = torch.linalg.inv(torch.sqrt(D))
    L_normalized = torch.matmul(torch.matmul(D_inverse_sqrt, L), D_inverse_sqrt)
    
    eigenvalue, eigenvector = eigh(L_normalized)
    
    return eigenvalue, eigenvector, L_normalized