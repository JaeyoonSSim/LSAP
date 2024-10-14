import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from utils.utility import *

def compute_heat_kernel(args, eigenvalue, eigenvector, t):
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    tmp = eigenvalue.type(torch.float) # (# of ROIs)
    one_tmp = torch.ones_like(eigenvector) # (# of ROIs, # of ROIs)
    eigval = torch.mul(one_tmp, torch.exp(-tmp).reshape(1, -1)) ** t.reshape(-1, 1) # (# of ROIs, # of ROIs)
    eigvec = eigenvector.type(torch.float) # (# of ROIs, # of ROIs)
    left = torch.mul(eigvec, eigval)
    right = eigvec.T
    hk = torch.matmul(left, right) # Compute heat kernel (# of ROIs, # of ROIs)
    hk[hk < args.hk_threshold] = 0
    
    hk_grad = torch.matmul(torch.matmul(left, -torch.diag(tmp)), right) # Compute gradient of heat kernel (# of ROIs, # of ROIs)
    hk_one = torch.ones_like(hk) 
    hk_zero = torch.zeros_like(hk) 
    hk_sign = torch.where(hk >= args.hk_threshold, hk_one, hk_zero)  
    hk_grad = torch.mul(hk_grad, hk_sign)

    hk_list.append(hk)
    hk_grad_list.append(hk_grad)
    
    hk_list = torch.stack(hk_list).squeeze()
    hk_grad_list = torch.stack(hk_grad_list).squeeze()
    
    return hk_list, hk_grad_list

### 1 Layer
class GraphConvolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution1(in_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        x1 = self.gc1.forward(x, adj)
        #x2 = F.relu(x1)
        x2 = x1
        
        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(x2 <= 0., x_zero, x_one)
        
        self.f = x

        return x2

class Exact1(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(Exact1, self).__init__()
        self.dropout = dropout
        self.gcn = GCN1(in_dim, hid_dim, out_dim, dropout) # Graph convolution
        
    def forward(self, x, adj): 
        x = self.gcn.forward(x, adj)
        
        return F.log_softmax(x, dim=1)





### 2 Layer
class GraphConvolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution2(in_dim, hid_dim) 
        self.gc2 = GraphConvolution2(hid_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        tmp = self.gc1.forward(x, adj)
        x2 = F.relu(tmp)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(tmp <= 0., x_zero, x_one)
            tmp = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((tmp != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)
        #x4 = F.relu(x3)
        x4 = x3
        
        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

        return x4

class Exact2(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(Exact2, self).__init__()
        self.dropout = dropout
        self.gcn = GCN2(in_dim, hid_dim, out_dim, dropout) # Graph convolution

    def forward(self, x, adj): 
        x = self.gcn.forward(x, adj)
        
        return F.log_softmax(x, dim=1)





### 3 Layer
class GraphConvolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN3(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN3, self).__init__()
        self.gc1 = GraphConvolution3(in_dim, hid_dim) 
        self.gc2 = GraphConvolution3(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution3(hid_dim, out_dim)
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        x1 = self.gc1.forward(x, adj)
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(x1 <= 0., x_zero, x_one)
            x1 = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((x1 != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)
        x4 = F.relu(x3)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)
            
        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)
            x3 = x4

        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        #x6 = F.relu(x5)
        x6 = x5

        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)

        return x6

class Exact3(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(Exact3, self).__init__()
        self.dropout = dropout
        self.gcn = GCN3(in_dim, hid_dim, out_dim, dropout) # Graph convolution

    def forward(self, x, adj): 
        x = self.gcn.forward(x, adj)
        
        return F.log_softmax(x, dim=1)





### 4 Layer
class GraphConvolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN4(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN4, self).__init__()
        self.gc1 = GraphConvolution4(in_dim, hid_dim) 
        self.gc2 = GraphConvolution4(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution4(hid_dim, hid_dim)
        self.gc4 = GraphConvolution4(hid_dim, out_dim)
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.f3 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
        self.rdp4 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        x1 = self.gc1.forward(x, adj)
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(x1 <= 0., x_zero, x_one)
            x1 = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((x1 != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)
        x4 = F.relu(x3)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)
            
        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)
            x3 = x4

        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        x6 = F.relu(x5)
        
        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)
            x5 = x6
            
        x6 = F.dropout(x6, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp2 = torch.mul(self.rdp3, torch.where((x5 != 0.) & (x6 == 0.), x_zero, x_one))

        self.f3 = x6
        
        ### Graph convolution 4
        x7 = self.gc4.forward(x6, adj)
        x8 = x7
        
        if self.training:
            self.final_conv_acts = x7
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x8)
            x_zero = torch.zeros_like(x8)
            self.rdp4 = torch.where(x8 <= 0., x_zero, x_one)

        return x8

class Exact4(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(Exact4, self).__init__()
        self.dropout = dropout
        self.gcn = GCN4(in_dim, hid_dim, out_dim, dropout) # Graph convolution

    def forward(self, x, adj): 
        x = self.gcn.forward(x, adj)
        
        return F.log_softmax(x, dim=1)