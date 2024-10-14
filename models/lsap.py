import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import iv

from torch.nn.parameter import Parameter
from utils.utility import *

#######################################################################################
####################################### LSAP-C ########################################
########################## Chebyshev polynomial approximation #########################
#######################################################################################
def chebyshev_recurrence(Pf, Pf_old, L, b):
    T_n = (L @ Pf * 4) / b - (2 * Pf) - Pf_old
    T_n = 0.5 * (T_n.T + T_n)
    
    return T_n

def compute_Tn(L, m, b, device):
    fir = torch.zeros_like(L).to(device)
    sec = torch.eye(L.shape[1]).to(device)
    T_n = [sec]

    T_0 = (L @ sec) * 2 / b - sec - fir
    T_0 = 0.5 * (T_0.T + T_0)
    fir, sec = sec, T_0

    ### Chebyshev polynomial T_n
    for n in range(1, m + 1):
        T_n.append(sec)
        fir, sec = sec, chebyshev_recurrence(sec, fir, L, b)

    del fir, sec
    
    return T_n

def compute_heat_kernel_chebyshev(args, T_n, m, t, b, device):
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    che = torch.zeros_like(T_n[0])
    che_grad = torch.zeros_like(T_n[0])

    t = t.reshape(-1, 1)
    z = t * b / 2 
    ez = torch.exp(-z) 
    deg = torch.arange(0, m + 1).to(device) 
    ivd = iv(deg.cpu(), z.cpu()).to(device) 
    
    coeff = 2 * ((-1) ** deg) * ez * ivd 
    coeff_grad = b * ((-1) ** deg) * ez * ((iv(deg.cpu() - 1, z.cpu()).to(device) - (deg / z) * ivd) - ivd)
    
    ### Chebyshev polynomial expansion of the solution to heat diffusion 
    for n in range(0, m + 1):
        coef = coeff[:,n].unsqueeze(dim=-1)
        che += coef * T_n[n]
        che_grad += coeff_grad[:,n] * T_n[n]

    che[che < args.hk_threshold] = 0
    che_sign = torch.zeros_like(che)
    che_sign[che >= args.hk_threshold] = 1
    che_grad = torch.mul(che_grad, che_sign)
    
    hk_list.append(che)
    hk_grad_list.append(che_grad)
            
    hk_list = torch.stack(hk_list).squeeze()
    hk_grad_list = torch.stack(hk_grad_list).squeeze()

    del t, z, ez, deg, ivd
    
    return hk_list, hk_grad_list



#######################################################################################
####################################### LSAP-H ########################################
########################### Hermite polynomial approximation ##########################
#######################################################################################
def hermite_recurrence(n, Pf, Pf_old, L):
    H_n = (2 * L @ Pf) - (2 * n * Pf_old)
    H_n = 0.5 * (H_n.T + H_n)
    return H_n

def compute_Hn(L, m, device):
    fir = torch.zeros_like(L).to(device)
    sec = torch.eye(L.shape[1]).to(device)
    H_n = [sec]
    
    H_0 = 2 * L @ sec
    H_0 = 0.5 * (H_0.T + H_0)
    fir, sec = sec, H_0
    
    ### Hermite polynomial H_n
    for n in range(1, m + 1):
        H_n.append(sec)
        fir, sec = sec, hermite_recurrence(n, sec, fir, L)
        print(sec)
        print(type(sec[0][0].item()))
    
    del fir, sec
    
    return H_n
    
def compute_heat_kernel_hermite(args, H_n, m, t, device):
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel
        
    her = torch.zeros_like(H_n[0])
    her_grad = torch.zeros_like(H_n[0])

    t = t.reshape(-1, 1)
    z = (t ** 2) / 4
    ez = torch.exp(z)
    deg = torch.arange(0, m + 1).to(device)
    coeff = ((-t / 2) ** deg) * ez 
    coeff_grad = ez * (t / 2) * ((-t / 2) ** deg) * ((2 * deg) / (t ** 2) + 1)
    
    her += coeff[:,0].unsqueeze(dim=-1) * H_n[0]
    her_grad += coeff_grad[:,0] * H_n[0]
    
    ### Hermite polynomial expansion of the solution to heat diffusion 
    fac = 1
    for n in range(1, m + 1):
        fac *= n
        coef = coeff[:,n].unsqueeze(dim=-1)
        her += (1 / fac) * coef * H_n[n]
        her_grad += (1 / fac) * coeff_grad[:,n] * H_n[n]
        
    her[her < args.hk_threshold] = 0

    her_sign = torch.zeros_like(her)
    her_sign[her >= args.hk_threshold] = 1
    her_grad = torch.mul(her_grad, her_sign)
    
    hk_list.append(her)
    hk_grad_list.append(her_grad)

    hk_list = torch.stack(hk_list).squeeze()
    hk_grad_list = torch.stack(hk_grad_list).squeeze()
    
    del t, z, ez, deg
    
    return hk_list, hk_grad_list



#######################################################################################
####################################### LSAP-L ########################################
########################## Laguerre polynomial approximation ##########################
#######################################################################################
def laguerre_recurrence(n, Pf, Pf_old, L):
    L_n = (-L @ Pf + (2 * n + 1) * Pf - n * Pf_old) / (n + 1)
    L_n = 0.5 * (L_n.T + L_n)
    return L_n

def compute_Ln(L, m, device):
    fir = torch.zeros_like(L).to(device)
    sec = torch.eye(L.shape[1]).to(device)
    L_n = [sec]
    
    L_0 = sec - L @ sec
    L_0 = 0.5 * (L_0.T + L_0)
    fir, sec = sec, L_0
    
    ### Laguerre polynomial L_n
    for n in range(1, m + 1):
        L_n.append(sec)
        fir, sec = sec, laguerre_recurrence(n, sec, fir, L)
        
    del fir, sec
        
    return L_n

def compute_heat_kernel_laguerre(args, L_n, m, t, device):
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel
        
    lag = torch.zeros_like(L_n[0])
    lag_grad = torch.zeros_like(L_n[0])

    t = t.reshape(-1, 1)
    deg = torch.arange(0, m + 1).to(device)
    coeff = (t ** deg) / ((t + 1) ** (deg + 1))
    coeff_grad = coeff * (deg - t) / (t * (t + 1))

    ### Laguerre polynomial expansion of the solution to heat diffusion 
    for n in range(0, m + 1):
        coef = coeff[:,n].unsqueeze(dim=-1)
        lag += coef * L_n[n]
        lag_grad += coeff_grad[:,n] * L_n[n]

    lag[lag < args.hk_threshold] = 0

    lag_sign = torch.zeros_like(lag)
    lag_sign[lag >= args.hk_threshold] = 1
    lag_grad = torch.mul(lag_grad, lag_sign)

    hk_list.append(lag)
    hk_grad_list.append(lag_grad)
    
    hk_list = torch.stack(hk_list).squeeze()
    hk_grad_list = torch.stack(hk_grad_list).squeeze()
    
    del t, deg
    
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

class LSAP1(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP1, self).__init__()
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

class LSAP2(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP2, self).__init__()
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

class LSAP3(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP3, self).__init__()
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

class LSAP4(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(LSAP4, self).__init__()
        self.dropout = dropout
        self.gcn = GCN4(in_dim, hid_dim, out_dim, dropout) # Graph convolution

    def forward(self, x, adj): 
        x = self.gcn.forward(x, adj)
        
        return F.log_softmax(x, dim=1)










# ### 2 Layer w/o weights
class GraphConvolution2_w(nn.Module):
    def __init__(self, in_features, out_features, adj_size, args):
        super(GraphConvolution2_w, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        #self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.t = Parameter(torch.empty(adj_size).fill_(-2.))
        #self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        #self.bias.data.uniform_(-stdv, stdv)
        pass
        
    def forward(self, x, P_n, eigmax): # Graph convolution
        #support = torch.matmul(x, self.weight)
        #output = torch.matmul(adj, support)
        if self.args.polynomial == 0: # Chebyshev
            heat_kernel, heat_kernel_grad = compute_heat_kernel_chebyshev(P_n, self.args.m_chebyshev, self.t, eigmax, self.args.device_num, self.args.hk_threshold)
        elif self.args.polynomial == 1: # Hermite
            heat_kernel, heat_kernel_grad = compute_heat_kernel_hermite(P_n, self.args.m_hermite, self.t, self.args.device_num, self.args.hk_threshold)
        elif self.args.polynomial == 2: # Laguerre
            heat_kernel, heat_kernel_grad = compute_heat_kernel_laguerre(P_n, self.args.m_laguerre, self.t, self.args.device_num, self.args.hk_threshold)
        
        #support = torch.matmul(x, self.weight)
        output = torch.matmul(heat_kernel, x)
        #print(self.t[0])
        return output, self.t


class GCN2_w(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, adj_size, args):
        super(GCN2_w, self).__init__()
        self.gc1 = GraphConvolution2_w(in_dim, hid_dim, adj_size, args) 
        self.gc2 = GraphConvolution2_w(hid_dim, out_dim, adj_size, args) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, P_n, eigmax): # Graph convolution part 
        ### Graph convolution 1
        tmp, t = self.gc1.forward(x, P_n, eigmax)
        # import pdb
        # pdb.set_trace()
        tmp = torch.tensor(tmp, requires_grad=True)
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
        x3, t = self.gc2.forward(x2, P_n, eigmax)
        x4 = F.relu(x3)
        #x4 = x3
        
        if self.training:
            self.final_conv_acts = x3
            #self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

        return x4, t


class ASAP2_w(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout, adj_size, args):
        super(ASAP2_w, self).__init__()
        self.dropout = dropout
        self.gcn = GCN2_w(in_dim, hid_dim, out_dim, dropout, adj_size, args) # Graph convolution

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, P_n, eigmax): 
        x, t = self.gcn.forward(x, P_n, eigmax)
        
        return F.log_softmax(x, dim=1), t


# class GraphConvolution2_w(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GraphConvolution2_w, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         #self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         self.bias = Parameter(torch.FloatTensor(out_features))

#         self.reset_parameters()

#     def reset_parameters(self): # Initialize weights and bias
#         #stdv = 1. / math.sqrt(self.weight.size(1))
#         #self.weight.data.uniform_(-stdv, stdv)
#         #self.bias.data.uniform_(-stdv, stdv)
#         pass

#     def forward(self, x, adj): # Graph convolution
#         #support = torch.matmul(x, self.weight)
#         output = torch.matmul(adj, x)

#         return output


# class GCN2_w(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, dropout):
#         super(GCN2_w, self).__init__()
#         self.gc1 = GraphConvolution2_w(in_dim, hid_dim) 
#         self.gc2 = GraphConvolution2_w(hid_dim, out_dim) 
#         self.dropout = dropout
#         self.f = None
#         self.rdp = None
#         self.rdp2 = None
#         self.final_conv_acts = None
#         self.final_conv_grads = None

#     def activations_hook(self, grad):
#         self.final_conv_grads = grad

#     def forward(self, x, adj): # Graph convolution part 
#         ### Graph convolution 1
#         tmp = self.gc1.forward(x, adj)
#         x2 = F.relu(tmp)

#         with torch.no_grad():
#             x_one = torch.ones_like(x2)
#             x_zero = torch.zeros_like(x2)
#             self.rdp = torch.where(tmp <= 0., x_zero, x_one)
#             tmp = x2

#         x2 = F.dropout(x2, self.dropout, training=self.training)

#         with torch.no_grad():
#             x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
#             x_zero = torch.zeros_like(x2)
#             self.rdp = torch.mul(self.rdp, torch.where((tmp != 0.) & (x2 == 0.), x_zero, x_one))

#         self.f = x2
        
#         ### Graph convolution 2
#         x3 = self.gc2.forward(x2, adj)
#         #x4 = F.relu(x3)
#         x4 = x3
        
#         if self.training:
#             self.final_conv_acts = x3
#             #self.final_conv_acts.register_hook(self.activations_hook)

#         with torch.no_grad():
#             x_one = torch.ones_like(x4)
#             x_zero = torch.zeros_like(x4)
#             self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

#         return x4


# class GAPHeat2_w(nn.Module):
#     def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
#         super(GAPHeat2_w, self).__init__()
#         self.dropout = dropout
#         self.gcn = GCN2_w(in_dim, hid_dim, out_dim, dropout) # Graph convolution

#     """
#     x: (# subjects, # ROI features, # used features)
#     adj: (# subjects, # ROI features, # ROI features)
#     """
#     def forward(self, x, adj): 
#         x = self.gcn.forward(x, adj)
        
#         return F.log_softmax(x, dim=1)
    