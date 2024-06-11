import torch
import sys

from scipy.special import iv

#######################################################################################
####################################### Exact #########################################
############################ Exact heat kernel computation ############################
#######################################################################################
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



#######################################################################################
####################################### ASAP-C ########################################
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
####################################### ASAP-H ########################################
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
####################################### ASAP-L ########################################
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