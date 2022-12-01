import torch

from scipy.special import iv

#######################################################################################
####################################### Exact #########################################
############################ Exact heat kernel computation ############################
#######################################################################################
def compute_heat_kernel(eigenvalue, eigenvector, t):
    hk_threshold = 1e-5
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    num_samples = eigenvalue.shape[0]

    for i in range(num_samples):
        tmp = eigenvalue[i].type(torch.float) # (# of ROIs)
        one_tmp = torch.ones_like(eigenvector[i]) # (# of ROIs, # of ROIs)
        eigval = torch.mul(one_tmp, torch.exp(-tmp).reshape(1, -1)) ** t.reshape(-1, 1) # (# of ROIs, # of ROIs)
        eigvec = eigenvector[i].type(torch.float) # (# of ROIs, # of ROIs)
        left = torch.mul(eigvec, eigval)
        right = eigvec.T
        hk = torch.matmul(left, right) # Compute heat kernel (# of ROIs, # of ROIs)
        hk[hk < hk_threshold] = 0
        
        hk_grad = torch.matmul(torch.matmul(left, -torch.diag(tmp)), right) # Compute gradient of heat kernel (# of ROIs, # of ROIs)
        hk_one = torch.ones_like(hk) 
        hk_zero = torch.zeros_like(hk) 
        hk_sign = torch.where(hk >= hk_threshold, hk_one, hk_zero)  
        hk_grad = torch.mul(hk_grad, hk_sign)

        hk_list.append(hk)
        hk_grad_list.append(hk_grad)
        
    hk_list = torch.stack(hk_list)
    hk_grad_list = torch.stack(hk_grad_list)
    
    return hk_list, hk_grad_list



#######################################################################################
####################################### ASAP-C ########################################
########################## Chebyshev polynomial approximation #########################
#######################################################################################
def chebyshev_recurrence(Pf, Pf_old, L, b):
    output = (L @ Pf * 4) / b - (2 * Pf) - Pf_old
    output = 0.5 * (output.T + output)
    return output

def compute_Tn(L, m, b, device):
    num_samples = L.shape[0]
    
    T_n_list = []
    for i in range(num_samples):
        fir = torch.zeros_like(L[i]).to(device)
        sec = torch.eye(L.shape[1]).to(device)
        T_n = [sec]

        T_0 = (L[i] @ sec * 2) / b[i] - sec - fir
        T_0 = 0.5 * (T_0.T + T_0)
        fir, sec = sec, T_0
        
        ### Chebyshev polynomial T_n
        for n in range(1, m + 1):
            T_n.append(sec)
            fir, sec = sec, chebyshev_recurrence(sec, fir, L[i], b[i])
        
        T_n_list.append(torch.stack(T_n))
        
    return torch.stack(T_n_list)

def compute_heat_kernel_chebyshev(P_n, L, m, t, b, device):
    hk_threshold = 1e-5
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    num_samples = L.shape[0]
    
    for j in range(num_samples):
        T_n = P_n[j]
        che = torch.zeros_like(T_n[0])
        che_grad = torch.zeros_like(T_n[0])

        t = t.reshape(-1, 1) # (160, 1)
        z = t * b[j] / 2 # (160, 1)
        ez = torch.exp(-z) # (160, 1)
        deg = torch.arange(0, m + 1).to(device) # (31)
        ivd = iv(deg.cpu(), z.cpu()).to(device) # (160, 31)

        coeff = 2 * ((-1) ** deg) * ez * ivd # (160, 31)
        coeff_grad = b[j] * ((-1) ** deg) * ez * ((iv(deg.cpu() - 1, z.cpu()).to(device) - (deg / z) * ivd) - ivd)

        ### Chebyshev polynomial expansion of the solution to heat diffusion 
        for n in range(0, m + 1):
            coef = coeff[:,n].unsqueeze(dim=-1)
            che += coef * T_n[n]
            che_grad += coeff_grad[:,n] * T_n[n]

        che[che < hk_threshold] = 0
        che_sign = torch.zeros_like(che)
        che_sign[che >= hk_threshold] = 1
        che_grad = torch.mul(che_grad, che_sign)
        
        hk_list.append(che)
        hk_grad_list.append(che_grad)
            
    hk_list = torch.stack(hk_list)
    hk_grad_list = torch.stack(hk_grad_list)

    return hk_list, hk_grad_list



#######################################################################################
####################################### ASAP-H ########################################
########################### Hermite polynomial approximation ##########################
#######################################################################################
def hermite_recurrence(n, Pf, Pf_old, L):
    output = (2 * L @ Pf) - (2 * n * Pf_old)
    output = 0.5 * (output.T + output)
    return output

def compute_Hn(L, m, device):
    num_samples = L.shape[0]
    
    H_n_list = []
    for i in range(num_samples):
        fir = torch.zeros_like(L[i]).to(device)
        sec = torch.eye(L.shape[1]).to(device)
        H_n = [sec]
        
        H_0 = 2 * L[i] @ sec
        H_0 = 0.5 * (H_0.T + H_0)
        fir, sec = sec, H_0
        
        ### Hermite polynomial H_n
        for n in range(1, m + 1):
            H_n.append(sec)
            fir, sec = sec, hermite_recurrence(n, sec, fir, L[i])
        
        H_n_list.append(torch.stack(H_n))
        
    return torch.stack(H_n_list)
    
def compute_heat_kernel_hermite(P_n, L, m, t, device):
    hk_threshold = 1e-5
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel
    
    num_samples = L.shape[0]
    
    for j in range(num_samples):
        H_n = P_n[j]
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
            
        her[her < hk_threshold] = 0

        her_sign = torch.zeros_like(her)
        her_sign[her >= hk_threshold] = 1
        her_grad = torch.mul(her_grad, her_sign)
        hk_list.append(her)
        hk_grad_list.append(her_grad)
        
    hk_list = torch.stack(hk_list)
    hk_grad_list = torch.stack(hk_grad_list)
    
    return hk_list, hk_grad_list


#######################################################################################
####################################### ASAP-L ########################################
########################## Laguerre polynomial approximation ##########################
#######################################################################################
def laguerre_recurrence(n, Pf, Pf_old, L):
    output = (-L @ Pf + (2 * n + 1) * Pf - n * Pf_old) / (n + 1)
    output = 0.5 * (output.T + output)
    return output

def compute_Ln(L, m, device):
    num_samples = L.shape[0]
    
    L_n_list = []
    for i in range(num_samples):
        fir = torch.zeros_like(L[i]).to(device)
        sec = torch.eye(L.shape[1]).to(device)
        L_n = [sec]
        
        L_0 = sec - L[i] @ sec
        L_0 = 0.5 * (L_0.T + L_0)
        fir, sec = sec, L_0
        
        ### Laguerre polynomial L_n
        for n in range(1, m + 1):
            L_n.append(sec)
            fir, sec = sec, laguerre_recurrence(n, sec, fir, L[i])
        
        L_n_list.append(torch.stack(L_n))
        
    return torch.stack(L_n_list)

def compute_heat_kernel_laguerre(P_n, L, m, t, device):
    hk_threshold = 1e-5
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    num_samples = L.shape[0]
    
    for j in range(num_samples):
        L_n = P_n[j]
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
            
        lag[lag < hk_threshold] = 0

        lag_sign = torch.zeros_like(lag)
        lag_sign[lag >= hk_threshold] = 1
        lag_grad = torch.mul(lag_grad, lag_sign)

        hk_list.append(lag)
        hk_grad_list.append(lag_grad)
        
    hk_list = torch.stack(hk_list)
    hk_grad_list = torch.stack(hk_grad_list)
    
    return hk_list, hk_grad_list