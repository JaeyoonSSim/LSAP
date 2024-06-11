import torch.nn.functional as F

from utils.utility import *
from utils.metric import *
from utils.approximate import *
from sklearn import metrics
from torch.autograd import Variable

### Trainer for 'SVM'
class SVM_Trainer:
    def __init__(self, args, model, X, y, idx_train, idx_val, idx_test):
        self.args = args
        self.model = model
        self.X = X
        self.y = y
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
    
    def train(self):
        self.model.fit(self.X[self.idx_train], self.y[self.idx_train])
    
    def test(self):
        output = self.model.predict(self.X)
        output = torch.FloatTensor(encode_onehot(output))

        loss_test = torch.tensor([0])
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()

### Trainer for 'MLP'
class MLP_Trainer:
    def __init__(self, args, model, optimizer, X, y, idx_train, idx_val, idx_test):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.X = X
        self.y = y
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test =idx_test
        
    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()

            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
            
            output = self.model.forward(self.X) # Shape: (# of samples, # of labels)

            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train])
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:
                    if lv[-1] <= loss_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_train: {loss_train.item():.5f} accuracy_train: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} accuracy_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    break
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())

            self.model.eval()

    def test(self):
        self.model.eval()
        
        output = self.model.forward(self.X) # Shape: (# of samples, # of labels)
        
        loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test])
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} accuracy_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()
    
### Trainer for 'GCN', 'GAT', 'GDC'
class GNN_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, idx_train, idx_val, idx_test):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()

            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
            
            output = self.model.forward(self.X, self.A) # Shape: (# of samples, # of labels)

            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train])
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:
                    if lv[-1] <= loss_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_train: {loss_train.item():.5f} accuracy_train: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} accuracy_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    break
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())

            self.model.eval()

    def test(self):
        self.model.eval()
        
        output = self.model.forward(self.X, self.A) # Shape: (# of samples, # of labels)
        
        loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test])
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} accuracy_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()

### Trainer for 'GraphHeat'
class GraphHeat_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, adj_size):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        self.laplacian = laplacian
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj_size = adj_size

        if args.use_t_local == 1: 
            self.t = torch.empty(adj_size).fill_(2.)
        else:
            self.t = torch.tensor([2.])
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            self.t = self.t.to(device)

    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0

        heat_kernel, heat_kernel_grad = compute_heat_kernel(self.args, self.eigenvalue, self.eigenvector, self.t) # Use heat kernel instead of adjacency matrix

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()

            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
            
            # Use heat kernel instead of adjacency matrix
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)

            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train])
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:
                    if lv[-1] <= loss_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_tr: {loss_train.item():.5f} acc_tr: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} acc_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    break
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())

            self.model.eval()

    def test(self):
        self.model.eval()

        heat_kernel, heat_kernel_grad = compute_heat_kernel(self.args, self.eigenvalue, self.eigenvector, self.t) # Use heat kernel instead of adjacency matrix

        # Use heat kernel instead of adjacency matrix
        output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
        
        loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test])
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()


### Trainer for 'Ours'
class Exact_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, adj_size):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        self.laplacian = laplacian
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj_size = adj_size

        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        if torch.cuda.is_available():
            self.t = self.t.to(device)

    ### Scale regularization of loss function
    def t_loss(self):
        t_one = torch.abs(self.t)
        t_zero = torch.zeros_like(self.t)

        t_l = torch.where(self.t < self.t_loss_threshold, t_one, t_zero)

        return self.t_lambda * torch.sum(t_l)

    def t_deriv(self):
        t_one = self.t_lambda * torch.ones_like(self.t)
        t_zero = torch.zeros_like(self.t)

        t_de = torch.where(self.t < self.t_loss_threshold, -t_one, t_zero)

        return t_de

    def scale_update(self, output, feature, label, heat_kernel, heat_kernel_grad):
        y_oh = torch.zeros_like(output)
        y_oh.scatter_(1, label.reshape(-1, 1), 1)
        dl_dh = (torch.exp(output) - y_oh) / output.shape[0] # (# node, # class)
        
        if self.args.layer_num == 1:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp) @ self.model.gcn.gc1.weight.T
            dl_dt = torch.mul((dl_dc @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad)
        elif self.args.layer_num == 2:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp2) @ self.model.gcn.gc2.weight.T
            dl_first = torch.mul((dl_dc @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad)
            dl_second = torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp) 
                        @ torch.matmul(self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)),
                        heat_kernel_grad), heat_kernel.swapaxes(0, 1))
            dl_dt = dl_first + dl_second
        elif self.args.layer_num == 3:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp3) @ self.model.gcn.gc3.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.model.gcn.f2.swapaxes(0, 1)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.model.gcn.rdp2) 
                                        @ self.model.gcn.gc2.weight.T, self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad), 
                                        heat_kernel.swapaxes(0, 1))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp2) 
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.rdp) 
                            @ self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third
        elif self.args.layer_num == 4:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp4) @ self.model.gcn.gc4.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.model.gcn.f3.swapaxes(0, 1)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.f2.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.rdp2)
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dc_fourth = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.rdp2)
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.rdp)
                            @ self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)), heat_kernel_grad),
                            heat_kernel.swapaxes(0, 1))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third + dl_dc_fourth
        
        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=1)
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1))]).to(self.device)

        dl_dt += self.t_deriv()
        now_lr = self.t_lr * dl_dt

        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold

        self.t = self.t - now_lr # Update t

        #if self.args.use_t_local == 1:
        #    print(f't:{self.t[0].item()}', end=' ')
        #else:
        #    print(f't:{self.t.item():.4f}', end=' ')

    ### Train
    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()

            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

            heat_kernel, heat_kernel_grad = compute_heat_kernel(self.args, self.eigenvalue, self.eigenvector, self.t) # Use heat kernel instead of adjacency matrix

            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)

            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train]) + self.t_loss()
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())
            
            with torch.no_grad():
                self.scale_update(output, self.X, self.y, heat_kernel, heat_kernel_grad)
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:# or epoch < 400:
                    if lv[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1

                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_tr: {loss_train.item():.5f} acc_tr: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} acc_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    print('Epoch: ', best_epoch)
                    break
                
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
        
            self.model.eval()
        
    ### Test
    def test(self):
        self.model.eval()

        heat_kernel, heat_kernel_grad = compute_heat_kernel(self.args, self.eigenvalue, self.eigenvector, self.t) # Use heat kernel instead of adjacency matrix

        output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
        
        loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test]) + self.t_loss()
        accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()


### Trainer for approximation version of 'Ours'
class ASAP_Trainer:
    def __init__(self, args, device, model, optimizer, A, X, y, eigenvalue, eigenvector, laplacian, idx_train, idx_val, idx_test, adj_size):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.A = A
        self.X = X
        self.y = y
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        self.laplacian = laplacian
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj_size = adj_size
        self.coef = None
        self.max_acc = 0

        if args.use_t_local == 1: # Local scale
            self.t = torch.empty(adj_size).fill_(2.)
        else: # Global scale
            self.t = torch.tensor([2.]) 
        
        self.t_lr = self.args.t_lr
        self.t_loss_threshold = self.args.t_loss_threshold
        self.t_lambda = self.args.t_lambda
        self.t_threshold = self.args.t_threshold
        
        self.m_che = self.args.m_chebyshev 
        self.m_her = self.args.m_hermite 
        self.m_lag = self.args.m_laguerre
        
        if torch.cuda.is_available():
            self.t = self.t.to(device)
            

    ### Scale regularization of loss function
    def t_loss(self):
        t_one = torch.abs(self.t)
        t_zero = torch.zeros_like(self.t)

        t_l = torch.where(self.t < self.t_loss_threshold, t_one, t_zero)

        return self.t_lambda * torch.sum(t_l)

    def t_deriv(self):
        t_one = self.t_lambda * torch.ones_like(self.t)
        t_zero = torch.zeros_like(self.t)

        t_de = torch.where(self.t < self.t_loss_threshold, -t_one, t_zero)

        return t_de

    def scale_update(self, output, feature, label, heat_kernel, heat_kernel_grad):
        y_real = torch.zeros_like(output) # (# node, # class)
        y_real.scatter_(1, label.reshape(-1, 1), 1) # (# node, # class)
        dl_dh = (torch.exp(output) - y_real) / output.shape[0] # (# node, # class)

        if self.args.layer_num == 1:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp) @ self.model.gcn.gc1.weight.T
            dl_dt = torch.mul((dl_dc @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad)
        elif self.args.layer_num == 2:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp2) @ self.model.gcn.gc2.weight.T # (# node, # hidden unit)
            dl_dc_first = torch.mul((dl_dc @ self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad) # (# node, # node)
            dl_dc_second = torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp) 
                            @ torch.matmul(self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1)) # (# node, # node)
            dl_dt = dl_dc_first + dl_dc_second # (# node, # node)
        elif self.args.layer_num == 3:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp3) @ self.model.gcn.gc3.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.model.gcn.f2.swapaxes(0, 1)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.model.gcn.rdp2) 
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp2) 
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.rdp) 
                            @ self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third
        elif self.args.layer_num == 4:
            dl_dc = torch.mul(dl_dh, self.model.gcn.rdp4) @ self.model.gcn.gc4.weight.T
            dl_dc_first = torch.mul((dl_dc @ self.model.gcn.f3.swapaxes(0, 1)), heat_kernel_grad)
            dl_dc_second = torch.matmul(torch.mul(torch.matmul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.f2.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dc_third = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.rdp2)
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.f.swapaxes(0, 1)), heat_kernel_grad), 
                            heat_kernel.swapaxes(0, 1))
            dl_dc_fourth = torch.matmul(torch.mul(torch.matmul(torch.mul(torch.mul(torch.mul(dl_dc, self.model.gcn.rdp3) 
                            @ self.model.gcn.gc3.weight.T, self.model.gcn.rdp2)
                            @ self.model.gcn.gc2.weight.T, self.model.gcn.rdp)
                            @ self.model.gcn.gc1.weight.T, feature.swapaxes(0, 1)), heat_kernel_grad),
                            heat_kernel.swapaxes(0, 1))
            dl_dt = dl_dc_first + dl_dc_second + dl_dc_third + dl_dc_fourth
            
        
        if self.args.use_t_local == 1:
            dl_dt = torch.sum(dl_dt, dim=1) # (# node)
        else:
            dl_dt = torch.tensor([torch.sum(dl_dt, dim=(0, 1))]).to(self.device) # (1)

        dl_dt += self.t_deriv()
        now_lr = self.t_lr * dl_dt

        now_lr[now_lr > self.t_threshold] = self.t_threshold
        now_lr[now_lr < -self.t_threshold] = -self.t_threshold

        self.t = self.t - now_lr # Update t

    ### Train
    def train(self):
        lt = [] # List of train loss
        at = [] # List of train accuracy

        lv = [] # List of validation loss
        av = [] # List of validation accuracy
        bad_counter = 0
        
        loss_best = np.inf
        accuracy_best = 0.0
        
        loss_min = np.inf
        accuracy_max = 0.0
        
        best_epoch = 0
        
        # Compute orthogonal polynomials
        if self.args.polynomial == 0: # Chebyshev
            b = self.eigenvalue[-1]
            T_n = compute_Tn(self.laplacian, self.m_che, b, self.device)
        elif self.args.polynomial == 1: # Hermite
            H_n = compute_Hn(self.laplacian, self.m_her, self.device)
        elif self.args.polynomial == 2: # Laguerre
            L_n = compute_Ln(self.laplacian, self.m_lag, self.device)
            
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            
            self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero

            if self.args.polynomial == 0: # Chebyshev
                heat_kernel, heat_kernel_grad = compute_heat_kernel_chebyshev(self.args, T_n, self.m_che, self.t, b, self.device)
            elif self.args.polynomial == 1: # Hermite
                heat_kernel, heat_kernel_grad = compute_heat_kernel_hermite(self.args, H_n, self.m_her, self.t, self.device)
            elif self.args.polynomial == 2: # Laguerre
                heat_kernel, heat_kernel_grad = compute_heat_kernel_laguerre(self.args, L_n, self.m_lag, self.t, self.device)
            import pdb;pdb.set_trace()
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
            
            loss_train = F.nll_loss(output[self.idx_train], self.y[self.idx_train]) + self.t_loss()
            accuracy_train = compute_accuracy(output[self.idx_train], self.y[self.idx_train])
            loss_train.backward() # Computes the gradient of current tensor w.r.t. graph leaves
            
            lt.append(loss_train.item())
            at.append(accuracy_train.item())
            
            if epoch > 2:
                if lt[-1] > lt[-2] * 10:
                    print('Epoch: ', best_epoch)
                    break
            
            with torch.no_grad():
                self.scale_update(output, self.X, self.y, heat_kernel, heat_kernel_grad)
            
            self.optimizer.step() # Updates the parameters
            
            if self.args.val_mode == 1:
                loss_val = F.nll_loss(output[self.idx_val], self.y[self.idx_val])
                accuracy_val = compute_accuracy(output[self.idx_val], self.y[self.idx_val])
                
                lv.append(loss_val.item())
                av.append(accuracy_val.item())
                
                if lv[-1] <= loss_min or av[-1] >= accuracy_max:# or epoch < 400:
                    if lv[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                        loss_best = lv[-1]
                        accuracy_best = av[-1]
                        best_epoch = epoch

                    loss_min = np.min((lv[-1], loss_min))
                    accuracy_max = np.max((av[-1], accuracy_max))
                    bad_counter = 0
                else:
                    bad_counter += 1
                
                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch [{epoch} / {self.args.epochs}] loss_tr: {loss_train.item():.5f} acc_tr: {accuracy_train.item():.5f} loss_val: {loss_val.item():.5f} acc_val: {accuracy_val.item():.5f}")
                
                if bad_counter == self.args.patience:
                    self.max_acc = accuracy_max
                    print('Early stop! Min loss: ', loss_min, ', Max accuracy: ', accuracy_max)
                    print('Early stop model validation loss: ', loss_best, ', accuracy: ', accuracy_best)
                    print('Epoch: ', best_epoch)
                    break
            
            #wandb.log({"loss_train": loss_train.item(),
            #           "accuracy_train": accuracy_train.item()})
            
            self.model.eval()

    ### Test
    def test(self):
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        if self.args.polynomial == 0: # Chebyshev
            b = self.eigenvalue[-1]
            T_n = compute_Tn(self.laplacian, self.m_che, b, self.device)
            heat_kernel, _ = compute_heat_kernel_chebyshev(self.args, T_n, self.m_che, self.t, b, self.device)
        elif self.args.polynomial == 1: # Hermite
            H_n = compute_Hn(self.laplacian, self.m_her, self.device)
            heat_kernel, _ = compute_heat_kernel_hermite(self.args, H_n, self.m_her, self.t, self.device)
        elif self.args.polynomial == 2: # Laguerre
            L_n = compute_Ln(self.laplacian, self.m_lag, self.device)
            heat_kernel, _ = compute_heat_kernel_laguerre(self.args, L_n, self.m_lag, self.t, self.device)

        with torch.no_grad():
            output = self.model.forward(self.X, heat_kernel) # Shape: (1, # node, # feature), (1, # node, # node) -> (# node, # class)
            
            loss_test = F.nll_loss(output[self.idx_test], self.y[self.idx_test]) + self.t_loss()
            accuracy_test = compute_accuracy(output[self.idx_test], self.y[self.idx_test])

        print("Prediction Labels >")
        print(output.max(1)[1])
        print("Real Labels >")
        print(self.y)
        
        print(f"Test set results: loss_test: {loss_test.item():.5f} acc_test: {accuracy_test.item():.5f}")
        print(f"Micro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='micro'):.5f}")
        print(f"Macro F1_score: {metrics.f1_score(self.y.detach().cpu().numpy(), output.max(1)[1].detach().cpu().numpy(), average='macro'):.5f}")

        return accuracy_test.detach().cpu()