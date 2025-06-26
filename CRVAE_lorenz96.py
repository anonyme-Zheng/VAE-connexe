# Cell 1: Import all required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.io
from scipy.integrate import odeint
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cell 2: Utility functions (missing from utils)
def train_test_divide(ori_data, generated_data, ori_time, generated_time, train_rate=0.8):
    """Divide train and test data for both original and generated data.
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - ori_time: original time
        - generated_time: generated time
        - train_rate: ratio of training data
    
    Returns:
        - train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat
    """
    # Divide train/test index (original data)
    no = len(ori_data)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]
    
    train_x = [ori_data[i] for i in train_idx]
    test_x = [ori_data[i] for i in test_idx]
    train_t = [ori_time[i] for i in train_idx]
    test_t = [ori_time[i] for i in test_idx]
    
    # Divide train/test index (generated data)
    no = len(generated_data)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]
    
    train_x_hat = [generated_data[i] for i in train_idx]
    test_x_hat = [generated_data[i] for i in test_idx]
    train_t_hat = [generated_time[i] for i in train_idx]
    test_t_hat = [generated_time[i] for i in test_idx]
    
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    
    Args:
        - data: original data
        
    Returns:
        - time: extracted time
        - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
        
    return time, max_seq_len

def batch_generator(data, time, batch_size):
    """Mini-batch generator.
    
    Args:
        - data: time-series data
        - time: time information
        - batch_size: the number of samples in each batch
        
    Returns:
        - X_mb: time-series data in each batch
        - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]     
            
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    
    return X_mb, T_mb

# Cell 3: Model Definitions

class GRU(nn.Module):
    def __init__(self, num_series, hidden):
        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        #Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               
    def forward(self, X, z, connection, mode = 'train'):
        X=X[:,:,np.where(connection!=0)[0]]
        device = self.gru.weight_ih_l0.device
        tau = 0
        if mode == 'train':
          X_right, hidden_out = self.gru(torch.cat((X[:,0:1,:],X[:,11:-1,:]),1), z)
          X_right = self.linear(X_right)
          return X_right, hidden_out

class VRAE4E(nn.Module):
    def __init__(self, num_series, hidden):
        '''
        Error VAE
        '''
        super(VRAE4E, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)#nn.Linear(hidden, 1)
        self.fc_std = nn.Linear(hidden, hidden)
        
        self.linear_hidden = nn.Linear(hidden, hidden)
        self.tanh = nn.Tanh()
        
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, num_series)

    def init_hidden(self, batch):
        '''Initialize hidden states for GRU cell.'''
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               
    def forward(self, X, mode = 'train'):
        X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
        if mode == 'train':
            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:,1:,:], hidden_0.detach())
            
            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)
            
            sigma = torch.exp(0.5*log_var)
            z = torch.randn(size = mu.size())
            z = z.type_as(mu) 
            z = mu + sigma*z
            z = self.tanh(self.linear_hidden(z))
            
            X_right, hidden_out = self.gru(X[:,:-1,:], z)
            pred = self.linear(X_right)
            
            return pred, log_var, mu
            
        if mode == 'test':
            X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
            h_t = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
            for i in range(int(20/1)+1):
                out, h_t = self.gru(X_seq[:,-1:,:], h_t)
                out = self.linear(out)
                #out = self.sigmoid(out)
                X_seq = torch.cat([X_seq,out],dim = 1)
            return X_seq

class CRVAE(nn.Module):
    def __init__(self, num_series, connection, hidden):
        '''
        connection: pruned networks
        '''
        super(CRVAE, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = num_series
        self.hidden = hidden
        
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        self.connection = connection

        # Set up networks.
        self.networks = nn.ModuleList([
            GRU(int(connection[:,i].sum()), hidden) for i in range(num_series)])

    def forward(self, X, noise = None, mode = 'train', phase = 0):
        if phase == 0:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu)
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
    
                return pred, log_var, mu
                
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    ht_new = []
                    for j in range(self.p):
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t
                    else:
                        X_seq = torch.cat([X_seq,X_t],dim = 1)
                return X_seq
        
        if phase == 1:
            X = torch.cat((torch.zeros(X.shape,device = self.device)[:,0:1,:],X),1)
            if mode == 'train':
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size())
                z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
                z = mu + sigma*z
    
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
    
                return pred, log_var, mu
                
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape).to(self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden)).to(self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):#int(20/2)+1
                    ht_new = []
                    for j in range(self.p):
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t + 0.1*noise[:,i:i+1,:] 
                    else:
                        X_seq = torch.cat([X_seq,X_t+0.1*noise[:,i:i+1,:]],dim = 1)
                return X_seq

    def GC(self, threshold=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.

        Returns:
          GC: (p x p) matrix. Entry (i, j) indicates whether variable j is
            Granger causal of variable i.
        '''
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0)
              for net in self.networks]
        GC = torch.stack(GC)
        #print(GC)
        if threshold:
            return (torch.abs(GC) > 0).int()
        else:
            return GC

# Cell 4: Training utility functions

def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.gru.flatten_parameters()

def regularize(network, lam):
    '''Calculate regularization term for first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))

def ridge_regularize(network, lam):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
    return lam * (
        torch.sum(network.linear.weight ** 2) +
        torch.sum(network.gru.weight_hh_l0 ** 2))

def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def arrange_input(data, context):
    '''
    Arrange a single time series into overlapping short sequences.

    Args:
      data: time series of shape (T, dim).
      context: length of short sequences.
    '''
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(len(data) - context, context, data.shape[1],
                        dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input.detach(), target.detach()

def MinMaxScaler(data):
  """Min-Max Normalizer.
  
  Args:
    - data: raw data
    
  Returns:
    - norm_data: normalized data
    - min_val: minimum values (for renormalization)
    - max_val: maximum values (for renormalization)
  """    
  min_val = np.min(np.min(data, axis = 0), axis = 0)
  data = data - min_val
    
  max_val = np.max(np.max(data, axis = 0), axis = 0)
  norm_data = data / (max_val + 1e-7)
    
  return norm_data

# Cell 5: Visualization functions

def visualization (ori_data, generated_data, analysis, name = 'tsne'):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2)#, label = "Original"
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2)#, label = "Synthetic"
  
    #ax.legend()  
    #plt.title('PCA plot', fontsize=21)
    # plt.xlabel('x-pca')
    # plt.ylabel('y_pca')
    plt.tick_params(labelsize=21)
    plt.savefig(name)
    plt.show()
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2)#
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2)#, label = "Synthetic"
  
    #ax.legend()
      
    #plt.title('t-SNE plot', fontsize=21)
    # plt.xlabel('x-tsne')
    # plt.ylabel('y_tsne')
    plt.tick_params(labelsize=21)
    plt.savefig(name)
    plt.show()

# Cell 6: Training functions

def train_phase1(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 256):
    '''Train model with Adam.'''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    Y = Y_all[idx]
    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate crvae error.
    pred,mu,log_var = crvae(X)

    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        crvae.zero_grad()

        pred,mu,log_var = crvae(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd

        # Check progress.
        if (it) % check_every == 0:     
            X_t = X
            Y_t = Y
            
            pred_t,mu_t ,log_var_t= crvae(X_t)
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)
                
            start_point = 0
            predicted_data = crvae(X_t,mode = 'test')
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            
            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list

def train_phase2(crvae, vrae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,sparsity = 100, batch_size = 256):
    '''Train model with Adam.'''
    optimizer = optim.Adam(vrae.parameters(), lr=1e-3)
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size
    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    idx = np.random.randint(len(X_all), size=(batch_size,))
    
    X = X_all[idx]
    Y = Y_all[idx]
    X_v = X_all[batch_size:]
    start_point = 0#context-10-1
    beta = 1#0.001
    beta_e = 1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # Calculate smooth error.
    pred,mu,log_var = crvae(X)#
    
    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
    
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    #mmd =  sum([MMD(torch.randn(200, Y[:, :, 0].shape[-1], requires_grad = False).to(device), latent[i][:,:,0]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
    smooth = loss + ridge + beta*mmd
    
    error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
    pred_e,mu_e,log_var_e = vrae(error)
    loss_e = loss_fn(pred_e, error)
    mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
    smooth_e = loss_e + beta_e*mmd_e

    best_mmd = np.inf       
            
########################################################################   
    #lr = 1e-3        
    for it in range(max_iter):
        # Take gradient step.
        smooth_e.backward()
        if lam == 0:
            optimizer.step()
            optimizer.zero_grad()  
        
        smooth.backward()
        for param in crvae.parameters():
            param.data -= lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        crvae.zero_grad()

        # Calculate loss for next iteration.
        idx = np.random.randint(len(X_all), size=(batch_size,))
        
        pred,mu,log_var = crvae(X)#
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
        
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
        smooth = loss + ridge + beta*mmd
        
        error = (-torch.stack(pred)[:, :, :, 0].permute(1,2,0) + X[:, 10:, :]).detach()
        pred_e,mu_e,log_var_e = vrae(error)
        loss_e = loss_fn(pred_e, error)
        mmd_e = (-0.5*(1+log_var_e - mu_e**2- torch.exp(log_var_e)).sum(dim = -1).sum(dim = 0)).mean(dim =0)
        smooth_e = loss_e + beta_e*mmd_e

        # Check progress.
        if (it) % check_every == 0:
            X_t = X
            pred_t,mu_t ,log_var_t= crvae(X_t)
        
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
            
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
        
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in crvae.networks])
            smooth_t = loss_t + ridge_t# + beta*mmd_t
            
            nonsmooth = sum([regularize(net, lam) for net in crvae.networks])
            mean_loss = (smooth_t) / p

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                print('Loss = %f' % mean_loss)
                print('KL = %f' % mmd)

                print('Loss_e = %f' % smooth_e)
                print('KL_e = %f' % mmd_e)
                
                if lam>0:
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(crvae.GC().float())))
            
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)
                
            start_point = 0
            predicted_error = vrae(error, mode = 'test').detach()
            
            predicted_data = crvae(X_t, predicted_error, mode = 'test', phase = 1)
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()

            if it % 1000 ==0:
                plt.plot(ori[0,:,1])
                plt.plot(syn[0,:,1])
                plt.show()

                visualization(ori, syn, 'pca')
                visualization(ori, syn, 'tsne')
                np.save('ori_henon.npy',ori)
                np.save('syn_henon.npy',syn)

    # Restore best model.
    restore_parameters(crvae, best_model)

    return train_loss_list

def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F
    return dxdt

def lorenz_96(d, t, t_eval, f, seed, delta_t=0.1, sd=0.1, burn_in=1000):
    if seed is not None:
        np.random.seed(seed)

    # 1. 初始状态与时间点
    x0 = np.random.normal(scale=0.01, size=d)
    tm = np.linspace(0, (t + t_eval + burn_in) * delta_t, t + t_eval + burn_in)

    # 2. 求解 ODE 并加噪声
    X = odeint(lorenz, x0, tm, args=(f,))
    X += np.random.normal(scale=sd, size=(t + t_eval + burn_in, d))

    X_stable = X[burn_in:]  

    # 4. （可选）沿 time 轴对每个变量做标准化
    mean = X_stable.mean(axis=0, keepdims=True)   # shape (1, d)
    std  = X_stable.std(axis=0, keepdims=True)    # shape (1, d)
    X_stable = (X_stable - mean) / (std + 1e-8)

    # 5. 直接返回 (time, dim)
    return X_stable.T.astype(np.float32)         

# 尝试加载论文版 Lorenz-96 数据，否则生成并保存
fname = '2_x.npy'
try:
    X_np = np.load(fname)
    print(f"Loaded `{fname}` with shape {X_np.shape}")
except FileNotFoundError:
    # … 原来的生成逻辑 …
    X_np = lorenz_96(
    d=10,        # 原来的 p=10
    t=2048,      # 原来的 T=2048
    t_eval=0,    # 不再额外生成测试段
    f=10.0,      # Lorenz-96 方程的常数 F
    seed=0       # 随机种子
)   
    np.save(fname, X_np)
    print(f"Generated and saved `{fname}` with shape {X_np.shape}")

# 如果刚好读到二维 (p, T)，就补一个 batch 维度
if X_np.ndim == 2:
    X_np = X_np[np.newaxis, :, :]   # 变成 (1, p, T)
    print(f"Expanded to 3D with shape {X_np.shape}")

# 转为模型输入：(batch, T, dim)
X = torch.tensor(X_np.transpose(0, 2, 1), dtype=torch.float32, device=device)
print("Data tensor shape:", X.shape)  

# 构造“真”因果矩阵：每个变量 i 的直接影响来自 i, i-1, i-2, i+1
p = 10
GC_true = np.zeros((p, p), dtype=int)
for i in range(p):
    GC_true[i, i]    = 1
    GC_true[i, (i-1)%p] = 1
    GC_true[i, (i-2)%p] = 1
    GC_true[i, (i+1)%p] = 1
print("True GC matrix:\n", GC_true)

# ——以下保持与你原有代码一致——
full_connect = np.ones(GC_true.shape)
cgru = CRVAE(p, full_connect, hidden=64).to(device)
vrae = VRAE4E(p, hidden=64).to(device)

# Phase 1 训练
train_loss_list = train_phase1(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=5000,
    check_every=50)

GC_est = cgru.GC(threshold=True).cpu().numpy()
print("Estimated GC:\n", GC_est)

# 可视化对比
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.imshow(GC_true,  cmap='Blues'); ax1.set_title('True GC')
ax2.imshow(GC_est, cmap='Blues', vmin=0, vmax=1, interpolation='nearest') 
ax2.set_title('Estimated GC')
plt.show()

# Phase 2：保存结构并重训
np.save('GC_lorenz96.npy', GC_est)
full_connect = GC_est
cgru = CRVAE(p, full_connect, hidden=64).to(device)
vrae = VRAE4E(p, hidden=64).to(device)

train_loss_list = train_phase2(
    cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
    check_every=50)

print("Phase 2 completed!")
