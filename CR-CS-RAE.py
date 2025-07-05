# CR-VAE Implementation for Jupyter Notebook
# ===========================================

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
from __future__ import annotations
import math
from typing import Sequence
from torch import nn, Tensor
from torch.nn import functional as F
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

class GMMPrior(nn.Module):
    """Learnable isotropic Gaussian mixture prior with equal weights."""
    def __init__(self, K: int, latent_dim: int):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.mu = nn.Parameter(torch.randn(K, latent_dim) * 0.05)
        self.logvar = nn.Parameter(torch.zeros(K, latent_dim))  # log σ²_k

    @property
    def var(self) -> Tensor:
        return self.logvar.exp()

    def forward(self) -> tuple[Tensor, Tensor]:
        return self.mu, self.var


def gaussian_overlap(mu1: Tensor, var1: Tensor, mu2: Tensor, var2: Tensor) -> Tensor:
    """Computes 𝓝(μ1 | μ2, Σ1 + Σ2) for diagonal covariances (isotropic per dim)."""
    var_sum = var1 + var2
    diff = mu1 - mu2
    
    # 在对数空间计算以提高数值稳定性
    D = mu1.size(-1)
    log_norm = -0.5 * D * math.log(2 * math.pi) - 0.5 * var_sum.log().sum(dim=-1)
    log_exp = -0.5 * (diff.pow(2) / var_sum).sum(dim=-1)
    
    return (log_norm + log_exp).exp()


def cs_divergence_gmm(mu_q: Tensor, var_q: Tensor, mu_p: Tensor, var_p: Tensor) -> Tensor:
    """Compute D_CS(q||p) for Gaussian q vs GMM p with equal weights."""
    K = mu_p.size(0)
    D = mu_q.size(-1)
    
    # Term 1: ∫ q(z)p(z)dz = 1/K ∑_k N(μ_q | μ_k, σ_q² + σ_k²)
    overlap_qp = gaussian_overlap(
        mu_q.unsqueeze(1), var_q.unsqueeze(1), 
        mu_p.unsqueeze(0), var_p.unsqueeze(0)
    )  # (B, K)
    term1 = overlap_qp.mean(dim=1)  # (B,)
    
    # Term 2: ∫ p(z)²dz = 1/K² ∑_{k,k'} N(μ_k | μ_k', 2σ_k'²)
    overlap_pp = gaussian_overlap(
        mu_p.unsqueeze(1), var_p.unsqueeze(1),  # (K, 1, D)
        mu_p.unsqueeze(0), var_p.unsqueeze(0)   # (1, K, D)
    )  # 结果: (K, K)
    term2 = overlap_pp.mean()
    
    # Term 3: ∫ q(z)²dz = N(μ_q | μ_q, 2σ_q²)
    log_term3 = -0.5 * D * math.log(2 * math.pi) - 0.5 * (2 * var_q).log().sum(dim=-1)
    term3 = log_term3.exp()  # (B,)
    
    # D_CS = -log(term1) + 0.5*log(term2) + 0.5*log(term3)
    cs_div = -term1.log() + 0.5 * term2.log() + 0.5 * term3.log()
    
    return cs_div.clamp(min=0)

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
    def __init__(self, num_series, connection, hidden, K, lambda_cs):
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
        
        self.lambda_cs = lambda_cs
        self.prior = GMMPrior(K, hidden)
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
# 假设你已经从 CSRAE 的代码中引入了这些函数
# from your_csrae_utils import gaussian_overlap, cs_divergence_gmm

def train_phase1(crvae, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1, sparsity=100,
                     batch_size=2048, lambda_cs=0.1): # <--- 步骤1: 增加 lambda_cs 参数
    '''
    Train CR-CSRAE model (Phase 1: Structure Learning).
    This version uses Cauchy-Schwarz divergence instead of KL divergence.
    '''
    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []
    batch_size = batch_size

    # Set up data.
    X_list, Y_list = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X_list, dim=0)
    
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    
    # --- 主训练循环 ---
    for it in range(max_iter):
        
        # --- 步骤2: 在循环内部重新计算所有损失 ---
        
        # 随机抽取一个mini-batch
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]
        
        # 前向传播
        pred, mu, log_var = crvae(X_batch)

        # 1. 计算重构损失 (Reconstruction Loss)
        # 注意: 原始代码中 X[:, 10:, i] 的 '10' 是一个硬编码，需要确认其含义。
        # 这里假设它表示从第11个时间点开始预测。
        reconstruction_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) for i in range(p)])

        # 2. 计算CS散度正则项 (CS Divergence Regularization)
        #   a. 准备后验分布 q(z|x) 的参数
        mu_q = mu.squeeze(0)
        var_q = torch.exp(log_var.squeeze(0))
        #   b. 获取先验分布 p(z) 的参数
        mu_p, var_p = crvae.prior()
        #   c. 计算CS散度
        cs_div = cs_divergence_gmm(mu_q, var_q, mu_p, var_p).mean()

        # 3. 计算其他正则项
        ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        
        # 4. 组合成可微分的总损失 (不含L1稀疏项)
        #    这是用于梯度下降的部分
        smooth_loss = reconstruction_loss + ridge + lambda_cs * cs_div

        # 5. 反向传播和梯度更新
        smooth_loss.backward()
        
        # 手动更新参数 (不包括L1惩罚的层)
        # 注意：这里没有使用PyTorch的optimizer，而是手动更新。
        # 确保 crvae.prior 的参数也被更新
        with torch.no_grad():
            for param in crvae.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad

        # 6. 对解码器的输入层权重进行L1稀疏性惩罚 (Proximal Gradient Step)
        if lam > 0:
            for net in crvae.networks:
                prox_update(net, lam, lr)

        # 清空梯度
        crvae.zero_grad()

        # --- 检查点和日志打印 ---
        if (it) % check_every == 0:     
            # 使用整个数据集（或一个固定的验证集）进行评估
            X_t = X_all 
            
            with torch.no_grad(): # 评估时不需要计算梯度
                pred_t, mu_t, log_var_t = crvae(X_t)
            
                # 计算评估损失
                loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
                ridge_t = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
                
                # 计算评估CS散度
                mu_q_t = mu_t.squeeze(0)
                var_q_t = torch.exp(log_var_t.squeeze(0))
                mu_p_t, var_p_t = crvae.prior()
                cs_div_t = cs_divergence_gmm(mu_q_t, var_q_t, mu_p_t, var_p_t).mean()

                # 用于比较和早停的总损失
                mean_loss = (loss_t + ridge_t + lambda_cs * cs_div_t) / p
            
            # --- 步骤3: 更新日志打印 ---
            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it))
                print('Mean Loss = %f' % mean_loss.item())
                print('Recon Loss = %f' % (loss_t.item() / p))
                print('CS_Div = %f' % cs_div_t.item()) # <--- 修改了打印内容
                
                if lam > 0:
                  gc_matrix = crvae.GC()
                  print('Variable usage = %.2f%%'
                        % (100 * torch.mean(gc_matrix.float())))

            # 早停逻辑
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(crvae)
                print(f"*** New best model at iter {best_it} with loss {best_loss:.4f} ***")

    # 训练结束后，恢复最佳模型
    if best_model is not None:
        restore_parameters(crvae, best_model)
        print(f"Finished training. Restored best model from iteration {best_it}.")
    else:
        print("Finished training. No best model found (loss did not improve).")


    return train_loss_list



# 同样，假设你已经引入了 CS 散度计算函数
# from your_csrae_utils import gaussian_overlap, cs_divergence_gmm

def train_phase2(crvae, vrae, X, context, lr, max_iter,lam=0, lam_ridge=0,
                     lookback=5, check_every=50, verbose=1,
                     batch_size=1024, lambda_cs=0.1, lambda_e=1.0): # <--- 步骤1: 增加超参数
    '''
    Train CR-CSRAE model (Phase 2: Fine-tuning for Generation).
    This version uses a fixed causal structure (from Phase 1) and
    optimizes for generation quality using CS divergence.
    It also trains the error-compensation network (vrae).
    '''
    # --- 初始化 ---
    # Phase 2 不再需要 L1 稀疏性惩罚，所以 lam=0
    lam = 0 
    
    # 为 vrae 设置优化器
    optimizer_vrae = optim.Adam(vrae.parameters(), lr=1e-3)

    p = X.shape[-1]
    device = crvae.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.MSELoss()
    train_loss_list = []

    # 准备数据
    X_list, _ = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X_list, dim=0)
    
    # 早停相关变量
    best_it = None
    best_loss = np.inf
    best_crvae_model = None
    best_vrae_model = None
    
    # --- 主训练循环 ---
    for it in range(max_iter):
        
        # --- 步骤2: 在循环内计算损失和梯度 ---
        
        # 随机抽取一个 mini-batch
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]
        
        # --- (A) 训练 CR-CSRAE 主模型 ---
        
        # 前向传播
        pred, mu, log_var = crvae(X_batch)

        # 1. 重构损失
        reconstruction_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) for i in range(p)])

        # 2. CS散度正则项
        mu_q = mu.squeeze(0)
        var_q = torch.exp(log_var.squeeze(0))
        mu_p, var_p = crvae.prior()
        cs_div = cs_divergence_gmm(mu_q, var_q, mu_p, var_p).mean()

        # 3. 其他正则项
        ridge = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
        
        # 4. CR-CSRAE 的总损失
        smooth_loss_crvae = reconstruction_loss + ridge + lambda_cs * cs_div
        
        # 5. CR-CSRAE 的反向传播和更新
        smooth_loss_crvae.backward()
        with torch.no_grad():
            # 手动更新，因为 L1 结构已固定，不再需要 prox_update
            for param in crvae.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
        crvae.zero_grad()

        # --- (B) 训练误差补偿网络 VRAE ---
        
        # 1. 计算重构误差 (作为VRAE的输入)
        #    使用 .detach() 来阻止梯度从 VRAE 流回 CR-CSRAE
        error = (-torch.stack(pred)[:, :, :, 0].permute(1, 2, 0) + X_batch[:, 10:, :]).detach()
        
        # 2. VRAE 前向传播
        pred_e, mu_e, log_var_e = vrae(error)
        
        # 3. VRAE 的重构损失
        loss_e = loss_fn(pred_e, error)
        
        # 4. VRAE 的 KL 散度正则项 (VRAE本身是标准的VAE, 用KL散度)
        kl_div_e = -0.5 * torch.mean(1 + log_var_e - mu_e.pow(2) - log_var_e.exp())
        
        # 5. VRAE 的总损失
        total_loss_vrae = loss_e + lambda_e * kl_div_e
        
        # 6. VRAE 的反向传播和更新 (使用 Adam 优化器)
        total_loss_vrae.backward()
        optimizer_vrae.step()
        optimizer_vrae.zero_grad()

        # --- 检查点和日志打印 ---
        if (it) % check_every == 0:
            # 使用整个数据集进行评估
            X_t = X_all
            
            with torch.no_grad():
                # 评估 CR-CSRAE
                pred_t, mu_t, log_var_t = crvae(X_t)
                loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
                ridge_t = sum([ridge_regularize(net, lam_ridge) for net in crvae.networks])
                mu_q_t, var_q_t = mu_t.squeeze(0), torch.exp(log_var_t.squeeze(0))
                mu_p_t, var_p_t = crvae.prior()
                cs_div_t = cs_divergence_gmm(mu_q_t, var_q_t, mu_p_t, var_p_t).mean()
                mean_loss = (loss_t + ridge_t + lambda_cs * cs_div_t) / p
                
                # 评估 VRAE
                error_t = (-torch.stack(pred_t)[:, :, :, 0].permute(1, 2, 0) + X_t[:, 10:, :]).detach()
                pred_e_t, mu_e_t, log_var_e_t = vrae(error_t)
                loss_e_t = loss_fn(pred_e_t, error_t)
                kl_div_e_t = -0.5 * torch.mean(1 + log_var_e_t - mu_e_t.pow(2) - log_var_e_t.exp())
                total_loss_vrae_t = loss_e_t + lambda_e * kl_div_e_t
                
            # --- 步骤3: 更新日志打印 ---
            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it))
                print('[CR-CSRAE] Mean Loss = %.4f | Recon = %.4f | CS_Div = %.4f'
                      % (mean_loss.item(), (loss_t.item() / p), cs_div_t.item()))
                print('[Error VAE] Total Loss = %.4f | Recon = %.4f | KL_Div = %.4f'
                      % (total_loss_vrae_t.item(), loss_e_t.item(), kl_div_e_t.item()))

            # 早停逻辑 (基于主模型的性能)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_crvae_model = deepcopy(crvae)
                best_vrae_model = deepcopy(vrae)
                print(f"*** New best models at iter {best_it} with main loss {best_loss:.4f} ***")

            # 可视化生成样本 (可选)
            if it % 1000 == 0 and it > 0:
                with torch.no_grad():
                    # 生成不带误差补偿的样本
                    # predicted_data_no_err = crvae(X_t, mode='test', phase=0)
                    # 生成带误差补偿的样本
                    predicted_error = vrae(error_t, mode='test')
                    predicted_data = crvae(X_t, predicted_error, mode='test', phase=1)
                    
                    syn = predicted_data[:, :-1, :].cpu().numpy()
                    ori = X_t.cpu().numpy()
                    
                    # 选择一个样本进行绘图
                    sample_idx = 0
                    plt.figure(figsize=(12, 4))
                    plt.title(f"Generated vs Original Data at Iter {it}")
                    plt.plot(ori[sample_idx, :, 0], label='Original', color='blue', alpha=0.7)
                    plt.plot(syn[sample_idx, :, 0], label='Generated', color='red', linestyle='--')
                    plt.legend()
                    plt.show()

    # 训练结束后恢复最佳模型
    if best_crvae_model is not None and best_vrae_model is not None:
        restore_parameters(crvae, best_crvae_model)
        restore_parameters(vrae, best_vrae_model)
        print(f"Finished training. Restored best models from iteration {best_it}.")
    else:
        print("Finished training. No best model found.")

    return train_loss_list

# Cell 7: Main Demo Code

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
    seed=0       # 随机种子，可根据需要修改或省略
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

full_connect = np.ones(GC_true.shape)
cgru = CRVAE(X.shape[-1], full_connect, hidden=64,K=10,lambda_cs=0.1).to(device)
vrae = VRAE4E(X.shape[-1], hidden=64).to(device)

print("Models created successfully")
print(f"Data shape: {X.shape}")
print(f"Device: {device}")

# Cell 8: Phase 1 Training

print("Starting Phase 1 Training...")
train_loss_list = train_phase1(
    cgru, X, context=20, lam=0.5, lam_ridge=0, lr=5e-2, max_iter=2000,
    check_every=50)

# Extract and evaluate Granger causality
GC_est = cgru.GC().cpu().data.numpy()

print('True variable usage = %.2f%%' % (100 * np.mean(GC_true)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC_true == GC_est)))

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(GC_true, cmap='Blues')
axarr[0].set_title('Causal-effect matrix')
axarr[0].set_ylabel('Effect series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_ylabel('Effect series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC_true[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.show()

# Save the learned structure for phase 2
np.save('GC_henon.npy', GC_est)
full_connect = GC_est  # Use learned structure

# Cell 9: Phase 2 Training

print("Starting Phase 2 Training...")
# Recreate models with learned structure
cgru = CRVAE(X.shape[-1], full_connect, hidden=64,K=10,lambda_cs=0.1).to(device)
vrae = VRAE4E(X.shape[-1], hidden=64).to(device)

train_loss_list = train_phase2(
    cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
    check_every=50)

print("Training completed!")
