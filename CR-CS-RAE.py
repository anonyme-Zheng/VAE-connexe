import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Sequence, Tuple
from torch import Tensor

# =====================================================
# GMM Prior for CS-CRVAE
# =====================================================

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

def prox_update(network, lam, lr):
    '''Perform in place proximal update on first layer weight matrix.'''
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.gru.flatten_parameters()

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

    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.mu, self.var

# =====================================================
# Cauchy-Schwarz Divergence Functions
# =====================================================
def gaussian_overlap(mu1: Tensor, var1: Tensor, mu2: Tensor, var2: Tensor) -> Tensor:
    """Computes N(μ1 | μ2, Σ1 + Σ2) for diagonal covariances."""
    var_sum = var1 + var2
    diff = mu1 - mu2
    
    # Compute in log space for numerical stability
    D = mu1.size(-1)
    log_norm = -0.5 * D * math.log(2 * math.pi) - 0.5 * var_sum.log().sum(dim=-1)
    log_exp = -0.5 * (diff.pow(2) / var_sum).sum(dim=-1)
    
    return (log_norm + log_exp).exp()

def cs_divergence_gmm(mu_q: Tensor, var_q: Tensor, mu_p: Tensor, var_p: Tensor) -> Tensor:
    """Compute D_CS(q||p) for Gaussian q vs GMM p with equal weights."""
    K = mu_p.size(0)
    D = mu_q.size(-1)
    
    # Term 1: ∫ q(z)p(z)dz
    overlap_qp = gaussian_overlap(
        mu_q.unsqueeze(1), var_q.unsqueeze(1), 
        mu_p.unsqueeze(0), var_p.unsqueeze(0)
    )  # (B, K)
    term1 = overlap_qp.mean(dim=1)  # (B,)
    
    # Term 2: ∫ p(z)²dz
    overlap_pp = []
    for k in range(K):
        for k_prime in range(K):
            overlap = gaussian_overlap(
                mu_p[k:k+1], torch.zeros_like(var_p[k:k+1]),
                mu_p[k_prime:k_prime+1], 2 * var_p[k_prime:k_prime+1]
            )
            overlap_pp.append(overlap)
    term2 = torch.stack(overlap_pp).mean()
    
    # Term 3: ∫ q(z)²dz
    log_term3 = -0.5 * D * math.log(2 * math.pi) - 0.5 * (2 * var_q).log().sum(dim=-1)
    term3 = log_term3.exp()  # (B,)
    
    # D_CS = -log(term1) + 0.5*log(term2) + 0.5*log(term3)
    cs_div = -term1.log() + 0.5 * term2.log() + 0.5 * term3.log()
    
    return cs_div.clamp(min=0)

# =====================================================
# CS-CRVAE Components
# =====================================================
class CS_GRU(nn.Module):
    """GRU module for each head in the multi-head decoder."""
    def __init__(self, num_series, hidden):
        super().__init__()
        self.p = num_series
        self.hidden = hidden
        
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)

    def init_hidden(self, batch):
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               
    def forward(self, X, z, connection, mode='train'):
        X = X[:, :, np.where(connection != 0)[0]]
        if mode == 'train':
            X_right, hidden_out = self.gru(torch.cat((X[:, 0:1, :], X[:, 11:-1, :]), 1), z)
            X_right = self.linear(X_right)
            return X_right, hidden_out

class CS_VRAE_Error(nn.Module):
    """Error compensation VAE with CS divergence."""
    def __init__(self, num_series, hidden, K=10, lambda_cs=1.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = num_series
        self.hidden = hidden
        self.lambda_cs = lambda_cs
        
        # Encoder
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        
        # GMM Prior
        self.prior = GMMPrior(K, hidden)
        
        # Decoder
        self.linear_hidden = nn.Linear(hidden, hidden)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, num_series)

    def forward(self, X, mode='train'):
        X = torch.cat((torch.zeros(X.shape, device=self.device)[:, 0:1, :], X), 1)
        
        if mode == 'train':
            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:, 1:, :], hidden_0.detach())
            
            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)
            
            sigma = torch.exp(0.5 * log_var)
            z = torch.randn(size=mu.size()).to(self.device)
            z = mu + sigma * z
            z = self.tanh(self.linear_hidden(z))
            
            X_right, hidden_out = self.gru(X[:, :-1, :], z)
            pred = self.linear(X_right)
            
            return pred, mu.squeeze(0), log_var.squeeze(0)
            
        elif mode == 'test':
            X_seq = torch.zeros(X[:, :1, :].shape).to(self.device)
            h_t = torch.randn(size=(1, X_seq[:, -2:-1, :].size(0), self.hidden)).to(self.device)
            for i in range(20):
                out, h_t = self.gru(X_seq[:, -1:, :], h_t)
                out = self.linear(out)
                X_seq = torch.cat([X_seq, out], dim=1)
            return X_seq

class CS_CRVAE(nn.Module):
    """Cauchy-Schwarz Causal Recurrent VAE."""
    def __init__(self, num_series, connection, hidden, K=10, lambda_cs=1.0):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = num_series
        self.hidden = hidden
        self.K = K
        self.lambda_cs = lambda_cs
        
        # Encoder
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True)
        self.gru_left.flatten_parameters()
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        
        # GMM Prior
        self.prior = GMMPrior(K, hidden)
        
        # Multi-head decoder
        self.connection = connection
        self.networks = nn.ModuleList([
            CS_GRU(int(connection[:, i].sum()), hidden) for i in range(num_series)
        ])

    def forward(self, X, noise=None, mode='train', phase=0):
        X = torch.cat((torch.zeros(X.shape, device=self.device)[:, 0:1, :], X), 1)
        
        if mode == 'train':
            # Encode
            hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
            out, h_t = self.gru_left(X[:, 1:11, :], hidden_0.detach())
            
            mu = self.fc_mu(h_t)
            log_var = self.fc_std(h_t)
            
            # Reparameterization
            sigma = torch.exp(0.5 * log_var)
            z = torch.randn(size=mu.size()).to(self.device)
            z = mu + sigma * z
            
            # Multi-head decode
            pred = [self.networks[i](X, z, self.connection[:, i])[0] 
                    for i in range(self.p)]
            
            return pred, mu.squeeze(0), log_var.squeeze(0)
            
        elif mode == 'test':
            # Generate new sequences
            X_seq = torch.zeros(X[:, :1, :].shape).to(self.device)
            h_0 = torch.randn(size=(1, X_seq[:, -2:-1, :].size(0), self.hidden)).to(self.device)
            ht_last = [h_0 for _ in range(self.p)]
            
            for i in range(20):
                ht_new = []
                for j in range(self.p):
                    out, h_t = self.networks[j](X_seq[:, -1:, :], ht_last[j], self.connection[:, j])
                    if j == 0:
                        X_t = out
                    else:
                        X_t = torch.cat((X_t, out), -1)
                    ht_new.append(h_t)
                ht_last = ht_new
                
                if phase == 1 and noise is not None:
                    X_t = X_t + 0.1 * noise[:, i:i+1, :]
                    
                if i == 0:
                    X_seq = X_t
                else:
                    X_seq = torch.cat([X_seq, X_t], dim=1)
                    
            return X_seq

    def GC(self, threshold=True):
        """Extract learned Granger causality."""
        GC = [torch.norm(net.gru.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (torch.abs(GC) > 0).int()
        else:
            return GC

# =====================================================
# Training Functions for CS-CRVAE
# =====================================================
def train_cs_crvae_phase1(model, X, context, lr, max_iter, lam=0, 
                          check_every=50, verbose=1, batch_size=2048):
    """Train CS-CRVAE with Cauchy-Schwarz divergence."""
    p = X.shape[-1]
    device = model.device
    loss_fn = nn.MSELoss()
    
    # Prepare data
    
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    
    best_loss = np.inf
    best_model = None
    
    for it in range(max_iter):
        # Sample batch
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]
        Y_batch = Y_all[idx]
        
        # Forward pass
        pred, mu, log_var = model(X_batch)
        
        # Reconstruction loss
        recon_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) 
                         for i in range(p)])
        
        # CS divergence loss
        mu_p, var_p = model.prior()
        cs_loss = cs_divergence_gmm(mu, log_var.exp(), mu_p, var_p).mean()
        
        # Total loss
        total_loss = recon_loss + model.lambda_cs * cs_loss
        
        # Backward and update
        total_loss.backward()
        for param in model.parameters():
            param.data -= lr * param.grad
            
        # Proximal update for sparsity
        if lam > 0:
            for net in model.networks:
                prox_update(net, lam, lr)
                
        model.zero_grad()
        
        # Check progress
        if it % check_every == 0:
            mean_loss = total_loss.item() / p
            if verbose > 0:
                print(f'Iter = {it}, Loss = {mean_loss:.4f}, CS = {cs_loss.item():.4f}')
                if lam > 0:
                    print(f'Variable usage = {100 * torch.mean(model.GC().float()):.2f}%')
                    
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model = model.state_dict()
    
    # Restore best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def train_cs_crvae_phase2(model, error_model, X, context, lr, max_iter,
                          check_every=50, verbose=1, batch_size=1024):
    """Train CS-CRVAE phase 2 with error compensation."""
    optimizer = optim.Adam(error_model.parameters(), lr=1e-3)
    p = X.shape[-1]
    loss_fn = nn.MSELoss()
    
    # Prepare data
    
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X_all = torch.cat(X, dim=0)
    
    best_loss = np.inf
    best_model = None
    best_error_model = None
    
    for it in range(max_iter):
        # Sample batch
        idx = np.random.randint(len(X_all), size=(batch_size,))
        X_batch = X_all[idx]
        
        # Forward pass for main model
        pred, mu, log_var = model(X_batch)
        
        # Reconstruction loss
        recon_loss = sum([loss_fn(pred[i][:, :, 0], X_batch[:, 10:, i]) 
                         for i in range(p)])
        
        # CS divergence loss
        mu_p, var_p = model.prior()
        cs_loss = cs_divergence_gmm(mu, log_var.exp(), mu_p, var_p).mean()
        
        # Total loss for main model
        total_loss = recon_loss + model.lambda_cs * cs_loss
        
        # Error compensation
        error = (-torch.stack(pred)[:, :, :, 0].permute(1, 2, 0) + 
                 X_batch[:, 10:, :]).detach()
        pred_e, mu_e, log_var_e = error_model(error)
        
        # Error model losses
        loss_e = loss_fn(pred_e, error)
        mu_p_e, var_p_e = error_model.prior()
        cs_loss_e = cs_divergence_gmm(mu_e, log_var_e.exp(), mu_p_e, var_p_e).mean()
        total_loss_e = loss_e + error_model.lambda_cs * cs_loss_e
        
        # Backward for error model
        total_loss_e.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Backward for main model
        total_loss.backward()
        for param in model.parameters():
            param.data -= lr * param.grad
        model.zero_grad()
        
        # Check progress
        if it % check_every == 0:
            mean_loss = total_loss.item() / p
            if verbose > 0:
                print(f'Iter = {it}, Loss = {mean_loss:.4f}, CS = {cs_loss.item():.4f}')
                print(f'Error Loss = {loss_e.item():.4f}, Error CS = {cs_loss_e.item():.4f}')
                
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_model = model.state_dict()
                best_error_model = error_model.state_dict()
    
    # Restore best models
    if best_model is not None:
        model.load_state_dict(best_model)
    if best_error_model is not None:
        error_model.load_state_dict(best_error_model)
    
    return model, error_model

# =====================================================
# Example Usage
# =====================================================
if __name__ == "__main__":
    # Generate demo data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example: 5-dimensional time series
    dim = 5
    X_np = np.random.randn(1, 1000, dim)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    
    # Initialize causal structure
    full_connect = np.ones((dim, dim))
    
    # Create CS-CRVAE models
    cs_crvae = CS_CRVAE(
        num_series=dim,
        connection=full_connect,
        hidden=64,
        K=10,  # Number of mixture components
        lambda_cs=1.0  # CS divergence weight
    ).to(device)
    
    cs_error_vae = CS_VRAE_Error(
        num_series=dim,
        hidden=64,
        K=10,
        lambda_cs=1.0
    ).to(device)
    
    print("CS-CRVAE initialized successfully!")
    print(f"Number of mixture components: {cs_crvae.K}")
    print(f"Lambda CS: {cs_crvae.lambda_cs}")



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from typing import Sequence, Tuple
from torch import Tensor

# Import necessary utility functions from the original CRVAE
def arrange_input(data, context):
    '''Arrange a single time series into overlapping short sequences.'''
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

def MinMaxScaler(data):
    """Min-Max Normalizer."""
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    return norm_data

def visualization(ori_data, generated_data, analysis, name='tsne'):
    """Using PCA or tSNE for generated and original data visualization."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

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
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1],
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        ax.legend()
        plt.title('PCA plot')
        plt.tick_params(labelsize=12)
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
                    c = colors[:anal_sample_no], alpha = 0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1],
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        ax.legend()
        plt.title('t-SNE plot')
        plt.tick_params(labelsize=12)
        plt.show()

# [Insert all CS-CRVAE class definitions here - from previous response]
# Including: GMMPrior, gaussian_overlap, cs_divergence_gmm, CS_GRU, CS_VRAE_Error, CS_CRVAE

# =====================================================
# Main Test Script for Henon Dataset
# =====================================================

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cell 7: Load Henon data
def generate_henon_data(n_samples=1000, n_dim=2):
    """Generate Henon map data for demo purposes"""
    np.random.seed(42)
    data = np.zeros((n_samples, n_dim))

    # Initial conditions
    x, y = 0.1, 0.1

    for i in range(n_samples):
        x_new = 1 - 1.4 * x**2 + y
        y_new = 0.3 * x
        data[i] = [x_new, y_new]
        x, y = x_new, y_new

    return data

# Try to load henon.npy
try:
    X_np = np.load('henon.npy').T
    print("Loaded henon.npy successfully")
except FileNotFoundError:
    print("henon.npy not found, generating demo data...")
    X_np = generate_henon_data(1000, 2)
    print(f"Generated demo data with shape: {X_np.shape}")

dim = X_np.shape[-1]
GC = np.zeros([dim, dim])
for i in range(dim):
    GC[i, i] = 1
    if i != 0:
        GC[i, i-1] = 1

X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
full_connect = np.ones(GC.shape)

# Create CS-CRVAE models
cs_cgru = CS_CRVAE(
    num_series=X.shape[-1],
    connection=full_connect,
    hidden=64,
    K=10,  # Number of mixture components
    lambda_cs=1.0  # CS divergence weight
).to(device)

cs_vrae = CS_VRAE_Error(
    num_series=X.shape[-1],
    hidden=64,
    K=10,
    lambda_cs=1.0
).to(device)

print("CS-CRVAE models created successfully")
print(f"Data shape: {X.shape}")
print(f"Number of mixture components: {cs_cgru.K}")
print(f"Lambda CS: {cs_cgru.lambda_cs}")

# Cell 8: Phase 1 Training
print("\n" + "="*50)
print("Starting Phase 1 Training with CS-CRVAE...")
print("="*50)

cs_cgru = train_cs_crvae_phase1(
    cs_cgru, X, context=20, lam=0.1, lr=5e-2, max_iter=1000,
    check_every=50, verbose=1, batch_size=2048
)

# Extract and evaluate Granger causality
GC_est = cs_cgru.GC().cpu().data.numpy()

print('\nCausal Discovery Results:')
print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(GC, cmap='Blues')
axarr[0].set_title('True Causal-effect matrix')
axarr[0].set_ylabel('Effect series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_title('CS-CRVAE Estimated matrix')
axarr[1].set_ylabel('Effect series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.tight_layout()
plt.show()

# Save the learned structure for phase 2
np.save('GC_henon_cs.npy', GC_est)
full_connect = GC_est  # Use learned structure

# Cell 9: Phase 2 Training
print("\n" + "="*50)
print("Starting Phase 2 Training with CS-CRVAE...")
print("="*50)

# Recreate models with learned structure
cs_cgru = CS_CRVAE(
    num_series=X.shape[-1],
    connection=full_connect,
    hidden=64,
    K=10,
    lambda_cs=1.0
).to(device)

cs_vrae = CS_VRAE_Error(
    num_series=X.shape[-1],
    hidden=64,
    K=10,
    lambda_cs=1.0
).to(device)

cs_cgru, cs_vrae = train_cs_crvae_phase2(
    cs_cgru, cs_vrae, X, context=20, lr=5e-2, max_iter=10000,
    check_every=50, verbose=1, batch_size=1024
)

print("\nTraining completed!")

# Generate synthetic data for visualization
print("\n" + "="*50)
print("Generating synthetic data...")
print("="*50)

# Generate synthetic sequences
with torch.no_grad():
    # Use a batch of initial sequences
    n_gen = 50
    X_init = X[:, :20, :].repeat(n_gen, 1, 1)

    # Generate with main model
    synthetic_data = cs_cgru(X_init, mode='test')

    # Generate with error compensation
    error_pred = cs_vrae(torch.zeros_like(synthetic_data), mode='test')
    synthetic_data_with_error = cs_cgru(X_init, noise=error_pred, mode='test', phase=1)

# Convert to numpy and normalize
syn_data = synthetic_data_with_error[:, :-1, :].cpu().numpy()
ori_data = X[:, :syn_data.shape[1], :].repeat(n_gen, 1, 1).cpu().numpy()

syn_data = MinMaxScaler(syn_data)
ori_data = MinMaxScaler(ori_data)

# Visualize results
print("\nVisualizing results...")

# Plot sample trajectories
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot first dimension
axes[0, 0].plot(ori_data[0, :100, 0], label='Original', alpha=0.7)
axes[0, 0].plot(syn_data[0, :100, 0], label='CS-CRVAE', alpha=0.7)
axes[0, 0].set_title('Dimension 1 - Time Series')
axes[0, 0].legend()

# Plot second dimension
axes[0, 1].plot(ori_data[0, :100, 1], label='Original', alpha=0.7)
axes[0, 1].plot(syn_data[0, :100, 1], label='CS-CRVAE', alpha=0.7)
axes[0, 1].set_title('Dimension 2 - Time Series')
axes[0, 1].legend()

# Phase space plot
axes[1, 0].scatter(ori_data[0, :, 0], ori_data[0, :, 1], alpha=0.5, label='Original', s=10)
axes[1, 0].scatter(syn_data[0, :, 0], syn_data[0, :, 1], alpha=0.5, label='CS-CRVAE', s=10)
axes[1, 0].set_title('Phase Space')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()

# GMM prior visualization
with torch.no_grad():
    mu_p, var_p = cs_cgru.prior()
    mu_p = mu_p.cpu().numpy()
    var_p = var_p.cpu().numpy()

axes[1, 1].scatter(mu_p[:, 0], mu_p[:, 1], s=100, c='red', marker='x', label='GMM centers')
for k in range(cs_cgru.K):
    circle = plt.Circle((mu_p[k, 0], mu_p[k, 1]), np.sqrt(var_p[k, 0]),
                       fill=False, edgecolor='red', alpha=0.3)
    axes[1, 1].add_patch(circle)
axes[1, 1].set_title('GMM Prior Components (first 2 dims)')
axes[1, 1].set_xlabel('Dimension 1')
axes[1, 1].set_ylabel('Dimension 2')
axes[1, 1].legend()
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.show()

# t-SNE and PCA visualization
visualization(ori_data, syn_data, 'pca', 'cs_crvae_pca.png')
visualization(ori_data, syn_data, 'tsne', 'cs_crvae_tsne.png')

# Save results
np.save('ori_henon_cs.npy', ori_data)
np.save('syn_henon_cs.npy', syn_data)

print("\nCS-CRVAE testing on Henon dataset completed!")
