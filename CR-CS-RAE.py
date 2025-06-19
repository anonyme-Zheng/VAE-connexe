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
