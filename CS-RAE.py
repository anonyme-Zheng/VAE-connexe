from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class Encoder(nn.Module):
    """Simple MLP encoder producing factorised Gaussian params."""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int] = (400,)):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dims[-1], latent_dim)
        self.logvar_head = nn.Linear(dims[-1], latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class Decoder(nn.Module):
    """MLP decoder for Bernoulli likelihood."""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: Sequence[int] = (400,)):
        super().__init__()
        dims = [latent_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.out_head = nn.Linear(dims[-1], output_dim)

    def forward(self, z: Tensor) -> Tensor:
        h = self.net(z)
        logits = self.out_head(h)
        return logits  # use Bernoulli over each pixel


# -----------------------------------------------------------------------------
# GMM Prior p(z)
# -----------------------------------------------------------------------------

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



# -----------------------------------------------------------------------------
# Cauchy–Schwarz divergence (closed form) between q = N(μ_q, σ_q² I) and GMM prior
# -----------------------------------------------------------------------------

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
    # 注意：根据论文，这里应该是 2σ_k'²
    overlap_pp = []
    for k in range(K):
        for k_prime in range(K):
            overlap = gaussian_overlap(
                mu_p[k:k+1], torch.zeros_like(var_p[k:k+1]),
                mu_p[k_prime:k_prime+1], 2 * var_p[k_prime:k_prime+1]
            )
            overlap_pp.append(overlap)
    term2 = torch.stack(overlap_pp).mean()
    
    # Term 3: ∫ q(z)²dz = N(μ_q | μ_q, 2σ_q²)
    log_term3 = -0.5 * D * math.log(2 * math.pi) - 0.5 * (2 * var_q).log().sum(dim=-1)
    term3 = log_term3.exp()  # (B,)
    
    # D_CS = -log(term1) + 0.5*log(term2) + 0.5*log(term3)
    cs_div = -term1.log() + 0.5 * term2.log() + 0.5 * term3.log()
    
    return cs_div.clamp(min=0)

# -----------------------------------------------------------------------------
# Mixture‑CSRAE model wrapper
# -----------------------------------------------------------------------------
class MixtureCSRAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (400,),
                 latent_dim: int = 20,
                 K: int = 10,
                 lambda_cs: float = 1.0):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1])
        self.prior = GMMPrior(K, latent_dim)
        self.lambda_cs = lambda_cs
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)
        logits = self.decoder(z)
        return logits, mu_q, logvar_q

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, mu_q, logvar_q = self.forward(x)
        # Bernoulli reconstruction loss (binary cross‑entropy)
        recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        # CS divergence
        mu_p, var_p = self.prior()
        cs_loss = cs_divergence_gmm(mu_q, logvar_q.exp(), mu_p, var_p).mean()
        total_loss = recon_loss + self.lambda_cs * cs_loss
        return total_loss, recon_loss, cs_loss
