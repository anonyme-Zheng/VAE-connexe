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
