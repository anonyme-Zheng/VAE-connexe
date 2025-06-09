import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):  # x: [B, T, D]
        _, h = self.rnn(x)
        h = h.squeeze(0)  # [B, hidden_dim]
        mu = self.fc_mu(h)
        logsig = self.fc_logsigma(h)
        return mu, logsig

class DecoderHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        # input mask for causal adjacency will be applied externally

    def forward(self, x, h0):  # x: [B, T, k] k = num parents
        out, _ = self.rnn(x, h0)
        return self.fc_out(out)  # [B, T, 1]

class CRVAE(nn.Module):
    def __init__(self, D, hidden_dim, z_dim, tau, adj_init):
        super().__init__()
        self.D = D
        self.tau = tau
        self.encoder = Encoder(D, hidden_dim, z_dim)
        # one head per variable
        self.heads = nn.ModuleList([DecoderHead(1 + D, hidden_dim) for _ in range(D)])
        # reparam weight
        self.re_fc = nn.Linear(z_dim, hidden_dim)
        # causal adjacency mask (learnable)
        self.A = nn.Parameter(adj_init.clone())

    def reparameterize(self, mu, logsig):
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_early, x_recent):
        # x_early: [B, tau, D], x_recent: [B, tau, D]
        mu, logsig = self.encoder(x_early)
        z = self.reparameterize(mu, logsig)  # [B, z_dim]
        # init decoder hidden from z
        h0 = torch.tanh(self.re_fc(z)).unsqueeze(0)  # [1, B, hidden]
        recon = []
        # apply each head
        for p, head in enumerate(self.heads):
            # mask inputs by A[p]
            mask = (self.A[p] > 0).float()  # [D]
            inp = x_recent * mask.unsqueeze(0).unsqueeze(1)
            # also include self past value
            inp_p = torch.cat([inp[..., p:p+1], inp], dim=-1)
            out_p = head(inp_p, h0)  # [B, tau, 1]
            recon.append(out_p)
        recon = torch.cat(recon, dim=-1)  # [B, tau, D]
        return recon, mu, logsig

# Training skeleton
model = CRVAE(D=5, hidden_dim=64, z_dim=16, tau=10, adj_init=torch.rand(5,5))
optimizer = torch.optim.Adam([p for p in model.parameters() if p is not model.A], lr=1e-3)
# proximal on A
lr_A = 1e-2
lambda_l1 = 1e-3

for epoch in range(epochs):
    for x in dataloader:
        # split x into early and recent
        x_early = x[:, :tau, :]
        x_recent = x[:, tau:2*tau, :]
        recon, mu, logsig = model(x_early, x_recent)
        # compute reconstruction loss
        loss_recon = F.mse_loss(recon, x_recent)
        # KL
        kl = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - torch.exp(2*logsig)) / x.size(0)
        # total
        loss = loss_recon + kl + lambda_l1 * torch.sum(torch.abs(model.A))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # proximal step on A
        with torch.no_grad():
            model.A.data = torch.sign(model.A) * torch.clamp(torch.abs(model.A) - lr_A*lambda_l1, min=0.0)

# After Phase I, freeze zeros in A and continue Phase II without l1

# Inference: sample z and autoregress
z = torch.randn(1, z_dim)
h0 = torch.tanh(model.re_fc(z)).unsqueeze(0)
x_gen = torch.zeros(1, tau, D)
# feed initial recent window
for t in range(T):
    inp = x_gen[:, -tau:, :]
    recon, _, _ = model(x_early=None, x_recent=inp)  # skip encoder, use z only
    x_gen = torch.cat([x_gen, recon[:, -1:, :]], dim=1)
