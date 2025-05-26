from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder module returning ``mu`` and ``logvar`` for a sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, rnn_type: str = "rnn") -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == "gru":
            rnn_cls = nn.GRU
        elif rnn_type.lower() == "lstm":
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.RNN

        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h = self.rnn(x)
        if self.rnn_type == "lstm":
            h = h[0]
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    """Decoder module reconstructing a sequence from ``z``."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, rnn_type: str = "rnn") -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.fc_z2h = nn.Linear(latent_dim, hidden_dim)

        if self.rnn_type == "gru":
            self.cell = nn.GRUCell(input_dim, hidden_dim)
        elif self.rnn_type == "lstm":
            self.cell = nn.LSTMCell(input_dim, hidden_dim)
        else:
            self.cell = nn.RNNCell(input_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, input_dim)


    def forward(self, z: torch.Tensor, seq_len: int, target: torch.Tensor) -> torch.Tensor:

        batch = z.size(0)
        # 1) 从 z 初始化 h (和 c)
        h = torch.tanh(self.fc_z2h(z))            # (batch, hidden_dim)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)               # LSTM 还需要 cell-state

        # 2) 自回归地生成 seq_len 步
        outputs= []

        for t in range(seq_len):

            x_in = target[:, t, :]

            if self.rnn_type == "lstm":
                h, c = self.cell(x_in, (h, c))     # LSTMCell
            else:
                h = self.cell(x_in, h)             # RNNCell / GRUCell

            # 由 h -> xₜ
            x_t = torch.sigmoid(self.fc_out(h))  # 根据数据分布可换激活
            outputs.append(x_t)

        # (seq_len 个 (batch, input_dim)) → (batch, seq_len, input_dim)
        return torch.stack(outputs, dim=1)



class VRAE(nn.Module):
    """A very small Variational Recurrent Auto‑Encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16,
                 rnn_type: str = "gru") -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, rnn_type)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, rnn_type)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int, target: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, seq_len, target)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1), target=x)
        return recon, mu, logvar

    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        rec = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (rec + kld) / x.size(0)


def train(model: VRAE, data: torch.Tensor, epochs: int = 10, lr: float = 1e-3) -> None:
    """Trains ``model`` on the provided tensor of shape ``[N, T, D]``."""

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        recon, mu, logvar = model(data)
        loss = model.loss(recon, data, mu, logvar)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 1 == 0:  # 想隔几轮打一次就改这里
            print(f"Epoch {epoch:3d}/{epochs}  |  loss = {loss.item():.4f}")
