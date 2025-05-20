"""Minimal example of a Variational Recurrent Auto-Encoder (VRAE).

This module implements a very small VRAE that can be trained on generic
time‑series data.  It intentionally avoids the MIDI specific utilities in
``VRAE.py`` so that it can be used with e.g. financial sequences.

Running the module will train the model on random data if no CSV file is
provided.  A CSV file should contain one sequence per row.  After training, the
script prints the reconstruction of the first sequence.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder module returning ``mu`` and ``logvar`` for a sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 rnn_type: str = "gru") -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.rnn_type = rnn_type.lower()

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

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 rnn_type: str = "gru") -> None:
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.fc_z2h = nn.Linear(latent_dim, hidden_dim)
        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.rnn_type = rnn_type.lower()

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h0 = torch.tanh(self.fc_z2h(z)).unsqueeze(0)
        if self.rnn_type == "lstm":
            h0 = (h0, torch.zeros_like(h0))
        dec_in = torch.zeros(z.size(0), seq_len, self.out.out_features,
                            device=z.device)
        out, _ = self.rnn(dec_in, h0)
        return self.out(out)


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

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.decoder(z, seq_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar

    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        rec = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (rec + kld) / x.size(0)


def train(model: VRAE, data: torch.Tensor, epochs: int = 10, lr: float = 1e-3) -> None:
    """Trains ``model`` on the provided tensor of shape ``[N, T, D]``."""

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        recon, mu, logvar = model(data)
        loss = model.loss(recon, data, mu, logvar)
        optim.zero_grad()
        loss.backward()
        optim.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple VRAE example")
    parser.add_argument("--csv", help="CSV file with sequences (rows)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.csv:
        # Load CSV using Python's builtin csv module to avoid numpy dependency
        import csv

        sequences = []
        with open(args.csv, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                seq = [float(x) for x in row]
                sequences.append(seq)
        data = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
    else:
        # Random data as a stand‑in for financial series
        data = torch.randn(64, 30, 1)

    model = VRAE(input_dim=data.size(-1))
    train(model, data, epochs=args.epochs)

    with torch.no_grad():
        recon, _, _ = model(data[:1])
    print("Reconstruction of the first sequence:\n", recon.squeeze().tolist())


if __name__ == "__main__":
    main()

