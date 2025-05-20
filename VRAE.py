"""
PyTorch implementation of a Variational Recurrent Auto‑Encoder (VRAE)
--------------------------------------------------------------------
Key features
~~~~~~~~~~~~
* Encodes a variable‑length (or fixed window) time‑series / piano‑roll into a single latent vector z.
* Supports any RNN core (GRU, LSTM) via a flag.
* Dataloader that converts MIDI files to binary piano‑roll tensors using pretty_midi.
* Training / evaluation loops with KL + reconstruction loss.
* Sampling & interpolation utilities to generate new sequences.
* Optional t‑SNE visualisation of latent vectors.

Usage
~~~~~
$ python vrae_pytorch.py --data_dir ./midi --seq_len 50 --latent_dim 20 \
                         --hidden_dim 500 --epochs 20000 --lr 1e-3

Dependencies
~~~~~~~~~~~~
* torch, torchvision
* pretty_midi (for MIDI parsing)
* scikit‑learn (for t‑SNE)
* matplotlib (for plots)
* tqdm (progress bars)

Tip: install with
  pip install torch pretty_midi scikit-learn matplotlib tqdm
"""

from __future__ import annotations
import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional, only used for visualisation/generation helpers
try:
    import pretty_midi
except ImportError:
    pretty_midi = None

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    TSNE = None
    plt = None

# -----------------------------------------------------------
# Data utilities
# -----------------------------------------------------------

# Keep 49 pitches as described in the paper. MIDI notes range from 21 to 107
# (inclusive) here so we drop the lowest 38 notes and keep the remaining 49.
USED_PITCHES = [p for p in range(21, 108) if p not in range(21, 59)]

class MidiDataset(Dataset):
    """Convert MIDI files to fixed‑length binary piano‑roll windows."""

    def __init__(self, midi_dir: str, seq_len: int = 50, overlap: float = 0.0):
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for MidiDataset, please `pip install pretty_midi`. ")
        self.seq_len = seq_len
        self.samples: List[np.ndarray] = []
        midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi"))
        step = int(seq_len * (1 - overlap)) or 1
        for path in midi_files:
            pm = pretty_midi.PrettyMIDI(path)
            piano_roll = pm.get_piano_roll(fs=20)[USED_PITCHES]  # shape [49, T] 0/127
            piano_roll = (piano_roll > 0).astype(np.float32)     # binarise
            # slide window
            for i in range(0, piano_roll.shape[1] - seq_len + 1, step):
                window = piano_roll[:, i:i + seq_len]            # [49, seq_len]
                self.samples.append(window.T)                    # transpose => [seq_len, 49]
        self.samples = np.stack(self.samples)                    # [N, seq_len, 49]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])

# -----------------------------------------------------------
# VRAE network
# -----------------------------------------------------------

class VRAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 500, latent_dim: int = 20, rnn_type: str = "gru", bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_dir = 2 if bidirectional else 1

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.enc_rnn = rnn_cls(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.fc_mu = nn.Linear(hidden_dim * self.n_dir, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * self.n_dir, latent_dim)

        self.fc_z2h = nn.Linear(latent_dim, hidden_dim * self.n_dir)
        self.dec_rnn = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.out_layer = nn.Linear(hidden_dim, input_dim)
        self.rnn_type = rnn_type.lower()

    def encode(self, x):
        _, h = self.enc_rnn(x)  # h: [n_dir, B, H]
        if self.rnn_type == "lstm":
            h = h[0]            # take hidden state for LSTM
        h = h.transpose(0,1).reshape(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, seq_len):
        h0 = torch.tanh(self.fc_z2h(z))                 # [B, H*n_dir]
        h0 = h0.view(self.n_dir, -1, self.hidden_dim)   # shape for RNN initial state
        if self.rnn_type == "lstm":
            h0 = (h0, torch.zeros_like(h0))             # (h0, c0)
        dec_in = torch.zeros(z.size(0), seq_len, self.input_dim, device=z.device)
        out, _ = self.dec_rnn(dec_in, h0)
        recon = torch.sigmoid(self.out_layer(out))      # piano‑roll 0‑1
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD) / x.size(0)

# -----------------------------------------------------------
# Training utilities
# -----------------------------------------------------------

def train(model: VRAE, loader: DataLoader, epochs: int, lr: float, device: str, log_every: int = 1000):
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.05, 0.001))
    model.to(device)
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = model.loss_function(recon, x, mu, logvar)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * x.size(0)
            if global_step % log_every == 0:
                print(f"Step {global_step}, Loss {loss.item():.4f}")
            global_step += 1
        print(f"Epoch {epoch} | ELBO loss = {epoch_loss / len(loader.dataset):.4f}")

# -----------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------

def sample_sequences(model: VRAE, n_samples: int, seq_len: int, device: str) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        return model.decode(z, seq_len)

# -----------------------------------------------------------
# t‑SNE visualisation
# -----------------------------------------------------------

def plot_tsne(latents: np.ndarray, labels: List[str]):
    if TSNE is None or plt is None:
        print("Install scikit-learn and matplotlib to enable t‑SNE plotting.")
        return
    tsne = TSNE(n_components=2, perplexity=40, init='pca', random_state=0)
    two_d = tsne.fit_transform(latents)
    plt.figure(figsize=(6,6))
    for lab in set(labels):
        idx = [i for i,l in enumerate(labels) if l == lab]
        plt.scatter(two_d[idx,0], two_d[idx,1], label=lab, s=8)
    plt.legend(fontsize=8)
    plt.title("t‑SNE of latent vectors")
    plt.show()

# -----------------------------------------------------------
# Main CLI
# -----------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train a Variational Recurrent Auto‑Encoder on MIDI data")
    p.add_argument('--data_dir', required=True, help='Folder with .mid files')
    p.add_argument('--seq_len', type=int, default=50)
    p.add_argument('--overlap', type=float, default=0.0)
    p.add_argument('--hidden_dim', type=int, default=500)
    p.add_argument('--latent_dim', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--rnn_type', choices=['gru','lstm'], default='gru')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tsne', action='store_true', help='Visualise latent space after training')
    return p.parse_args()


def main():
    args = get_args()
    dataset = MidiDataset(args.data_dir, args.seq_len, args.overlap)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = VRAE(input_dim=len(USED_PITCHES), hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, rnn_type=args.rnn_type)
    train(model, loader, args.epochs, args.lr, args.device)

    # Optionally t‑SNE visualisation
    if args.tsne and TSNE is not None:
        latents, labels = [], []
        model.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(args.device)
                mu, _ = model.encode(x)
                latents.append(mu.cpu().numpy())
                labels.extend(["song"]*x.size(0))
        plot_tsne(np.concatenate(latents), labels)

    # Example generation
    gen = sample_sequences(model, n_samples=8, seq_len=args.seq_len, device=args.device)
    np.save("generated.npy", gen.cpu().numpy())
    print("Saved 8 generated sequences to generated.npy")

if __name__ == '__main__':
    main()

