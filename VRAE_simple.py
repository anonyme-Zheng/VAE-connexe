from __future__ import annotations

import argparse
from typing import Tuple, Optional

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

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, rnn_type: str = "rnn", output_activation: str = "sigmoid") -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.output_activation = output_activation
        self.fc_z2h = nn.Linear(latent_dim, hidden_dim)

        if self.rnn_type == "gru":
            self.cell = nn.GRUCell(input_dim, hidden_dim)
        elif self.rnn_type == "lstm":
            self.cell = nn.LSTMCell(input_dim, hidden_dim)
        else:
            self.cell = nn.RNNCell(input_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "tanh":
            return torch.tanh(x)
        elif self.output_activation == "relu":
            return torch.relu(x)
        else:  
            return x
            
    def forward(self, z: torch.Tensor, seq_len: int, target: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:

        batch = z.size(0)
        input_dim = self.fc_out.out_features
        # 1) 从 z 初始化 h (和 c)
        h = torch.tanh(self.fc_z2h(z))            # (batch, hidden_dim)
        if self.rnn_type == "lstm":
            c = torch.zeros_like(h)               # LSTM 还需要 cell-state

        # 2) 自回归地生成 seq_len 步
        outputs= []

        if target is not None and teacher_forcing_ratio > 0:
            x_in = target[:, 0, :]  # Use first timestep from target
        else:
            x_in = torch.zeros(batch, input_dim, device=z.device)
            
        for t in range(seq_len):
            if self.rnn_type == "lstm":
                h, c = self.cell(x_in, (h, c))
            else:
                h = self.cell(x_in, h)

            x_t = self._apply_activation(self.fc_out(h))  # 修改：使用可配置激活函数替代固定的sigmoid
            outputs.append(x_t)

            if t < seq_len - 1:  # Not the last timestep
                use_teacher_forcing = (target is not None and 
                                     torch.rand(1).item() < teacher_forcing_ratio)  # 新增：随机决定是否使用teacher forcing
                if use_teacher_forcing:
                    x_in = target[:, t + 1, :]  # Use next timestep from target
                else:
                    x_in = x_t  # Use model's own output  # 新增：使用模型自己的输出

        return torch.stack(outputs, dim=1)
        

class VRAE(nn.Module):
    """A very small Variational Recurrent Auto‑Encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 2,
                 rnn_type: str = "gru", output_activation: str = "sigmoid") -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, rnn_type)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, rnn_type, output_activation)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int, target: Optional[torch.Tensor] = None,  
               teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        return self.decoder(z, seq_len, target, teacher_forcing_ratio)

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1), target=x, teacher_forcing_ratio=teacher_forcing_ratio)
        return recon, mu, logvar

    def generate(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """新增：Generate sequences from latent codes without teacher forcing，原来没有纯生成功能."""
        return self.decode(z, seq_len, target=None, teacher_forcing_ratio=0.0)

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
        """新增：Sample random sequences from the prior，原来没有从先验采样的功能."""
        z = torch.randn(batch_size, self.encoder.fc_mu.out_features, device=device)
        return self.generate(z, seq_len)
        
    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        rec_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = rec_loss + beta * kld_loss  
        return total_loss, rec_loss, kld_loss


def train(model: VRAE, data: torch.Tensor, epochs: int = 10, lr: float = 1e-3, beta: float = 1.0, teacher_forcing_schedule: Optional[callable] = None) -> None:
    """Trains ``model`` on the provided tensor of shape ``[N, T, D]``."""

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if teacher_forcing_schedule:
            tf_ratio = teacher_forcing_schedule(epoch)
        else:
            tf_ratio = 1.0
            
        recon, mu, logvar = model(data, teacher_forcing_ratio=tf_ratio)  # 修改：传入teacher_forcing_ratio
        total_loss, rec_loss, kld_loss = model.loss(recon, data, mu, logvar, beta)
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if epoch % 10 == 0:  
            print(f"Epoch {epoch:3d}/{epochs}  |  Total: {total_loss.item():.4f}  |  "
                  f"Rec: {rec_loss.item():.4f}  |  KLD: {kld_loss.item():.4f}  |  "
                  f"TF: {tf_ratio:.2f}")



def exponential_teacher_forcing_schedule(epoch: int, initial_ratio: float = 1.0, 
                                       decay_rate: float = 0.05) -> float:
    """新增：Exponentially decay teacher forcing ratio."""
    return initial_ratio * (1 - decay_rate) ** epoch


def linear_teacher_forcing_schedule(epoch: int, initial_ratio: float = 1.0, 
                                  final_ratio: float = 0.0, total_epochs: int = 100) -> float:
    """新增：Linearly decay teacher forcing ratio."""
    return initial_ratio - (initial_ratio - final_ratio) * (epoch / total_epochs)


if __name__ == "__main__":
    # 新增：Example usage - 原来没有完整的使用示例
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create synthetic data
    batch_size, seq_len, input_dim = 32, 20, 10
    data = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # 修改：Create model - 现在使用新的参数配置
    model = VRAE(input_dim=input_dim, hidden_dim=64, latent_dim=32, 
                 rnn_type="gru", output_activation="tanh").to(device)  # 修改：使用tanh激活而非固定的sigmoid
    
    # 修改：Train with teacher forcing schedule - 现在支持复杂的训练配置
    train(model, data, epochs=100, lr=1e-3, beta=0.5,  # 修改：使用beta-VAE
          teacher_forcing_schedule=lambda epoch: exponential_teacher_forcing_schedule(epoch, decay_rate=0.02))  # 新增：使用teacher forcing调度
    
    # 新增：Generate new samples - 原来没有生成功能演示
    with torch.no_grad():
        samples = model.sample(batch_size=5, seq_len=20, device=device)
        print(f"Generated samples shape: {samples.shape}")
