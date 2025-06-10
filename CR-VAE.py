import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        # Use GRU as specified in paper
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
    """Single decoder head for one variable"""
    def __init__(self, hidden_dim):
        super().__init__()
        # Each head has its own GRU
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, h0, W_in):
        """
        x: [B, T, D] - all variables (causal selection done via W_in)
        h0: [1, B, hidden_dim] - initial hidden state
        W_in: [D, hidden_dim] - learnable input weight matrix (sparse)
        """
        B, T, D = x.shape
        
        # Apply causal input transformation: x * W_in -> [B, T, hidden_dim]
        x_transformed = torch.matmul(x, W_in)  # [B, T, hidden_dim]
        
        # Apply GRU
        output, final_h = self.gru(x_transformed, h0)  # [B, T, hidden_dim], [1, B, hidden_dim]
        
        # Apply output layer
        output = self.fc_out(output)  # [B, T, 1]
        
        return output, final_h

class ErrorCompensationVAE(nn.Module):
    """Error compensation network for instantaneous effects (epsilon_t)"""
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, z_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.fc_z_to_h = nn.Linear(z_dim, hidden_dim)

    def forward(self, x):
        # Encode error sequence
        _, h_enc = self.encoder(x)
        mu = self.fc_mu(h_enc.squeeze(0))
        logsig = self.fc_logsigma(h_enc.squeeze(0))
        
        # Reparameterize
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode to reconstruct error
        h0 = torch.tanh(self.fc_z_to_h(z)).unsqueeze(0)
        out, _ = self.decoder(x, h0)
        recon = self.fc_out(out)
        
        return recon, mu, logsig

class CRVAE(nn.Module):
    def __init__(self, D, hidden_dim, z_dim, tau=10):
        super().__init__()
        self.D = D  # Number of variables
        self.tau = tau  # Time lag
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Encoder (processes past observations)
        self.encoder = Encoder(D, hidden_dim, z_dim)
        
        # Multi-head decoder - one head per variable
        self.heads = nn.ModuleList([DecoderHead(hidden_dim) for _ in range(D)])
        
        # Reparameterization layer
        self.re_fc = nn.Linear(z_dim, hidden_dim)
        
        # Learnable input weight matrices for each head (U_p_in in paper)
        # These encode the causal relationships A
        self.W_in_list = nn.ParameterList([
            nn.Parameter(torch.randn(D, hidden_dim) * 0.1) for _ in range(D)
        ])
        
        # Error compensation network
        self.error_vae = ErrorCompensationVAE(D, hidden_dim//2, z_dim//2)

    def reparameterize(self, mu, logsig):
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_causal_adjacency_matrix(self, threshold=1e-6):
        """Extract causal adjacency matrix A from learned weights"""
        with torch.no_grad():
            A = torch.zeros(self.D, self.D)
            for p in range(self.D):
                # For each variable p, check which input dimensions have significant weights
                W_p = self.W_in_list[p]  # [D, hidden_dim]
                # Use L2 norm across hidden dimensions to determine causality
                causal_strength = torch.norm(W_p, dim=1)  # [D]
                A[p, :] = (causal_strength > threshold).float()
            return A

    def apply_proximal_operator(self, lambda_reg, lr):
        """Apply proximal operator for L1 sparsity (soft thresholding)"""
        with torch.no_grad():
            for W_in in self.W_in_list:
                W_in.data = torch.sign(W_in) * torch.clamp(
                    torch.abs(W_in) - lr * lambda_reg, min=0.0
                )

    def forward(self, x_past, x_current=None, mode='train', phase=1, seq_len=20):
        """
        x_past: [B, tau, D] - past observations for encoder
        x_current: [B, tau, D] - current observations for decoder target
        mode: 'train' or 'generate'
        phase: 1 (causal discovery) or 2 (generation improvement)
        seq_len: length of sequence to generate (only used in generate mode)
        """
        if mode == 'train':
            return self._forward_train(x_past, x_current, phase)
        else:
            return self._forward_generate(x_past, seq_len)

    def _forward_train(self, x_past, x_current, phase):
        # Encode past observations
        mu, logsig = self.encoder(x_past)
        z = self.reparameterize(mu, logsig)
        
        # Initialize decoder hidden state from latent variable
        h0 = torch.tanh(self.re_fc(z)).unsqueeze(0)  # [1, B, hidden_dim]
        
        # Create input for decoder (shifted by one time step for autoregressive)
        # Use past and beginning of current for prediction
        decoder_input = torch.cat([x_past[:, -1:, :], x_current[:, :-1, :]], dim=1)
        
        # Multi-head decoding
        recon_list = []
        
        for p in range(self.D):
            # Each head uses its learned input transformation
            W_in_p = self.W_in_list[p]  # [D, hidden_dim]
            
            # Decode for p-th variable
            recon_p, _ = self.heads[p](decoder_input, h0, W_in_p)  # [B, tau, 1]
            recon_list.append(recon_p)
        
        recon = torch.cat(recon_list, dim=-1)  # [B, tau, D]
        
        if phase == 2:
            # Phase 2: Add error compensation
            error = x_current - recon.detach()  # Detach to avoid affecting causal learning
            error_recon, error_mu, error_logsig = self.error_vae(error)
            recon = recon + error_recon
            return recon, mu, logsig, error_mu, error_logsig
        
        return recon, mu, logsig

    def _forward_generate(self, x_past, seq_len=20):
        """Generate sequences autoregressively"""
        batch_size = x_past.size(0)
        device = x_past.device
        
        # Encode initial context
        mu, logsig = self.encoder(x_past)
        z = self.reparameterize(mu, logsig)
        h0 = torch.tanh(self.re_fc(z)).unsqueeze(0)
        
        # Initialize generation with last observation from past
        generated = [x_past[:, -1:, :]]  # Start with last observation
        
        # Hidden states for each head
        hidden_states = [h0.clone() for _ in range(self.D)]
        
        for t in range(seq_len):
            # Use last tau observations as input
            if len(generated) >= self.tau:
                recent_input = torch.cat(generated[-self.tau:], dim=1)
            else:
                # Pad with past observations if needed
                padding_needed = self.tau - len(generated)
                padding = x_past[:, -padding_needed:, :]
                recent_input = torch.cat([padding] + generated, dim=1)
            
            # Generate next time step for each variable
            next_step = []
            new_hidden_states = []
            
            for p in range(self.D):
                W_in_p = self.W_in_list[p]
                
                # Generate single step
                out_p, new_h = self.heads[p](recent_input, hidden_states[p], W_in_p)
                next_step.append(out_p[:, -1:, :])  # Take only last time step
                new_hidden_states.append(new_h)
            
            hidden_states = new_hidden_states
            next_values = torch.cat(next_step, dim=-1)  # [B, 1, D]
            generated.append(next_values)
        
        # Return generated sequence (excluding initial observation)
        return torch.cat(generated[1:], dim=1)  # [B, seq_len, D]

    def get_granger_causality(self, threshold=1e-6):
        """Extract Granger causality matrix"""
        return self.get_causal_adjacency_matrix(threshold).cpu().numpy()


def train_phase1(model, dataloader, epochs=100, lr=1e-3, lambda_l1=1e-2, lr_prox=1e-2):
    """Phase 1: Causal discovery with sparsity regularization using ISTA"""
    # Main parameters (excluding W_in matrices)
    main_params = [p for name, p in model.named_parameters() 
                   if not any(x in name for x in ['W_in_list'])]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_sparsity_loss = 0
        
        for batch_idx, x in enumerate(dataloader):
            # Ensure we have enough time steps
            if x.size(1) < 2 * model.tau:
                continue
                
            # Split into past and current
            x_past = x[:, :model.tau, :]
            x_current = x[:, model.tau:2*model.tau, :]
            
            # Forward pass
            recon, mu, logsig = model(x_past, x_current, mode='train', phase=1)
            
            # Compute losses
            loss_recon = F.mse_loss(recon, x_current)
            kl_loss = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - torch.exp(2*logsig)) / x.size(0)
            
            # Sparsity loss on input weight matrices
            sparsity_loss = 0
            for W_in in model.W_in_list:
                sparsity_loss += torch.sum(torch.abs(W_in))
            sparsity_loss *= lambda_l1
            
            # Total loss (convex part only for main parameters)
            total_loss_batch = loss_recon + kl_loss
            
            # Backward pass on convex part
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # Proximal gradient step on W_in matrices (ISTA)
            model.apply_proximal_operator(lambda_l1, lr_prox)
            
            # Accumulate losses for logging
            total_loss += (total_loss_batch + sparsity_loss).item()
            total_recon_loss += loss_recon.item()
            total_kl_loss += kl_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        if epoch % 20 == 0:  # 减少打印频率
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_kl = total_kl_loss / len(dataloader)
            avg_sparsity = total_sparsity_loss / len(dataloader)
            
            print(f"Phase 1 Epoch {epoch}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Recon Loss: {avg_recon:.4f}")
            print(f"  KL Loss: {avg_kl:.4f}")
            print(f"  Sparsity Loss: {avg_sparsity:.4f}")
            
            # Compute sparsity percentage
            A = model.get_causal_adjacency_matrix()
            sparsity = (A == 0).float().mean() * 100
            print(f"  Sparsity: {sparsity:.1f}%")
            
            # Check current causal matrix
            print(f"  Current causal edges: {torch.sum(A).item():.0f}/{model.D*model.D}")
            print()


def train_phase2(model, dataloader, epochs=100, lr=1e-3):
    """Phase 2: Fix causal structure and improve generation with error compensation"""
    # Prune the weight matrices to fix causal structure
    with torch.no_grad():
        threshold = 1e-6
        for W_in in model.W_in_list:
            # Zero out small weights
            mask = torch.abs(W_in) > threshold
            W_in.data = W_in.data * mask.float()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, x in enumerate(dataloader):
            if x.size(1) < 2 * model.tau:
                continue
                
            x_past = x[:, :model.tau, :]
            x_current = x[:, model.tau:2*model.tau, :]
            
            # Forward pass with error compensation
            recon, mu, logsig, error_mu, error_logsig = model(
                x_past, x_current, mode='train', phase=2)
            
            # Compute losses
            loss_recon = F.mse_loss(recon, x_current)
            kl_loss = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - torch.exp(2*logsig)) / x.size(0)
            error_kl = -0.5 * torch.sum(1 + 2*error_logsig - error_mu.pow(2) - torch.exp(2*error_logsig)) / x.size(0)
            
            total_loss_batch = loss_recon + kl_loss + error_kl
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Keep causal structure fixed
            with torch.no_grad():
                threshold = 1e-6
                for W_in in model.W_in_list:
                    mask = torch.abs(W_in) > threshold
                    W_in.grad = W_in.grad * mask.float()
            
            optimizer.step()
            total_loss += total_loss_batch.item()
        
        if epoch % 10 == 0:
            print(f"Phase 2 Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    D = 5  # Number of variables
    hidden_dim = 64
    z_dim = 16
    tau = 10
    
    # Create model
    model = CRVAE(D, hidden_dim, z_dim, tau)
    
    # Generate synthetic data with known causal structure
    def generate_var_data(n_samples=1000, seq_len=50):
        """Generate VAR data with known causal structure"""
        # True causal matrix (sparse)
        A_true = torch.zeros(D, D)
        A_true[0, 1] = 0.5  # x1 -> x0
        A_true[1, 2] = 0.3  # x2 -> x1
        A_true[2, 0] = 0.4  # x0 -> x2
        # Add self-connections
        for i in range(D):
            A_true[i, i] = 0.6
        
        print("True causal matrix:")
        print(A_true.numpy())
        
        data = []
        for _ in range(n_samples):
            x = torch.zeros(seq_len, D)
            x[0] = torch.randn(D) * 0.1  # Initial condition
            
            for t in range(1, seq_len):
                x[t] = torch.matmul(A_true, x[t-1]) + torch.randn(D) * 0.1
            
            data.append(x.unsqueeze(0))
        
        return torch.cat(data, dim=0), A_true  # [n_samples, seq_len, D]
    
    # Generate training data with more samples
    train_data, A_true = generate_var_data(n_samples=500, seq_len=2*tau+10)  # 增加数据量
    
    # Create simple dataloader
    def create_dataloader(data, batch_size=32):
        n_samples, seq_len, D = data.shape
        max_start = seq_len - 2*tau
        
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_data = []
            for j in range(i, min(i + batch_size, n_samples)):
                # Random start position
                start = torch.randint(0, max_start, (1,)).item()
                sequence = data[j, start:start+2*tau, :]
                batch_data.append(sequence.unsqueeze(0))
            
            if batch_data:
                batches.append(torch.cat(batch_data, dim=0))
        
        return batches
    
    dataloader = create_dataloader(train_data, batch_size=16)
    
    # Training with better balanced parameters
    print("\nStarting Phase 1: Causal Discovery")
    train_phase1(model, dataloader, epochs=100, lambda_l1=0.02, lr_prox=0.01)  # 减少稀疏化强度
    
    print("\nStarting Phase 2: Generation Improvement")
    train_phase2(model, dataloader, epochs=30)
    
    # Evaluate causal discovery
    causal_matrix = model.get_granger_causality(threshold=0.1)  # 增加阈值
    print("\nLearned Granger Causality Matrix:")
    print(causal_matrix)
    
    print("\nTrue Causal Matrix:")
    print((A_true > 0).float().numpy())
    
    # Also try different thresholds
    print("\nTrying different thresholds:")
    for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
        causal_thresh = model.get_granger_causality(threshold=thresh)
        true_binary = (A_true > 0).float().numpy()
        learned_binary = (causal_thresh > 0).astype(float)
        accuracy = np.mean(learned_binary == true_binary)
        sparsity = np.mean(causal_thresh == 0) * 100
        print(f"  Threshold {thresh}: Accuracy={accuracy:.3f}, Sparsity={sparsity:.1f}%")
    
    # Generate new sequences
    model.eval()
    with torch.no_grad():
        # Use first sequence as context
        initial_context = train_data[0:1, :tau, :]
        generated_seq = model(initial_context, mode='generate', seq_len=20)
        print(f"\nGenerated sequence shape: {generated_seq.shape}")
        print("Sample generated values:")
        print(generated_seq[0, :5, :])  # First 5 time steps
