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
    def __init__(self, max_input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(max_input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, h0):  # x: [B, T, k] where k is number of causal parents
        out, final_h = self.rnn(x, h0)
        return self.fc_out(out), final_h  # [B, T, 1], [1, B, hidden]

class ErrorCompensationVAE(nn.Module):
    """Error compensation network for instantaneous effects"""
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, z_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.fc_z_to_h = nn.Linear(z_dim, hidden_dim)

    def forward(self, x):
        # Encode
        _, h_enc = self.encoder(x)
        mu = self.fc_mu(h_enc.squeeze(0))
        logsig = self.fc_logsigma(h_enc.squeeze(0))
        
        # Reparameterize
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode
        h0 = torch.tanh(self.fc_z_to_h(z)).unsqueeze(0)
        out, _ = self.decoder(x, h0)
        recon = self.fc_out(out)
        
        return recon, mu, logsig

class CRVAE(nn.Module):
    def __init__(self, D, hidden_dim, z_dim, tau, adj_init=None):
        super().__init__()
        self.D = D
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        # Encoder
        self.encoder = Encoder(D, hidden_dim, z_dim)
        
        # Multi-head decoder - one head per variable
        self.heads = nn.ModuleList([DecoderHead(D, hidden_dim) for _ in range(D)])
        
        # Reparameterization layer
        self.re_fc = nn.Linear(z_dim, hidden_dim)
        
        # Learnable causal adjacency matrix
        if adj_init is None:
            adj_init = torch.ones(D, D)  # Start with full connectivity
        self.A = nn.Parameter(adj_init.clone().float())
        
        # Error compensation network
        self.error_vae = ErrorCompensationVAE(D, hidden_dim//2, z_dim//2)

    def reparameterize(self, mu, logsig):
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_causal_mask(self, threshold=1e-6):
        """Get binary causal mask"""
        return (torch.abs(self.A) > threshold).float()

    def forward(self, x_early, x_recent, mode='train', phase=1):
        """
        x_early: [B, tau, D] - past observations for encoder
        x_recent: [B, tau, D] - recent observations for decoder input
        mode: 'train' or 'generate'
        phase: 1 (causal discovery) or 2 (generation improvement)
        """
        if mode == 'train':
            return self._forward_train(x_early, x_recent, phase)
        else:
            return self._forward_generate(x_early)

    def _forward_train(self, x_early, x_recent, phase):
        # Encode past observations
        mu, logsig = self.encoder(x_early)
        z = self.reparameterize(mu, logsig)
        
        # Initialize decoder hidden state
        h0 = torch.tanh(self.re_fc(z)).unsqueeze(0)  # [1, B, hidden]
        
        # Multi-head decoding
        recon_list = []
        causal_mask = self.get_causal_mask() if phase == 1 else self.get_causal_mask(threshold=0)
        
        for p in range(self.D):
            # Apply causal mask for p-th variable
            mask = causal_mask[p]  # [D]
            
            # Select causal parents based on mask
            parent_indices = torch.nonzero(mask, as_tuple=True)[0]
            if len(parent_indices) > 0:
                causal_input = x_recent[:, :, parent_indices]  # [B, tau, num_parents]
            else:
                # If no parents, use zero input
                causal_input = torch.zeros(x_recent.size(0), x_recent.size(1), 1, 
                                         device=x_recent.device)
            
            # Decode for p-th variable
            recon_p, _ = self.heads[p](causal_input, h0)  # [B, tau, 1]
            recon_list.append(recon_p)
        
        recon = torch.cat(recon_list, dim=-1)  # [B, tau, D]
        
        if phase == 2:
            # Phase 2: Add error compensation
            error = x_recent - recon.detach()  # Detach to avoid affecting causal learning
            error_recon, error_mu, error_logsig = self.error_vae(error)
            recon = recon + error_recon
            return recon, mu, logsig, error_mu, error_logsig
        
        return recon, mu, logsig

    def _forward_generate(self, initial_context, seq_len=20):
        """Generate sequences autoregressively"""
        batch_size = initial_context.size(0)
        device = initial_context.device
        
        # Encode initial context
        mu, logsig = self.encoder(initial_context)
        z = self.reparameterize(mu, logsig)
        h0 = torch.tanh(self.re_fc(z)).unsqueeze(0)
        
        # Initialize with context
        generated = initial_context.clone()
        causal_mask = self.get_causal_mask(threshold=0)
        
        # Hidden states for each head
        hidden_states = [h0.clone() for _ in range(self.D)]
        
        for t in range(seq_len):
            # Get most recent tau steps as input
            if generated.size(1) >= self.tau:
                recent_input = generated[:, -self.tau:, :]
            else:
                # Pad if necessary
                pad_len = self.tau - generated.size(1)
                padding = torch.zeros(batch_size, pad_len, self.D, device=device)
                recent_input = torch.cat([padding, generated], dim=1)
            
            # Generate next time step
            next_step = []
            new_hidden_states = []
            
            for p in range(self.D):
                mask = causal_mask[p]
                parent_indices = torch.nonzero(mask, as_tuple=True)[0]
                
                if len(parent_indices) > 0:
                    causal_input = recent_input[:, :, parent_indices]
                else:
                    causal_input = torch.zeros(batch_size, self.tau, 1, device=device)
                
                # Generate single step
                out_p, new_h = self.heads[p](causal_input[:, -1:, :], hidden_states[p])
                next_step.append(out_p[:, -1:, :])  # [B, 1, 1]
                new_hidden_states.append(new_h)
            
            hidden_states = new_hidden_states
            next_values = torch.cat(next_step, dim=-1)  # [B, 1, D]
            generated = torch.cat([generated, next_values], dim=1)
        
        return generated[:, self.tau:, :]  # Return only generated part

    def get_granger_causality(self):
        """Extract Granger causality matrix"""
        with torch.no_grad():
            mask = self.get_causal_mask(threshold=0)
            return mask.cpu().numpy()


def train_phase1(model, dataloader, epochs=100, lr=1e-3, lambda_l1=1e-2, lr_prox=1e-2):
    """Phase 1: Causal discovery with sparsity regularization"""
    # Separate optimizers for different parameter groups
    main_params = [p for name, p in model.named_parameters() if 'A' not in name]
    optimizer = torch.optim.Adam(main_params, lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, x in enumerate(dataloader):
            # Split into early and recent
            x_early = x[:, :model.tau, :]
            x_recent = x[:, model.tau:2*model.tau, :]
            
            # Forward pass
            recon, mu, logsig = model(x_early, x_recent, mode='train', phase=1)
            
            # Losses
            loss_recon = F.mse_loss(recon, x_recent)
            kl_loss = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - torch.exp(2*logsig)) / x.size(0)
            sparsity_loss = lambda_l1 * torch.sum(torch.abs(model.A))
            
            total_loss_batch = loss_recon + kl_loss + sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            # Proximal update on A
            with torch.no_grad():
                model.A.data = torch.sign(model.A) * torch.clamp(
                    torch.abs(model.A) - lr_prox * lambda_l1, min=0.0)
            
            total_loss += total_loss_batch.item()
        
        if epoch % 10 == 0:
            print(f"Phase 1 Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            sparsity = torch.mean((torch.abs(model.A) > 1e-6).float()) * 100
            print(f"Sparsity: {sparsity:.1f}%")


def train_phase2(model, dataloader, epochs=100, lr=1e-3):
    """Phase 2: Fix causal structure and improve generation"""
    # Freeze sparsity pattern in A
    with torch.no_grad():
        model.A.data = (torch.abs(model.A) > 1e-6).float() * model.A
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, x in enumerate(dataloader):
            x_early = x[:, :model.tau, :]
            x_recent = x[:, model.tau:2*model.tau, :]
            
            # Forward pass with error compensation
            recon, mu, logsig, error_mu, error_logsig = model(
                x_early, x_recent, mode='train', phase=2)
            
            # Losses
            loss_recon = F.mse_loss(recon, x_recent)
            kl_loss = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - torch.exp(2*logsig)) / x.size(0)
            error_kl = -0.5 * torch.sum(1 + 2*error_logsig - error_mu.pow(2) - torch.exp(2*error_logsig)) / x.size(0)
            
            total_loss_batch = loss_recon + kl_loss + error_kl
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if epoch % 10 == 0:
            print(f"Phase 2 Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")


# Example usage
if __name__ == "__main__":
    # Model parameters
    D = 5  # Number of variables
    hidden_dim = 64
    z_dim = 16
    tau = 10
    
    # Create model
    model = CRVAE(D, hidden_dim, z_dim, tau)
    
    # Dummy data (replace with your actual dataloader)
    # Data should be of shape [batch_size, 2*tau, D]
    dummy_data = torch.randn(32, 2*tau, D)
    dataloader = [dummy_data]  # Replace with actual DataLoader
    
    # Training
    print("Starting Phase 1: Causal Discovery")
    train_phase1(model, dataloader, epochs=50)
    
    print("\nStarting Phase 2: Generation Improvement")
    train_phase2(model, dataloader, epochs=50)
    
    # Extract causal relationships
    causal_matrix = model.get_granger_causality()
    print("\nLearned Granger Causality Matrix:")
    print(causal_matrix)
    
    # Generate new sequences
    model.eval()
    with torch.no_grad():
        initial_context = torch.randn(1, tau, D)
        generated_seq = model(initial_context, None, mode='generate')
        print(f"\nGenerated sequence shape: {generated_seq.shape}")
