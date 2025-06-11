import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ---------- 编码器 ----------
class Encoder(nn.Module):
    def __init__(self, d_in: int, h: int, z: int):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.fc_mu = nn.Linear(h, z)
        self.fc_logsig = nn.Linear(h, z)

    def forward(self, x):                           # x: [B, τ, D]
        _, h = self.gru(x)                          # h: [1,B,H]
        h = h.squeeze(0)
        mu, logσ = self.fc_mu(h), self.fc_logsig(h)
        return mu, logσ

# ---------- 单头解码器 ----------
class DecoderHead(nn.Module):
    def __init__(self, d_in: int, h: int):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.fc_out = nn.Linear(h, 1)

    def forward(self, x_in, h0):                    # x_in:[B,T,d_in]
        out, h = self.gru(x_in, h0)                 # out:[B,T,H]
        return self.fc_out(out), h                  # ->[B,T,1]

# ---------- 误差补偿 ----------
class ErrorVAE(nn.Module):
    def __init__(self, d: int, h: int, z: int):
        super().__init__()
        self.enc = nn.GRU(d, h, batch_first=True)
        self.dec = nn.GRU(d, h, batch_first=True)
        self.mu, self.logσ  = nn.Linear(h, z), nn.Linear(h, z)
        self.z2h = nn.Linear(z, h)
        self.out = nn.Linear(h, d)

    def reparam(self, μ, logσ):
        # Fixed: Use .mul() instead of .mul_() to avoid in-place operations
        return μ + torch.exp(logσ).mul(0.5).mul(torch.randn_like(logσ))

    def forward(self, ε):                           # ε:[B,T,D]
        _, h = self.enc(ε)
        μ, logσ = self.mu(h.squeeze(0)), self.logσ(h.squeeze(0))
        z = self.reparam(μ, logσ)
        h0 = torch.tanh(self.z2h(z)).unsqueeze(0)
        dec_out, _ = self.dec(ε, h0)
        return self.out(dec_out), μ, logσ           # recon ε̂

# ---------- 主模型 ----------
class CRVAE(nn.Module):
    def __init__(self, D: int, H: int, Z: int, τ: int):
        super().__init__()
        self.D, self.H, self.τ = D, H, τ
        self.encoder = Encoder(D, H, Z)
        self.z2h = nn.Linear(Z, H)
        # 可学习W_in：D×H，每行代表一个父节点的隐向量映射
        self.W_in = nn.ParameterList(
            [nn.Parameter(0.01*torch.randn(D, H)) for _ in range(D)]
        )
        self.heads = nn.ModuleList([DecoderHead(H, H) for _ in range(D)])
        self.err_vae = ErrorVAE(D, H//2, Z//2)

    # ---------- util ----------
    @staticmethod
    def _reparam(μ, logσ):
        # Fixed: Use .mul() instead of .mul_() to avoid in-place operations
        return μ + torch.exp(logσ).mul(0.5).mul(torch.randn_like(logσ))

    # ---------- 前向 ----------
    def forward(self,
                x_past: torch.Tensor,              # [B,τ,D]
                x_cur: torch.Tensor,               # [B,τ,D]
                phase: int = 1):
        μ, logσ = self.encoder(x_past)
        z = self._reparam(μ, logσ)
        h0 = torch.tanh(self.z2h(z)).unsqueeze(0)   # [1,B,H]

        # 解码输入 = 上一时刻真实 + teacher forcing
        dec_in = torch.cat([x_past[:, -1:, :], x_cur[:, :-1, :]], dim=1)  # [B,τ,D]
        recon = []
        h_each = [h0 for _ in range(self.D)]

        # 多头并行
        for p in range(self.D):
            x_sel = dec_in @ self.W_in[p]          # [B,τ,H]
            out, h_new = self.heads[p](x_sel, h_each[p])
            recon.append(out)                      # [B,τ,1]
            h_each[p] = h_new
        recon = torch.cat(recon, dim=-1)           # [B,τ,D]

        if phase == 1:                             # 仅主 VAE
            return recon, μ, logσ, None, None
        # phase==2 : 误差补偿
        ε = (x_cur - recon).detach()
        ε̂, μe, logσe = self.err_vae(ε)
        recon_plus = recon + ε̂
        return recon_plus, μ, logσ, μe, logσe

    # ---------- 生成 ----------
    @torch.no_grad()
    def generate(self, x_context: torch.Tensor, T: int):
        B = x_context.size(0)
        μ, logσ = self.encoder(x_context[:, -self.τ:, :])
        z = self._reparam(μ, logσ)
        h_each = [torch.tanh(self.z2h(z)).unsqueeze(0) for _ in range(self.D)]

        seq = [x_context[:, -1:, :]]               # last obs as t0
        for _ in range(T):
            recent = torch.cat(seq[-self.τ:], dim=1) if len(seq)>=self.τ \
                     else torch.cat([x_context[:, -(self.τ-len(seq)):, :]]+seq,1)
            x_next=[]
            for p in range(self.D):
                x_sel = recent @ self.W_in[p]
                out, h_new = self.heads[p](x_sel[:, -1:, :], h_each[p])
                x_next.append(out)
                h_each[p] = h_new
            seq.append(torch.cat(x_next, -1))
        return torch.cat(seq[1:],1)                # drop initial

    # ---------- 提取因果图 ----------
    def granger_matrix(self, thr: float = 1e-6) -> torch.Tensor:
        A = torch.zeros(self.D, self.D, device=self.W_in[0].device)
        for p in range(self.D):
            col_norm = torch.norm(self.W_in[p], dim=1)  # L2 over H
            A[p] = (col_norm > thr).float()
        return A

    # ---------- ISTA (FIXED) ----------
    def ista_step(self, λ: float, lr: float):
        """ISTA: W <- prox_{λ·lr}( W - lr * ∇L_c )"""
        with torch.no_grad():
            for p in range(self.D):
                W = self.W_in[p]
                if W.grad is None:                   # 保证有梯度
                    continue
                # ❶ 先做一次梯度下降
                W_tmp = W - lr * W.grad

                # ❷ 再做 group-L1 软阈值（对每行做 L2-norm）
                row_norm = torch.norm(W_tmp, dim=1, keepdim=True)   # [D,1]
                shrink = torch.clamp(1 - lr*λ/row_norm, min=0.)
                self.W_in[p].data = W_tmp * shrink

                # ❸ 手动清零 grad，防止累积
                W.grad.zero_()

# ---------- 两阶段训练器 ----------
class CRVAETrainer:
    def __init__(self, model: CRVAE, λ_l1: float = 5e-2, lr: float = 1e-3):
        self.m = model
        self.lr = lr
        self.λ = λ_l1
        self.opt = torch.optim.Adam(
            [p for n,p in model.named_parameters() if "W_in" not in n], lr=lr)

    # ---- phase‑I ----
    def step_stage1(self, x_batch: torch.Tensor):
        B, T, D = x_batch.shape
        x_past, x_cur = torch.split(x_batch, self.m.τ, dim=1)
        recon, μ, logσ, *_ = self.m(x_past, x_cur, phase=1)

        recon_loss = F.mse_loss(recon, x_cur)
        kl = -0.5*torch.mean(1+2*logσ - μ.pow(2) - torch.exp(2*logσ))
        loss = recon_loss + kl

        self.opt.zero_grad()
        loss.backward()
        self.m.ista_step(self.λ, self.lr)           # 在backward之后，step之前
        self.opt.step()
        return loss.item()

    # ---- phase‑II ----
    def step_stage2(self, x_batch: torch.Tensor):
        B, T, D = x_batch.shape
        x_past, x_cur = torch.split(x_batch, self.m.τ, dim=1)
        recon, μ, logσ, μe, logσe = self.m(x_past, x_cur, phase=2)

        loss_recon = F.mse_loss(recon, x_cur)
        kl_main = -0.5*torch.mean(1+2*logσ - μ.pow(2) - torch.exp(2*logσ))
        kl_err  = -0.5*torch.mean(1+2*logσe - μe.pow(2) - torch.exp(2*logσe))
        loss = loss_recon + kl_main + kl_err

        self.opt.zero_grad()
        loss.backward()

        # 固定判零后的 W_in
        with torch.no_grad():
            mask = self.m.granger_matrix().bool()
            for p in range(self.m.D):
                grad = self.m.W_in[p].grad
                grad[~mask[p]] = 0.

        self.opt.step()
        return loss.item()



# ---------- 1. 生成或读取原始序列 ----------
# 例：生成 1 条 Henon 6 维序列，长度 3000
def henon_map_steps(T=3000, D=6, a=1.4, b=0.3, e=0.3):
    x = np.zeros((T, D), dtype=np.float32)
    x[0] = np.random.randn(D)
    x[1] = np.random.randn(D)
    for t in range(2, T):
        x[t, 0] = a - x[t-1, 0]**2 + b * x[t-2, 0]
        for p in range(1, D):
            parent = e * x[t-1, p-1] + (1-e) * x[t-1, p]
            x[t, p] = a - parent**2 + b * x[t-2, p]
    # 归一化到 [0,1]（和论文一致）
    x_min, x_max = x.min(0, keepdims=True), x.max(0, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-7)

raw_series = henon_map_steps()        # shape [3000, 6]

# ---------- 2. 切成长度 2τ (=20) 的滑动窗口 ----------
tau = 10
seq_len = 2 * tau
windows = []
for start in range(0, raw_series.shape[0] - seq_len + 1):
    windows.append(raw_series[start:start + seq_len])
windows = np.stack(windows)           # shape [N, 20, 6]
print("生成窗口数:", windows.shape[0])

# ---------- 3. 转成 DataLoader ----------
tensor_data = torch.from_numpy(windows)          # dtype=float32
dataset = TensorDataset(tensor_data)             # 只有一个字段
dataloader = DataLoader(dataset,
                        batch_size=256,
                        shuffle=True,
                        drop_last=True)          # 保证每批大小一致

# ---------- 4. Set device and create model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 调整超参数以更好地学习因果关系
model   = CRVAE(D=6, H=64, Z=32, τ=tau).to(device)
trainer = CRVAETrainer(model, λ_l1=0.03, lr=2e-3)  # 降低λ_l1，提高lr

# -------- Stage I --------
print("Starting Stage I training...")
for epoch in range(2000):
    epoch_loss = 0
    for xb, in dataloader:                 # 注意逗号：TensorDataset 只有一个字段
        loss = trainer.step_stage1(xb.to(device))
        epoch_loss += loss
    if (epoch + 1) % 100 == 0:
        print(f"Stage I - Epoch {epoch+1}/2000, Loss: {epoch_loss/len(dataloader):.4f}")
        # 检查因果矩阵
        causal_matrix = model.granger_matrix(thr=1e-6).cpu().numpy()
        non_zero_edges = (causal_matrix > 0).sum()
        print(f"  Non-zero edges discovered: {non_zero_edges}")

# -------- Stage II --------
print("\nStarting Stage II training...")
for epoch in range(1000):
    epoch_loss = 0
    for xb, in dataloader:
        loss = trainer.step_stage2(xb.to(device))
        epoch_loss += loss
    if (epoch + 1) % 100 == 0:
        print(f"Stage II - Epoch {epoch+1}/1000, Loss: {epoch_loss/len(dataloader):.4f}")

print("\nTraining completed!")


# ---------- 5. 评估和可视化 ----------
import matplotlib.pyplot as plt

# 5.1 提取并可视化因果图
print("\n=== Causal Graph Discovery ===")
causal_matrix = model.granger_matrix(thr=1e-6).cpu().numpy()  # 使用更低的阈值
print("Discovered causal structure:")
print(causal_matrix)

# 真实的因果关系（Henon map: node i depends on i-1）
true_causal = np.zeros((6, 6))
for i in range(1, 6):
    true_causal[i, i-1] = 1  # node i depends on node i-1

# 计算准确率
correct = (causal_matrix == true_causal).sum()
total = causal_matrix.size
accuracy = correct / total
print(f"\nCausal discovery accuracy: {accuracy:.2%}")

# 计算更详细的指标
true_edges = (true_causal > 0).sum()
discovered_edges = (causal_matrix > 0).sum()
tp = ((causal_matrix > 0) & (true_causal > 0)).sum()
fp = ((causal_matrix > 0) & (true_causal == 0)).sum()
fn = ((causal_matrix == 0) & (true_causal > 0)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDetailed metrics:")
print(f"  True edges: {true_edges}")
print(f"  Discovered edges: {discovered_edges}")
print(f"  True positives: {tp}")
print(f"  False positives: {fp}")
print(f"  False negatives: {fn}")
print(f"  Precision: {precision:.2%}")
print(f"  Recall: {recall:.2%}")
print(f"  F1 score: {f1:.2%}")

# 可视化因果图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(true_causal, cmap='Blues', vmin=0, vmax=1)
ax1.set_title('True Causal Graph')
ax1.set_xlabel('Parent Node')
ax1.set_ylabel('Child Node')
for i in range(6):
    for j in range(6):
        ax1.text(j, i, f'{int(true_causal[i,j])}', ha='center', va='center')

ax2.imshow(causal_matrix, cmap='Blues', vmin=0, vmax=1)
ax2.set_title('Discovered Causal Graph')
ax2.set_xlabel('Parent Node')
ax2.set_ylabel('Child Node')
for i in range(6):
    for j in range(6):
        ax2.text(j, i, f'{int(causal_matrix[i,j])}', ha='center', va='center')
plt.tight_layout()
plt.show()

# 5.2 测试预测性能
print("\n=== Prediction Performance ===")
model.eval()
with torch.no_grad():
    # 使用测试数据
    test_batch = next(iter(dataloader))[0].to(device)
    x_past, x_cur = torch.split(test_batch, tau, dim=1)
    
    # Stage 1 预测
    recon1, _, _, _, _ = model(x_past, x_cur, phase=1)
    mse1 = F.mse_loss(recon1, x_cur).item()
    
    # Stage 2 预测（带误差补偿）
    recon2, _, _, _, _ = model(x_past, x_cur, phase=2)
    mse2 = F.mse_loss(recon2, x_cur).item()
    
    print(f"Stage 1 MSE: {mse1:.6f}")
    print(f"Stage 2 MSE (with error compensation): {mse2:.6f}")
    print(f"Improvement: {(1-mse2/mse1)*100:.1f}%")

# 5.3 可视化预测结果
print("\n=== Visualizing Predictions ===")
# 选择一个样本
sample_idx = 0
x_true = x_cur[sample_idx].cpu().numpy()
x_pred1 = recon1[sample_idx].cpu().numpy()
x_pred2 = recon2[sample_idx].cpu().numpy()

# 绘制每个维度的预测
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for dim in range(6):
    axes[dim].plot(x_true[:, dim], 'k-', label='True', linewidth=2)
    axes[dim].plot(x_pred1[:, dim], 'b--', label='Stage 1', alpha=0.7)
    axes[dim].plot(x_pred2[:, dim], 'r--', label='Stage 2', alpha=0.7)
    axes[dim].set_title(f'Dimension {dim}')
    axes[dim].set_xlabel('Time')
    axes[dim].legend()
    axes[dim].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.4 测试生成能力
print("\n=== Generation Test ===")
# 使用前tau个时间步作为context
context = test_batch[:1, :tau, :].to(device)
generated = model.generate(context, T=50)  # 生成50个时间步

# 可视化生成的序列
gen_np = generated[0].cpu().numpy()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for dim in range(6):
    axes[dim].plot(gen_np[:, dim], 'g-', linewidth=2)
    axes[dim].set_title(f'Generated Dimension {dim}')
    axes[dim].set_xlabel('Time')
    axes[dim].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.5 统计分析
print("\n=== Statistical Analysis ===")
# 计算生成数据的统计特性
gen_mean = gen_np.mean(axis=0)
gen_std = gen_np.std(axis=0)
true_mean = raw_series.mean(axis=0)
true_std = raw_series.std(axis=0)

print("Mean comparison:")
for i in range(6):
    print(f"  Dim {i}: True={true_mean[i]:.3f}, Generated={gen_mean[i]:.3f}")
print("\nStd comparison:")
for i in range(6):
    print(f"  Dim {i}: True={true_std[i]:.3f}, Generated={gen_std[i]:.3f}")

# 5.6 查看学到的W_in权重模式
print("\n=== Learned Weight Patterns ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for p in range(6):
    W = model.W_in[p].data.cpu().numpy()
    im = axes[p].imshow(W.T, cmap='RdBu_r', aspect='auto')
    axes[p].set_title(f'W_in[{p}] (Node {p} inputs)')
    axes[p].set_xlabel('Input Dimension')
    axes[p].set_ylabel('Hidden Units')
    plt.colorbar(im, ax=axes[p])
plt.tight_layout()
plt.show()

# 5.7 可视化W_in的行范数（显示稀疏性）
print("\n=== W_in Row Norms (Sparsity) ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
axes = axes.flatten()
for p in range(6):
    W = model.W_in[p].data.cpu().numpy()
    row_norms = np.linalg.norm(W, axis=1)
    axes[p].bar(range(len(row_norms)), row_norms)
    axes[p].set_title(f'W_in[{p}] Row Norms')
    axes[p].set_xlabel('Input Dimension')
    axes[p].set_ylabel('L2 Norm')
    axes[p].axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Threshold')
    axes[p].legend()
plt.tight_layout()
plt.show()
