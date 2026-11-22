"""DDPM (Ho et al., 2020) for 28x28 grayscale images in [-1, 1].

Forward  : q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
Objective: L_simple = E || eps - eps_theta(x_t, t) ||^2          (epsilon-prediction)
Reverse  : x_{t-1} = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1-abar_t) eps_theta) + sigma_t z,  sigma_t^2 = beta_t
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base: int = 64, tdim: int = 256):
        super().__init__()
        self.base, self.tdim = base, tdim
        self.t_mlp = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, base, 3, padding=1)
        self.d1, self.down1 = ResBlock(base, base, tdim), nn.Conv2d(base, base, 3, 2, 1)             # 28 -> 14
        self.d2, self.down2 = ResBlock(base, 2 * base, tdim), nn.Conv2d(2 * base, 2 * base, 3, 2, 1)  # 14 -> 7
        self.m1, self.m2 = ResBlock(2 * base, 2 * base, tdim), ResBlock(2 * base, 2 * base, tdim)
        self.up2, self.u2 = nn.ConvTranspose2d(2 * base, 2 * base, 4, 2, 1), ResBlock(4 * base, 2 * base, tdim)
        self.up1, self.u1 = nn.ConvTranspose2d(2 * base, base, 4, 2, 1), ResBlock(2 * base, base, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.t_mlp(timestep_embedding(t, self.base))
        h0 = self.inc(x)
        h1 = self.d1(h0, temb)                  # 28x28
        h2 = self.d2(self.down1(h1), temb)      # 14x14
        h = self.down2(h2)                      # 7x7
        h = self.m2(self.m1(h, temb), temb)
        h = self.u2(torch.cat([self.up2(h), h2], 1), temb)
        h = self.u1(torch.cat([self.up1(h), h1], 1), temb)
        return self.out(h)


class Diffusion(nn.Module):
    def __init__(self, T: int = 1000, beta1: float = 1e-4, betaT: float = 0.02, base: int = 64):
        super().__init__()
        self.T = T
        self.net = UNet(base)
        betas = torch.linspace(beta1, betaT, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(self.net(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, n: int, device, traj_steps=None, batch: int = 500):
        """Ancestral sampling. traj_steps: iterable of t values whose x_t is recorded (e.g. [999, 500, 0]).

        Returns images (n,1,28,28) in [-1,1]; if traj_steps given, also {t: x_t} (for the first <=16 samples).
        """
        outs, traj = [], {}
        for s in range(0, n, batch):
            b = min(batch, n - s)
            x = torch.randn(b, 1, 28, 28, device=device)
            for t in reversed(range(self.T)):
                tt = torch.full((b,), t, device=device, dtype=torch.long)
                eps = self.net(x, tt)
                mean = (x - self.betas[t] / (1 - self.abar[t]).sqrt() * eps) / self.alphas[t].sqrt()
                x = mean + (self.betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0.0)
                if traj_steps is not None and s == 0 and t in traj_steps:
                    traj[t] = x[:16].clamp(-1, 1).cpu()
            outs.append(x.clamp(-1, 1))
        out = torch.cat(outs)
        return (out, traj) if traj_steps is not None else out


class EMA:
    """Exponential moving average of weights; sampling uses the EMA copy."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ps, p in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
