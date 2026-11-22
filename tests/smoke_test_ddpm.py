"""Offline smoke test for DDPM with RANDOM tensors (shapes, loss, tiny T=20 sampling).
Run:  python -m tests.smoke_test_ddpm
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ddpm import Diffusion
from src.train_ddpm import run

torch.manual_seed(0)
m = Diffusion(T=20, base=16)
x = torch.rand(8, 1, 28, 28) * 2 - 1
t = torch.randint(0, 20, (8,))
assert m.net(x, t).shape == x.shape
loss = m.loss(x)
loss.backward()
assert all(p.grad is not None for p in m.parameters())
full = Diffusion(T=1000, base=16)  # full schedule: x_T must be ~pure noise
assert (full.abar[1:] < full.abar[:-1]).all() and full.abar[-1] < 1e-3
print("abar_T (T=1000):", full.abar[-1].item())

dl = DataLoader(TensorDataset(torch.rand(128, 1, 28, 28), torch.zeros(128)), batch_size=64)
with tempfile.TemporaryDirectory() as d:
    _, res = run((dl, dl, dl), epochs=1, T=20, base=16, out_dir=d, device=torch.device("cpu"))
    tr = torch.load(f"{d}/ckpt/ddpm_traj.pt")
assert set(tr["traj"].keys()) == set(res["traj_steps"]) and tr["samples64"].shape == (64, 1, 28, 28)
print("n_params (base=64):", sum(p.numel() for p in Diffusion(1000, base=64).parameters()))
print("DDPM SMOKE TEST PASSED")
