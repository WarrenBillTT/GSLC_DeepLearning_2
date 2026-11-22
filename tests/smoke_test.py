"""Offline smoke test with RANDOM tensors (shapes / gradients / metrics only; NOT experiment results).
Run:  python -m tests.smoke_test
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import psnr_per_image, ssim_per_image
from src.models.autoencoder import VAE, Autoencoder, SparseAutoencoder
from src.train_ae import run

torch.manual_seed(0)
x = torch.rand(256, 1, 28, 28)
y = torch.randint(0, 10, (256,))
dl = DataLoader(TensorDataset(x, y), batch_size=64)

# metrics sanity: identical images -> SSIM 1, PSNR very large
assert torch.allclose(ssim_per_image(x[:8], x[:8]), torch.ones(8), atol=1e-5)
assert psnr_per_image(x[:8], x[:8]).min() > 90
assert ssim_per_image(x[:8], torch.rand(8, 1, 28, 28)).max() < 0.5

for m in [Autoencoder(32), SparseAutoencoder(64), SparseAutoencoder(256), VAE(32)]:
    loss, logs = m.loss(x[:64])
    loss.backward()
    assert all(p.grad is not None for p in m.parameters()), type(m)
    print(type(m).__name__, "ok", {k: round(v, 3) for k, v in logs.items()})

with tempfile.TemporaryDirectory() as d:
    for name, z in [("ae", 32), ("sparse_ae", 64), ("vae", 32)]:
        _, res = run(name, (dl, dl, dl), latent=z, epochs=1, out_dir=d, device=torch.device("cpu"))
        assert {"test_mse", "test_psnr", "test_ssim", "latency_ms_batch64"} <= res.keys()
print("SMOKE TEST PASSED")
