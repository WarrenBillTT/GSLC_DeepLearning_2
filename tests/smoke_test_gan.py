"""Offline smoke test for GAN/DCGAN with RANDOM tensors (shapes + training loop only).
Run:  python -m tests.smoke_test_gan
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gan import build_gan
from src.train_gan import run

torch.manual_seed(0)
x = torch.rand(256, 1, 28, 28)
dl = DataLoader(TensorDataset(x, torch.zeros(256)), batch_size=64)

for kind in ["gan", "dcgan"]:
    G, D = build_gan(kind)
    z = torch.randn(8, 100)
    out = G.eval()(z)
    assert out.shape == (8, 1, 28, 28) and out.abs().max() <= 1.0
    assert D.eval()(out).shape == (8,)
    print(kind, "params G/D:", sum(p.numel() for p in G.parameters()), sum(p.numel() for p in D.parameters()))
    with tempfile.TemporaryDirectory() as d:
        _, _, res = run(kind, (dl, dl, dl), epochs=2, out_dir=d, device=torch.device("cpu"))
    assert len(res["history"]["loss_D"]) == 2 and res["latency_ms_batch64"] > 0
print("GAN SMOKE TEST PASSED")
