"""Train Vanilla GAN / DCGAN and log the diagnostics needed for Q3 (stability, vanishing gradient, mode collapse).

Usage:
    python -m src.train_gan --model gan   --epochs 50
    python -m src.train_gan --model dcgan --epochs 50
Writes results/<model>.json and results/ckpt/<model>_G.pt / <model>_D.pt
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from .common import SEED, get_device, get_loaders, measure_latency, set_seed
from .models.gan import NZ, build_gan


def diversity(x: torch.Tensor) -> float:
    """Mean pairwise L2 distance between flattened images (low value => mode collapse)."""
    f = x.flatten(1)
    d = torch.cdist(f, f)
    n = f.size(0)
    return (d.sum() / (n * (n - 1))).item()


def grad_norm(params) -> float:
    g = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return torch.cat(g).norm().item() if g else 0.0


def run(kind, loaders, epochs=50, lr=2e-4, nz=NZ, out_dir="results", device=None):
    set_seed(SEED)
    device = device or get_device()
    train_loader = loaders[0]
    G, D = build_gan(kind, nz)
    G, D = G.to(device), D.to(device)
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    fixed_z = torch.randn(256, nz, device=device)  # fixed noise -> comparable samples across epochs
    real_ref = next(iter(train_loader))[0][:256].to(device) * 2 - 1
    real_div = diversity(real_ref)

    keys = ["loss_D", "loss_G", "D_real", "D_fake", "grad_G_first", "grad_G_last", "diversity", "epoch_time"]
    hist = {k: [] for k in keys}
    snaps = {}

    for ep in range(epochs):
        G.train(); D.train()
        t0 = time.perf_counter()
        acc = {k: 0.0 for k in keys[:6]}
        n = 0
        for x, _ in train_loader:
            x = x.to(device) * 2 - 1
            b = x.size(0)
            ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
            # --- D step
            z = torch.randn(b, nz, device=device)
            fake = G(z).detach()
            lr_, lf_ = D(x), D(fake)
            lossD = F.binary_cross_entropy_with_logits(lr_, ones) + F.binary_cross_entropy_with_logits(lf_, zeros)
            oD.zero_grad(); lossD.backward(); oD.step()
            # --- G step (non-saturating)
            z = torch.randn(b, nz, device=device)
            lg = D(G(z))
            lossG = F.binary_cross_entropy_with_logits(lg, ones)
            oG.zero_grad(); lossG.backward()
            gp = [p for p in G.parameters() if p.dim() > 1]  # weight matrices only
            first, last = grad_norm([gp[0]]), grad_norm([gp[-1]])
            oG.step()
            for k, v in zip(keys[:6], [lossD.item(), lossG.item(), torch.sigmoid(lr_).mean().item(),
                                       torch.sigmoid(lf_).mean().item(), first, last]):
                acc[k] += v * b
            n += b
        if device.type == "cuda":
            torch.cuda.synchronize()
        for k in acc:
            hist[k].append(acc[k] / n)
        G.eval()
        with torch.no_grad():
            samples = G(fixed_z)
        hist["diversity"].append(diversity(samples))
        hist["epoch_time"].append(time.perf_counter() - t0)
        if ep in (0, epochs // 2, epochs - 1):
            snaps[ep + 1] = samples[:16].cpu()
        print(f"[{kind}] ep {ep + 1}/{epochs} D={hist['loss_D'][-1]:.3f} G={hist['loss_G'][-1]:.3f} "
              f"D(x)={hist['D_real'][-1]:.2f} D(G(z))={hist['D_fake'][-1]:.2f} "
              f"div={hist['diversity'][-1]:.2f}/{real_div:.2f} t={hist['epoch_time'][-1]:.1f}s")

    G.eval()
    z64 = torch.randn(64, nz, device=device)
    tail = hist["loss_G"][-10:]
    res = {
        "model": kind, "latent_dim": nz, "epochs": epochs, "seed": SEED, "lr": lr,
        "train_time_per_epoch_s": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
        "latency_ms_batch64": measure_latency(G, z64),
        "real_diversity_ref": real_div,
        "final_diversity_ratio": hist["diversity"][-1] / real_div,
        "min_diversity_ratio": min(hist["diversity"]) / real_div,
        "loss_G_std_last10": float(torch.tensor(tail).std().item()) if len(tail) > 1 else 0.0,
        "history": hist,
    }
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    torch.save(G.state_dict(), os.path.join(out_dir, "ckpt", f"{kind}_G.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "ckpt", f"{kind}_D.pt"))
    torch.save(snaps, os.path.join(out_dir, "ckpt", f"{kind}_snaps.pt"))
    with open(os.path.join(out_dir, f"{kind}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return G, D, res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gan", "dcgan"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--data_root", default="./data")
    a = p.parse_args()
    run(a.model, get_loaders(a.data_root), a.epochs, a.lr)
