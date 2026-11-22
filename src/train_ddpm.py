"""Train DDPM and record sampling latency + the reverse-diffusion trajectory (Figure 5).

Usage:  python -m src.train_ddpm --epochs 30            (T=1000, base=64)
Writes results/ddpm.json, results/ckpt/ddpm_ema.pt, results/ckpt/ddpm_traj.pt
"""
import argparse
import json
import os
import time

import torch

from .common import SEED, get_device, get_loaders, set_seed
from .models.ddpm import EMA, Diffusion


def run(loaders, epochs=30, lr=2e-4, T=1000, base=64, out_dir="results", device=None, n_traj=8):
    set_seed(SEED)
    device = device or get_device()
    train_loader, val_loader, _ = loaders
    model = Diffusion(T, base=base).to(device)
    ema = EMA(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "epoch_time": []}

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device) * 2 - 1
            loss = model.loss(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        hist["epoch_time"].append(time.perf_counter() - t0)
        hist["train_loss"].append(tot / n)
        # validation noise-prediction loss with the EMA weights
        ema.shadow.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for x, _ in val_loader:
                x = x.to(device) * 2 - 1
                vt += ema.shadow.loss(x).item() * x.size(0)
                vn += x.size(0)
        hist["val_loss"].append(vt / vn)
        print(f"[ddpm] ep {ep + 1}/{epochs} train={hist['train_loss'][-1]:.5f} "
              f"val(EMA)={hist['val_loss'][-1]:.5f} t={hist['epoch_time'][-1]:.1f}s")

    # --- sampling latency for ONE batch of 64 (full T-step reverse process) + trajectory for Figure 5
    g = ema.shadow
    steps = sorted({T - 1, int(T * .75), T // 2, T // 4, T // 10, T // 20, 0}, reverse=True)
    set_seed(SEED)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    imgs, traj = g.sample(64, device, traj_steps=set(steps), batch=64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) * 1000
    traj = {k: v[:n_traj] for k, v in traj.items()}

    res = {
        "model": "ddpm", "T": T, "base_channels": base, "epochs": epochs, "seed": SEED, "lr": lr,
        "train_time_per_epoch_s": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
        "latency_ms_batch64": lat, "latency_ms_per_step": lat / T,
        "n_params": sum(p.numel() for p in g.parameters()),
        "traj_steps": steps, "history": hist,
    }
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    torch.save({"state": g.state_dict(), "T": T, "base": base}, os.path.join(out_dir, "ckpt", "ddpm_ema.pt"))
    torch.save({"traj": traj, "samples64": imgs.cpu()}, os.path.join(out_dir, "ckpt", "ddpm_traj.pt"))
    with open(os.path.join(out_dir, "ddpm.json"), "w") as f:
        json.dump(res, f, indent=2)
    return g, res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--data_root", default="./data")
    a = p.parse_args()
    run(get_loaders(a.data_root), a.epochs, a.lr, a.T, a.base)
