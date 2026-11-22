"""Train + evaluate Vanilla AE / Sparse AE / VAE.

Usage:
    python -m src.train_ae --model ae        --latent 32 --epochs 30
    python -m src.train_ae --model sparse_ae --latent 64 --lam 1e-4 --epochs 30
    python -m src.train_ae --model vae       --latent 32 --epochs 30
Writes results/<tag>.json (metrics + loss history) and results/ckpt/<tag>.pt
"""
import argparse
import json
import os
import time

import torch

from .common import SEED, get_device, get_loaders, measure_latency, set_seed
from .metrics import evaluate_reconstruction
from .models.autoencoder import VAE, Autoencoder, SparseAutoencoder


def build_model(name: str, latent: int):
    if name == "ae":
        return Autoencoder(latent)
    if name == "sparse_ae":
        return SparseAutoencoder(latent)
    if name == "vae":
        return VAE(latent)
    raise ValueError(name)


def run(name, loaders, latent=32, epochs=30, lr=1e-3, lam=1e-4, out_dir="results", tag=None, device=None):
    set_seed(SEED)
    device = device or get_device()
    tag = tag or (f"{name}_z{latent}" + (f"_lam{lam:g}" if name == "sparse_ae" else ""))
    train_loader, val_loader, test_loader = loaders
    model = build_model(name, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_fn = model.reconstruct if name == "vae" else (lambda x: model(x)[0])

    hist = {"train_loss": [], "val_rec_mse": [], "epoch_time": []}
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        run_loss, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            loss, _ = model.loss(x, lam) if name == "sparse_ae" else model.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        hist["epoch_time"].append(time.perf_counter() - t0)
        hist["train_loss"].append(run_loss / n)
        model.eval()
        hist["val_rec_mse"].append(evaluate_reconstruction(recon_fn, val_loader, device)["mse"])
        print(f"[{tag}] ep {ep + 1}/{epochs} loss={hist['train_loss'][-1]:.5f} "
              f"val_mse={hist['val_rec_mse'][-1]:.5f} t={hist['epoch_time'][-1]:.1f}s")

    model.eval()
    test = evaluate_reconstruction(recon_fn, test_loader, device)
    x64 = next(iter(test_loader))[0][:64].to(device)
    res = {
        "tag": tag, "model": name, "latent_dim": latent, "lambda": lam if name == "sparse_ae" else None,
        "epochs": epochs, "seed": SEED,
        "train_time_per_epoch_s": sum(hist["epoch_time"]) / len(hist["epoch_time"]),
        "latency_ms_batch64": measure_latency(recon_fn, x64),
        "test_mse": test["mse"], "test_psnr": test["psnr"], "test_ssim": test["ssim"],
        "history": hist,
    }
    if name == "sparse_ae":
        res["latent_sparsity"] = model.sparsity(test_loader, device)
    if name == "vae":
        res["latency_ms_sample64"] = measure_latency(lambda _: model.sample(64, device), x64)

    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "ckpt", f"{tag}.pt"))
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return model, res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ae", "sparse_ae", "vae"], required=True)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--data_root", default="./data")
    a = p.parse_args()
    set_seed(SEED)
    run(a.model, get_loaders(a.data_root), a.latent, a.epochs, a.lr, a.lam)
