"""Generate N samples from every trained generative model and compute FID against N real test images.

Usage (after training everything):  python -m src.evaluate_fid --n 2000
Writes results/fid.json. Uses identical N for all models so FID values are comparable
(FID with small N is biased upward; report N in the paper).
"""
import argparse
import json
import os

import torch

from .common import SEED, get_device, get_loaders, set_seed
from .fid import collect_images, compute_fid
from .models.autoencoder import VAE
from .models.ddpm import Diffusion
from .models.gan import NZ, build_gan


@torch.no_grad()
def generate(name, ckpt_dir, n, device, batch=500):
    set_seed(SEED)
    if name == "vae":
        m = VAE(32).to(device)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, "vae_z32.pt"), map_location=device))
        m.eval()
        return torch.cat([m.sample(min(batch, n - s), device) for s in range(0, n, batch)]).cpu()
    if name in ("gan", "dcgan"):
        G, _ = build_gan(name)
        G.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{name}_G.pt"), map_location=device))
        G.to(device).eval()
        out = [G(torch.randn(min(batch, n - s), NZ, device=device)) for s in range(0, n, batch)]
        return ((torch.cat(out) + 1) / 2).cpu()               # [-1,1] -> [0,1]
    if name == "ddpm":
        ck = torch.load(os.path.join(ckpt_dir, "ddpm_ema.pt"), map_location=device)
        m = Diffusion(ck["T"], base=ck["base"]).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        return ((m.sample(n, device, batch=batch) + 1) / 2).cpu()
    raise ValueError(name)


def run(models, n, test_loader, device, out_dir="results", fid_fn=compute_fid):
    real = collect_images(test_loader, n)
    res = {"n": n, "seed": SEED, "fid": {}}
    for name in models:
        fake = generate(name, os.path.join(out_dir, "ckpt"), n, device)
        res["fid"][name] = fid_fn(real, fake, device)
        print(f"FID[{name}] = {res['fid'][name]:.2f} (N={n})")
    with open(os.path.join(out_dir, "fid.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--models", nargs="+", default=["vae", "gan", "dcgan", "ddpm"])
    p.add_argument("--data_root", default="./data")
    a = p.parse_args()
    run(a.models, a.n, get_loaders(a.data_root)[2], get_device())
