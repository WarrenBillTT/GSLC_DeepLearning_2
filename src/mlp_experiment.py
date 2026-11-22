"""Section 2.1 of the assignment: empirical proof that a discriminative MLP classifier cannot generate images.

Step 1  Train MLP classifier 784-256-64-10 with cross-entropy (target: test acc >= 88%); keep W1,W2,W3 and z_MLP (64-d).
Step 2  Top-down weight transposition  x^ = Sigmoid(W1^T ReLU(W2^T ReLU(W3^T y_c)))       (class one-hot -> pixels)
Step 3  Latent transpose projection    x^ = Sigmoid(W1^T ReLU(W2^T z_MLP))                (test image -> pixels)
Step 4  Frozen z_MLP + trained decoder (64-256-784, MSE)  vs  Vanilla AE latent z_AE (64-d)
Step 5  Generative sampling  z ~ N(mu_z, diag(sigma_z^2)) fitted on z_MLP, decoded with the Step-4 decoder
Plus diagnostics for Q5: ||W^T W - I||_F, ReLU information loss, within/between-class variance, effective rank.

Usage: python -m src.mlp_experiment [--fid]
Writes results/mlp_experiment.json and results/ckpt/fig3_data.pt (tensors for Figure 3).
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SEED, get_device, get_loaders, set_seed
from .metrics import evaluate_reconstruction
from .models.autoencoder import Autoencoder, Decoder
from .train_ae import run as run_ae


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, 10)

    def features(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x.flatten(1)))))  # z_MLP in R^64

    def forward(self, x):
        return self.fc3(self.features(x))


@torch.no_grad()
def accuracy(model, loader, device):
    c = n = 0
    for x, y in loader:
        c += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        n += y.numel()
    return c / n


def train_classifier(model, loaders, device, max_epochs=20, target_val=0.89, lr=1e-3):
    tr, va, _ = loaders
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(max_epochs):
        model.train()
        for x, y in tr:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        hist.append(accuracy(model, va, device))
        print(f"[mlp-cls] ep {ep + 1} val_acc={hist[-1]:.4f}")
        if hist[-1] >= target_val:
            break
    return hist


# --------------------------------------------------------------------- transposition maps (biases ignored on purpose)
def class_to_pixels(model):
    W1, W2, W3 = model.fc1.weight, model.fc2.weight, model.fc3.weight       # (256,784) (64,256) (10,64)
    Y = torch.eye(10, device=W1.device)
    h2 = torch.relu(Y @ W3)          # rows: ReLU(W3^T y_c)   (10,64)
    h1 = torch.relu(h2 @ W2)         # ReLU(W2^T h2)          (10,256)
    return torch.sigmoid(h1 @ W1)    # Sigmoid(W1^T h1)       (10,784)


def latent_transpose(model, z):
    return torch.sigmoid(torch.relu(z @ model.fc2.weight) @ model.fc1.weight).view(-1, 1, 28, 28)


# --------------------------------------------------------------------- diagnostics
def dev_from_identity(W):
    """Relative Frobenius deviation of W^T W (and W W^T) from I. 0 would mean orthonormal columns/rows."""
    out = {}
    for name, M in (("WtW", W.T @ W), ("WWt", W @ W.T)):
        I = torch.eye(M.size(0), device=W.device)
        out[name] = ((M - I).norm() / I.norm()).item()
    s = torch.linalg.svdvals(W)
    out["sv_max"], out["sv_min"] = s.max().item(), s.min().item()
    out["cond"] = (s.max() / s.min().clamp_min(1e-12)).item()
    return out


@torch.no_grad()
def collect(fn, loader, device):
    Z, Y = [], []
    for x, y in loader:
        Z.append(fn(x.to(device)).cpu()); Y.append(y)
    return torch.cat(Z), torch.cat(Y)


def latent_stats(Z, Y):
    mu = Z.mean(0)
    cm = torch.stack([Z[Y == c].mean(0) for c in range(10)])
    within = torch.stack([((Z[Y == c] - cm[c]) ** 2).sum(1).mean() for c in range(10)]).mean().item()
    between = ((cm - mu) ** 2).sum(1).mean().item()
    s = torch.linalg.svdvals(Z - mu)
    p = s ** 2 / (s ** 2).sum()
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    return {"within_class_var": within, "between_class_var": between,
            "within_over_between": within / between, "effective_rank": eff_rank,
            "dead_units": (Z.abs().max(0).values < 1e-8).float().mean().item(),
            "zero_fraction": (Z.abs() < 1e-8).float().mean().item()}


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


# --------------------------------------------------------------------- main experiment
def run(loaders, out_dir="results", device=None, max_epochs=20, dec_epochs=15, ae_epochs=15,
        n_fid=2000, fid_fn=None):
    from .fid import collect_images

    set_seed(SEED)
    device = device or get_device()
    tr, va, te = loaders
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    res = {"seed": SEED}

    # ---- Step 1
    clf = MLPClassifier().to(device)
    hist = train_classifier(clf, loaders, device, max_epochs)
    for p in clf.parameters():
        p.requires_grad_(False)
    clf.eval()
    res["step1"] = {"val_acc_history": hist, "test_acc": accuracy(clf, te, device), "reached_88pct": None}
    res["step1"]["reached_88pct"] = res["step1"]["test_acc"] >= 0.88
    torch.save(clf.state_dict(), os.path.join(out_dir, "ckpt", "mlp_cls.pt"))

    # ---- Step 2 (class -> pixels). Compare against the real class-mean image.
    templ = class_to_pixels(clf).detach()
    xs, ys = collect(lambda x: x, te, device)
    class_mean = torch.stack([xs[ys == c].mean(0).flatten() for c in range(10)]).to(device)
    corr = torch.tensor([[pearson(templ[a], class_mean[b]) for b in range(10)] for a in range(10)])
    res["step2"] = {"corr_matched_mean": corr.diag().mean().item(),
                    "corr_mismatched_mean": ((corr.sum() - corr.diag().sum()) / 90).item(),
                    "pixel_std_of_templates": templ.std().item(),
                    "mse_vs_class_mean": ((templ - class_mean) ** 2).mean().item()}

    # ---- Step 3 (z_MLP -> pixels through W^T only)
    res["step3"] = evaluate_reconstruction(lambda x: latent_transpose(clf, clf.features(x)), te, device)

    # ---- Step 4 (frozen z_MLP + trained decoder) vs Vanilla AE (z=64)
    dec = Decoder(64, 256).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    for ep in range(dec_epochs):
        dec.train()
        for x, _ in tr:
            x = x.to(device)
            with torch.no_grad():
                z = clf.features(x)
            loss = F.mse_loss(dec(z), x)
            opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    torch.save(dec.state_dict(), os.path.join(out_dir, "ckpt", "mlp_frozen_decoder.pt"))
    res["step4_mlp_frozen"] = evaluate_reconstruction(lambda x: dec(clf.features(x)), te, device)

    ae_ckpt = os.path.join(out_dir, "ckpt", "ae_z64.pt")
    if os.path.exists(ae_ckpt):
        ae = Autoencoder(64).to(device)
        ae.load_state_dict(torch.load(ae_ckpt, map_location=device))
    else:
        ae, _ = run_ae("ae", loaders, latent=64, epochs=ae_epochs, out_dir=out_dir, tag="ae_z64", device=device)
    ae.eval()
    res["step4_vanilla_ae"] = evaluate_reconstruction(lambda x: ae(x)[0], te, device)

    # ---- Step 5 (Gaussian sampling in latent space, decoded)
    Z_mlp, Y_te = collect(clf.features, te, device)
    Z_ae, _ = collect(ae.encoder, te, device)
    Ztr_mlp, _ = collect(clf.features, tr, device)
    Ztr_ae, _ = collect(ae.encoder, tr, device)

    def sample(Ztr, decoder, n):
        mu, sd = Ztr.mean(0).to(device), Ztr.std(0).to(device)
        return decoder(mu + sd * torch.randn(n, Ztr.size(1), device=device)).detach()

    set_seed(SEED)
    s_mlp, s_ae = sample(Ztr_mlp, dec, 64), sample(Ztr_ae, ae.decoder, 64)
    # z_MLP = ReLU(.) lives in the non-negative orthant, but N(mu, diag sigma^2) puts mass on negative coordinates:
    # measure how many sampled latents fall outside the support of the real z_MLP.
    z_g = Ztr_mlp.mean(0) + Ztr_mlp.std(0) * torch.randn(5000, Ztr_mlp.size(1))
    res["step5_support"] = {"frac_samples_with_negative_coord": (z_g < 0).any(1).float().mean().item(),
                            "frac_negative_coords": (z_g < 0).float().mean().item(),
                            "real_zmlp_frac_exact_zero": (Ztr_mlp == 0).float().mean().item()}
    if fid_fn is not None:
        real = collect_images(te, n_fid)
        res["step5_fid"] = {"mlp_gaussian_sampling": fid_fn(real, sample(Ztr_mlp, dec, n_fid).cpu(), device),
                            "ae_gaussian_sampling": fid_fn(real, sample(Ztr_ae, ae.decoder, n_fid).cpu(), device),
                            "n": n_fid}

    # ---- Q5 diagnostics
    with torch.no_grad():
        x = xs[:2000].to(device)
        h1 = torch.relu(clf.fc1(x.flatten(1)))
    res["diagnostics"] = {
        "W1": dev_from_identity(clf.fc1.weight), "W2": dev_from_identity(clf.fc2.weight),
        "W3": dev_from_identity(clf.fc3.weight),
        "relu_zero_fraction_h1": (h1 == 0).float().mean().item(),
        "z_mlp": latent_stats(Z_mlp, Y_te), "z_ae": latent_stats(Z_ae, Y_te),
    }

    # ---- tensors for Figure 3
    idx = [int((Y_te == c).nonzero()[0]) for c in range(10)]
    xg = xs[idx].to(device)
    with torch.no_grad():
        z = clf.features(xg)
        fig3 = {"labels": torch.arange(10), "gt": xg.cpu(),
                "class_templates": templ.view(-1, 1, 28, 28).cpu(),
                "latent_transpose": latent_transpose(clf, z).cpu(),
                "frozen_decoder": dec(z).cpu(), "ae_recon": ae(xg)[0].cpu(),
                "sampling_mlp": s_mlp.cpu(), "sampling_ae": s_ae.cpu()}
    torch.save(fig3, os.path.join(out_dir, "ckpt", "fig3_data.pt"))
    with open(os.path.join(out_dir, "mlp_experiment.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--dec_epochs", type=int, default=15)
    p.add_argument("--ae_epochs", type=int, default=15)
    p.add_argument("--fid", action="store_true", help="also compute FID of Step-5 samples (needs internet)")
    a = p.parse_args()
    fid_fn = None
    if a.fid:
        from .fid import compute_fid
        fid_fn = compute_fid
    r = run(get_loaders(a.data_root), dec_epochs=a.dec_epochs, ae_epochs=a.ae_epochs, fid_fn=fid_fn)
    print(json.dumps({k: v for k, v in r.items() if k != "diagnostics"}, indent=2))
