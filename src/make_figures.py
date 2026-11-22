"""Generate the five mandatory figures of Section V from trained checkpoints / result files.

Usage (after all training scripts):  python -m src.make_figures
Outputs figures/fig1..fig5 (.png, 200 dpi) and results/interp_smoothness.json (quantitative backing for Q4).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from .common import SEED, get_device, get_loaders, set_seed
from .evaluate_fid import generate
from .mlp_experiment import MLPClassifier
from .models.autoencoder import VAE, Autoencoder, SparseAutoencoder
from .models.gan import NZ, build_gan

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def grid_image(imgs: torch.Tensor, nrow: int, ncol: int, pad: int = 2) -> np.ndarray:
    imgs = imgs.detach().cpu().clamp(0, 1)[: nrow * ncol, 0].numpy()
    canvas = np.ones((nrow * 28 + (nrow - 1) * pad, ncol * 28 + (ncol - 1) * pad))
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncol)
        canvas[r * (28 + pad): r * (28 + pad) + 28, c * (28 + pad): c * (28 + pad) + 28] = im
    return canvas


def show(ax, imgs, nrow, ncol, title=None, ylabel=None):
    ax.imshow(grid_image(imgs, nrow, ncol), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", fontsize=9)


def _load(path):
    with open(path) as f:
        return json.load(f)


def load_ae_family(ckpt, sparse_tag, device):
    ae = Autoencoder(32).to(device); ae.load_state_dict(torch.load(f"{ckpt}/ae_z32.pt", map_location=device))
    sp = SparseAutoencoder(64).to(device); sp.load_state_dict(torch.load(f"{ckpt}/{sparse_tag}.pt", map_location=device))
    vae = VAE(32).to(device); vae.load_state_dict(torch.load(f"{ckpt}/vae_z32.pt", map_location=device))
    return ae.eval(), sp.eval(), vae.eval()


def load_gans(ckpt, device):
    out = {}
    for k in ("gan", "dcgan"):
        G, _ = build_gan(k)
        G.load_state_dict(torch.load(f"{ckpt}/{k}_G.pt", map_location=device))
        out[k] = G.to(device).eval()
    return out


# ------------------------------------------------------------------------------------------ Figure 1
@torch.no_grad()
def fig1(models, x, path):
    ae, sp, vae = models
    rows = [("Ground truth", x), ("Vanilla AE", ae(x)[0]), ("Sparse AE", sp(x)[0]), ("VAE", vae.reconstruct(x))]
    fig, axes = plt.subplots(4, 1, figsize=(9, 4.6))
    for ax, (name, im) in zip(axes, rows):
        show(ax, im, 1, x.size(0), ylabel=name)
    fig.suptitle("Figure 1. Side-by-side reconstruction (Test Set)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------ Figure 2
@torch.no_grad()
def fig2(ckpt_dir, device, path):
    panels = []
    for key, title in [("vae", "A: VAE, z ~ N(0,I)"), ("gan", "B: Vanilla GAN, z ~ N(0,I)"),
                       ("dcgan", "C: DCGAN, z ~ N(0,I)"), ("ddpm", "D: DDPM, reverse denoising from x_T")]:
        if key == "ddpm":  # reuse the 64 EMA samples saved at training time (avoids a second 1000-step run)
            s = (torch.load(f"{ckpt_dir}/ddpm_traj.pt", map_location="cpu")["samples64"][:16] + 1) / 2
        else:
            s = generate(key, ckpt_dir, 16, device)
        panels.append((title, s))
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.4))
    for ax, (t, s) in zip(axes.ravel(), panels):
        show(ax, s, 4, 4, title=t)
    fig.suptitle("Figure 2. Random generative sampling (16 samples per model)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------ Figure 3
def fig3(ckpt_dir, path):
    d = torch.load(f"{ckpt_dir}/fig3_data.pt", map_location="cpu")
    rows = [("(1) Ground truth", d["gt"]),
            ("(2a) W$^T$ from class $y_c$", d["class_templates"]),
            ("(2b) W$^T$ from $z_{MLP}$", d["latent_transpose"]),
            ("(3a) Frozen $z_{MLP}$ + decoder", d["frozen_decoder"]),
            ("(3b) Vanilla AE ($z_{AE}$)", d["ae_recon"]),
            ("(4a) Prior sampling, $z_{MLP}$", d["sampling_mlp"][:10]),
            ("(4b) Prior sampling, $z_{AE}$", d["sampling_ae"][:10])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 7.4))
    for ax, (name, im) in zip(axes, rows):
        show(ax, im, 1, 10, ylabel=name)
    for c in range(10):  # class name above each column (column stride = 28 px image + 2 px pad)
        axes[0].text(c * 30 + 14, -4, f"{c}: {CLASSES[c]}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Figure 3. Empirical limits of MLP weights and latent features for image generation", fontsize=12)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.9, bottom=0.01, hspace=0.12)
    fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------ Figure 4
@torch.no_grad()
def pick_gan_pair(G, clf, a, b, device, n=500):
    """Pick two noise vectors whose generated images the MLP classifier labels as class a and class b."""
    set_seed(SEED)
    z = torch.randn(n, NZ, device=device)
    pred = clf(((G(z) + 1) / 2)).argmax(1) if clf is not None else torch.arange(n) % 10
    ia, ib = (pred == a).nonzero().flatten(), (pred == b).nonzero().flatten()
    if len(ia) == 0 or len(ib) == 0:
        return z[0], z[1], False
    return z[ia[0]], z[ib[0]], True


@torch.no_grad()
def fig4(models, gans, clf, test_loader, device, path, class_a, class_b, json_path):
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x); ys.append(y)
    xs, ys = torch.cat(xs), torch.cat(ys)
    xa = xs[(ys == class_a).nonzero()[0, 0]][None].to(device)
    xb = xs[(ys == class_b).nonzero()[0, 0]][None].to(device)
    ae, sp, vae = models
    alphas = torch.linspace(0, 1, 21, device=device)
    show_idx = [0, 4, 8, 12, 16, 20]  # alpha = 0, .2, .4, .6, .8, 1

    def lerp(za, zb):
        return (1 - alphas[:, None]) * za + alphas[:, None] * zb

    outs, notes = {}, {}
    outs["Vanilla AE"] = ae.decoder(lerp(ae.encoder(xa), ae.encoder(xb)))
    outs["Sparse AE"] = sp.decoder(lerp(sp.encoder(xa), sp.encoder(xb)))
    outs["VAE"] = vae.decoder(lerp(vae.encode(xa)[0], vae.encode(xb)[0]))
    for k, name in (("gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        za, zb, ok = pick_gan_pair(gans[k], clf, class_a, class_b, device)
        outs[name] = (gans[k](lerp(za[None], zb[None])) + 1) / 2
        notes[name] = ("endpoints chosen so the MLP classifier labels them as the two classes" if ok else
                       "no endpoint pair of the two classes found; arbitrary noise pair used")

    stats = {}
    for name, im in outs.items():
        step = (im[1:] - im[:-1]).flatten(1).norm(dim=1)
        stats[name] = {"mean_step": step.mean().item(), "max_over_mean_step": (step.max() / step.mean()).item(),
                       "cv_step": (step.std() / step.mean()).item(), "note": notes.get(name, "endpoints = encoded test images")}
    with open(json_path, "w") as f:
        json.dump({"classes": [CLASSES[class_a], CLASSES[class_b]], "n_alpha": 21, "stats": stats}, f, indent=2)

    fig, axes = plt.subplots(len(outs), 1, figsize=(7.2, 1.35 * len(outs) + 0.9))
    for ax, (name, im) in zip(axes, outs.items()):
        show(ax, im[show_idx], 1, 6, ylabel=name)
    axes[0].set_title("   ".join(f"$\\alpha$={a:.1f}" for a in alphas[show_idx].tolist()), fontsize=8)
    fig.suptitle(f"Figure 4. Linear latent interpolation ({CLASSES[class_a]} → {CLASSES[class_b]})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------ Figure 5
def fig5(res_dir, sparse_tag, path):
    ae, sp, vae = (_load(f"{res_dir}/{t}.json") for t in ("ae_z32", sparse_tag, "vae_z32"))
    gan, dc, dd = (_load(f"{res_dir}/{t}.json") for t in ("gan", "dcgan", "ddpm"))
    traj = torch.load(f"{res_dir}/ckpt/ddpm_traj.pt", map_location="cpu")["traj"]
    fig = plt.figure(figsize=(17, 11.5))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 1.05], figure=fig)

    def curve(ax, ys, labels, title, ylab=None, logy=False):
        for y, l in zip(ys, labels):
            ax.plot(range(1, len(y) + 1), y, label=l)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        if ylab: ax.set_ylabel(ylab)
        if logy: ax.set_yscale("log")

    curve(fig.add_subplot(gs[0, 0]), [ae["history"]["val_rec_mse"], sp["history"]["val_rec_mse"]], ["Vanilla AE", "Sparse AE"],
          "(a) AE family: validation reconstruction MSE", "MSE", True)
    curve(fig.add_subplot(gs[0, 1]), [vae["history"]["train_loss"]], ["VAE (BCE+KL, train)"], "(b) VAE negative ELBO")
    curve(fig.add_subplot(gs[0, 2]), [dd["history"]["train_loss"], dd["history"]["val_loss"]], ["DDPM train", "DDPM val (EMA)"],
          "(c) DDPM noise-prediction MSE", None, True)
    r1, r2 = gan["real_diversity_ref"], dc["real_diversity_ref"]
    curve(fig.add_subplot(gs[0, 3]), [[d / r1 for d in gan["history"]["diversity"]], [d / r2 for d in dc["history"]["diversity"]]],
          ["Vanilla GAN", "DCGAN"], "(d) Diversity ratio (low = mode collapse)")
    for j, (r, name) in enumerate(((gan, "Vanilla GAN"), (dc, "DCGAN"))):
        curve(fig.add_subplot(gs[1, j]), [r["history"]["loss_D"], r["history"]["loss_G"]], ["Discriminator", "Generator"],
              f"({'ef'[j]}) {name}: training loss")
        axd = fig.add_subplot(gs[1, 2 + j])
        curve(axd, [r["history"]["D_real"], r["history"]["D_fake"]], ["D(x)", "D(G(z))"],
              f"({'gh'[j]}) {name}: D outputs (left), G grad norm (right)")
        axd.set_ylim(0, 1)
        axd.legend(fontsize=8, loc="upper left")
        axg = axd.twinx()  # gradient norm lives on its own scale
        axg.plot(range(1, len(r["history"]["grad_G_first"]) + 1), r["history"]["grad_G_first"], color="tab:green",
                 linestyle="--", label="‖∇W₁‖ of G")
        axg.set_yscale("log"); axg.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[2, :])
    steps = sorted(traj.keys(), reverse=True)
    n_rows = min(3, traj[steps[0]].size(0))
    canvas = np.ones((n_rows * 30, len(steps) * 30))
    for c, t in enumerate(steps):
        for r in range(n_rows):
            canvas[r * 30: r * 30 + 28, c * 30: c * 30 + 28] = ((traj[t][r, 0] + 1) / 2).clamp(0, 1).numpy()
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([c * 30 + 14 for c in range(len(steps))]); ax.set_xticklabels([f"$x_{{{t}}}$" for t in steps])
    ax.set_yticks([]); ax.set_title("(i) DDPM reverse trajectory $x_T \\rightarrow x_0$ (3 samples)", fontsize=10)
    fig.suptitle("Figure 5. Training loss curves and DDPM denoising trajectory", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------ driver
def run(res_dir="results", fig_dir="figures", test_loader=None, device=None, sparse_tag="sparse_ae_z64_lam0.0001",
        class_a=1, class_b=8):
    set_seed(SEED)
    device = device or get_device()
    os.makedirs(fig_dir, exist_ok=True)
    ckpt = f"{res_dir}/ckpt"
    models = load_ae_family(ckpt, sparse_tag, device)
    gans = load_gans(ckpt, device)
    clf = None
    if os.path.exists(f"{ckpt}/mlp_cls.pt"):
        clf = MLPClassifier().to(device)
        clf.load_state_dict(torch.load(f"{ckpt}/mlp_cls.pt", map_location=device)); clf.eval()
    x8 = next(iter(test_loader))[0][:8].to(device)
    fig1(models, x8, f"{fig_dir}/fig1_reconstruction.png")
    fig2(ckpt, device, f"{fig_dir}/fig2_random_sampling.png")
    fig3(ckpt, f"{fig_dir}/fig3_mlp_limits.png")
    fig4(models, gans, (lambda x: clf(x)) if clf is not None else None, test_loader, device,
         f"{fig_dir}/fig4_interpolation.png", class_a, class_b, f"{res_dir}/interp_smoothness.json")
    fig5(res_dir, sparse_tag, f"{fig_dir}/fig5_curves_ddpm_trajectory.png")
    print("figures saved to", fig_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--sparse_tag", default="sparse_ae_z64_lam0.0001")
    p.add_argument("--class_a", type=int, default=1, help="Fashion-MNIST class id (0-9) of interpolation start")
    p.add_argument("--class_b", type=int, default=8, help="class id of interpolation end")
    a = p.parse_args()
    run(test_loader=get_loaders(a.data_root)[2], sparse_tag=a.sparse_tag, class_a=a.class_a, class_b=a.class_b)
