"""Offline smoke test of make_figures / make_table on a TINY pipeline trained on RANDOM tensors.
Only checks that the scripts run end-to-end and write files; the images/numbers are meaningless.
Run:  python -m tests.smoke_test_figs
"""
import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate_fid, make_figures, make_table
from src.mlp_experiment import run as run_mlp
from src.train_ae import run as run_ae
from src.train_ddpm import run as run_ddpm
from src.train_gan import run as run_gan

torch.manual_seed(0)
dev = torch.device("cpu")
x, y = torch.rand(400, 1, 28, 28), torch.arange(400) % 10
dl = DataLoader(TensorDataset(x, y), batch_size=64)
L = (dl, dl, dl)
dummy_fid = lambda real, fake, device: 1.0  # placeholder

with tempfile.TemporaryDirectory() as d:
    res, figs = f"{d}/results", f"{d}/figures"
    run_ae("ae", L, latent=32, epochs=2, out_dir=res, device=dev)
    run_ae("sparse_ae", L, latent=64, epochs=2, lam=1e-4, out_dir=res, device=dev)
    run_ae("sparse_ae", L, latent=64, epochs=2, lam=1e-3, out_dir=res, device=dev)
    run_ae("vae", L, latent=32, epochs=2, out_dir=res, device=dev)
    run_gan("gan", L, epochs=2, out_dir=res, device=dev)
    run_gan("dcgan", L, epochs=2, out_dir=res, device=dev)
    run_ddpm(L, epochs=2, T=20, base=16, out_dir=res, device=dev)
    run_mlp(L, out_dir=res, device=dev, max_epochs=1, dec_epochs=1, ae_epochs=1, n_fid=64, fid_fn=dummy_fid)
    evaluate_fid.run(["vae", "gan", "dcgan", "ddpm"], 32, dl, dev, out_dir=res, fid_fn=dummy_fid)
    make_figures.run(res, figs, dl, dev)
    rows = make_table.run(res)
    for f in ["fig1_reconstruction", "fig2_random_sampling", "fig3_mlp_limits", "fig4_interpolation",
              "fig5_curves_ddpm_trajectory"]:
        assert os.path.getsize(f"{figs}/{f}.png") > 5000, f
    assert len(rows) == 6 and os.path.exists(f"{res}/interp_smoothness.json")
    print(open(f"{res}/interp_smoothness.json").read()[:300])
print("FIGS/TABLE SMOKE TEST PASSED")
