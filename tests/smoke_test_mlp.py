"""Offline smoke test for the MLP experiment + FID plumbing, RANDOM tensors only.
FID itself (Inception download) is replaced by a dummy function here -> NOT a real FID.
Run:  python -m tests.smoke_test_mlp
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate_fid
from src.mlp_experiment import class_to_pixels, latent_transpose, MLPClassifier, run
from src.train_ae import run as run_ae
from src.train_gan import run as run_gan
from src.train_ddpm import run as run_ddpm

torch.manual_seed(0)
dev = torch.device("cpu")
x, y = torch.rand(300, 1, 28, 28), torch.randint(0, 10, (300,))
dl = DataLoader(TensorDataset(x, y), batch_size=64)
L = (dl, dl, dl)

m = MLPClassifier()
assert class_to_pixels(m).shape == (10, 784)
assert latent_transpose(m, m.features(x[:5])).shape == (5, 1, 28, 28)

dummy_fid = lambda real, fake, device: float((real.mean() - fake.mean()).abs())  # placeholder, NOT FID
with tempfile.TemporaryDirectory() as d:
    r = run(L, out_dir=d, device=dev, max_epochs=1, dec_epochs=1, ae_epochs=1, n_fid=64, fid_fn=dummy_fid)
    assert {"step1", "step2", "step3", "step4_mlp_frozen", "step4_vanilla_ae", "step5_fid", "diagnostics"} <= r.keys()
    f3 = torch.load(f"{d}/ckpt/fig3_data.pt")
    assert f3["gt"].shape == (10, 1, 28, 28) and f3["sampling_mlp"].shape == (64, 1, 28, 28)
    print("diag keys ok; W1 ||WtW-I||:", round(r["diagnostics"]["W1"]["WtW"], 3))

with tempfile.TemporaryDirectory() as d:
    run_ae("vae", L, latent=32, epochs=1, out_dir=d, device=dev)
    run_gan("gan", L, epochs=1, out_dir=d, device=dev)
    run_gan("dcgan", L, epochs=1, out_dir=d, device=dev)
    run_ddpm(L, epochs=1, T=10, base=16, out_dir=d, device=dev)
    out = evaluate_fid.run(["vae", "gan", "dcgan", "ddpm"], 32, dl, dev, out_dir=d, fid_fn=dummy_fid)
    assert set(out["fid"]) == {"vae", "gan", "dcgan", "ddpm"}
print("MLP/FID SMOKE TEST PASSED")
