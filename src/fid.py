"""FID (Frechet Inception Distance) wrapper around torchmetrics.

NOTE: needs `torchmetrics[image]` + `torch-fidelity`; the Inception weights are downloaded on first use,
so this only works with internet access (e.g. Google Colab).
Inputs: float tensors in [0, 1], shape (N, 1, 28, 28). Grayscale is repeated to 3 channels.
Use the SAME number of real/fake images for every model, otherwise FID is not comparable.
"""
import torch


def compute_fid(real: torch.Tensor, fake: torch.Tensor, device, batch: int = 200) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for x in real.split(batch):
        fid.update(x.clamp(0, 1).repeat(1, 3, 1, 1).to(device), real=True)
    for x in fake.split(batch):
        fid.update(x.clamp(0, 1).repeat(1, 3, 1, 1).to(device), real=False)
    return float(fid.compute().item())


def collect_images(loader, n: int) -> torch.Tensor:
    xs, tot = [], 0
    for x, _ in loader:
        xs.append(x)
        tot += x.size(0)
        if tot >= n:
            break
    return torch.cat(xs)[:n]
