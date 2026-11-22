"""Standardized setup shared by ALL models: seed, device, data split.

Dataset : Fashion-MNIST (train 60k + test 10k merged = 70k)
Split   : 80% train / 10% val / 10% test, fixed generator seed (identical for every model)
Batch   : 128 (shuffle=True on train, shuffle=False on val/test)
Pixels  : [0, 1] (ToTensor). GAN/DCGAN/DDPM rescale to [-1, 1] internally.
"""
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(root: str = "./data", batch_size: int = BATCH_SIZE, seed: int = SEED):
    """Return (train_loader, val_loader, test_loader) with the 80/10/10 split."""
    tf = transforms.ToTensor()
    full = ConcatDataset([
        datasets.FashionMNIST(root, train=True, download=True, transform=tf),
        datasets.FashionMNIST(root, train=False, download=True, transform=tf),
    ])
    n = len(full)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    n_te = n - n_tr - n_va
    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(full, [n_tr, n_va, n_te], generator=g)
    g_loader = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size, shuffle=True, generator=g_loader),
        DataLoader(va, batch_size, shuffle=False),
        DataLoader(te, batch_size, shuffle=False),
    )


@torch.no_grad()
def measure_latency(fn, x: torch.Tensor, n_runs: int = 30, warmup: int = 5) -> float:
    """Mean wall-clock latency (ms) of fn(x) for one batch (default: 64 samples)."""
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    for _ in range(warmup):
        fn(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(x)
    sync()
    return (time.perf_counter() - t0) / n_runs * 1000.0
