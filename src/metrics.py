"""Reconstruction metrics. Inputs are tensors in [0, 1], shape (B, C, H, W)."""
import torch
import torch.nn.functional as F


def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(1)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = mse_per_image(x, y).clamp_min(1e-10)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 1, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.repeat(channels, 1, 1, 1)


def ssim_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Gaussian-window SSIM (Wang et al. 2004), 11x11, sigma=1.5, valid padding."""
    c = x.shape[1]
    w = _gaussian_window(11, 1.5, c, x.device)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


@torch.no_grad()
def evaluate_reconstruction(recon_fn, loader, device) -> dict:
    """Average MSE / PSNR / SSIM of recon_fn(x) vs x over a loader."""
    tot = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        xh = recon_fn(x).clamp(0, 1)
        b = x.size(0)
        tot["mse"] += mse_per_image(x, xh).sum().item()
        tot["psnr"] += psnr_per_image(x, xh).sum().item()
        tot["ssim"] += ssim_per_image(x, xh).sum().item()
        n += b
    return {k: v / n for k, v in tot.items()}
