"""
Interview-style learned image compression (end-to-end) using PyTorch.
This is a minimal, illustrative pipeline (not a full paper implementation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_ch, 5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_ch=64, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 128, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_ch, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, y):
        return self.net(y)


class EntropyModel(nn.Module):
    """
    Simplified factorized Gaussian likelihood model.
    Returns per-latent probability (likelihood).
    """

    def __init__(self, latent_ch=64):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(latent_ch, 1, 1))

    def forward(self, y_q):
        scale = torch.exp(self.log_scale) + 1e-6
        # Gaussian PDF for each latent element (factorized).
        likelihood = (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi)) * scale)) * torch.exp(
            -0.5 * (y_q / scale) ** 2
        )
        return torch.clamp(likelihood, min=1e-9)
class LearnedCompressionModel(nn.Module):
    def __init__(self, in_ch=3, latent_ch=64):
        super().__init__()
        self.g_a = Encoder(in_ch, latent_ch)
        self.g_s = Decoder(latent_ch, in_ch)
        self.entropy_model = EntropyModel(latent_ch)
    def forward(self, x):
        y = self.g_a(x)
        if self.training:
            y_q = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_q = torch.round(y)
        x_hat = self.g_s(y_q)
        likelihood = self.entropy_model(y_q)
        return x_hat, y_q, likelihood
def rd_loss(x, x_hat, likelihood, lam=0.01):
    bpp = (-torch.log2(likelihood)).mean()
    dist = torch.mean((x - x_hat) ** 2)
    return bpp + lam * dist, bpp, dist
def train_step(model, x, optimizer, lam=0.01):
    model.train()
    x_hat, y_q, likelihood = model(x)
    loss, bpp, dist = rd_loss(x, x_hat, likelihood, lam=lam)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "bpp": float(bpp.item()), "dist": float(dist.item())}
@torch.no_grad()
def inference(model, x):
    model.eval()
    x_hat, y_q, likelihood = model(x)
    bpp = (-torch.log2(likelihood)).mean()
    return x_hat, bpp
if __name__ == "__main__":
    B, H, W = 2, 128, 128
    x = torch.rand(B, 3, H, W)
    model = LearnedCompressionModel(in_ch=3, latent_ch=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    stats = train_step(model, x, optimizer, lam=0.01)
    print("train:", stats)
    x_hat, bpp = inference(model, x)
    print("inference bpp:", float(bpp.item()))
