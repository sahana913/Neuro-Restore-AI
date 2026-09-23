"""
Script: src/evaluation/evaluate_vaegan.py
Purpose: Evaluates the trained VAE-GAN restoration model:
         1. Computes quantitative metrics: PSNR, SSIM, and Mean Absolute Error (MAE).
         2. Generates a visual 4-panel comparison plot:
            [Clean Original] | [Noisy Input] | [VAE Restored] | [Noise Removed]
         3. Saves the plot to 'outputs/vaegan_evaluation.png'.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Residual U-VAE Architecture (Exact match to trained model)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

class VAE_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2)) # 256x256
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))   # 128x128
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)) # 64x64
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))# 32x32

        # Latent spatial bottleneck
        self.res = ResBlock(256)
        self.conv_mu = nn.Conv2d(256, 32, 1)
        self.conv_logvar = nn.Conv2d(256, 32, 1)
        self.conv_z = nn.Conv2d(32, 256, 1)

        # Decoder with U-Net Skip Connections
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU()) # 64x64
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())   # 128x128
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())    # 256x256
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.out_delta = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Tanh())

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        feat = self.res(e4)
        mu = self.conv_mu(feat)
        logvar = self.conv_logvar(feat)
        return mu, logvar, (e1, e2, e3)

    def decode(self, z, skips):
        e1, e2, e3 = skips
        d = self.conv_z(z)
        d3 = self.up3(d)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        delta = self.out_delta(d1) * 0.5
        return delta

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        delta = self.decode(mu, skips) # Use mean directly for evaluation
        restored = torch.clamp(x + delta, 0.0, 1.0)
        return restored

# 2. Metric Calculations
def calculate_psnr(target, pred):
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Lightweight SSIM computation
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)

def evaluate_model(weights_path="models/vaegan/vaegan_best.pth", data_dir="data/dummy"):
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'!")
        return

    # Load model
    print(f"Loading trained weights from: {weights_path}...")
    model = VAE_Generator()
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

    # Find evaluation slice
    sample_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not sample_files:
        print(f"No sample files found in '{data_dir}'.")
        return

    sample = np.load(sample_files[0])
    clean = sample['dwi'] if 'dwi' in sample else sample['image']
    clean = clean.astype(np.float32)

    # Simulate realistic scanner noise
    noise = np.random.normal(0, 0.08, clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)

    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0) # (1, 1, 256, 256)
        restored_tensor = model(input_tensor)
        restored = restored_tensor.squeeze().numpy()

    # Calculate metrics
    noisy_psnr = calculate_psnr(clean, noisy)
    restored_psnr = calculate_psnr(clean, restored)
    
    noisy_ssim = calculate_ssim(clean, noisy)
    restored_ssim = calculate_ssim(clean, restored)
    
    mae = np.mean(np.abs(clean - restored))

    print("\n" + "="*50)
    print("      [EVALUATION] VAE-GAN RESTORATION ACCURACY REPORT")
    print("="*50)
    print(f"1. PSNR (Clarity in dB):")
    print(f"   * Noisy Scan:    {noisy_psnr:.2f} dB")
    print(f"   * Restored Scan: {restored_psnr:.2f} dB (Improvement: +{restored_psnr - noisy_psnr:.2f} dB)")
    print(f"2. SSIM (Structural Fidelity):")
    print(f"   * Noisy Scan:    {noisy_ssim:.4f}")
    print(f"   * Restored Scan: {restored_ssim:.4f} (Closer to 1.0 is better)")
    print(f"3. Mean Pixel Error (MAE): {mae:.4f}")
    print("="*50)

    # Plot visual proof
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/vaegan_evaluation.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(clean, cmap='gray')
    axes[0].set_title("Ground Truth (Clean)")

    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title(f"Noisy Scan (PSNR: {noisy_psnr:.1f} dB)")

    axes[2].imshow(restored, cmap='gray')
    axes[2].set_title(f"Restored VAE (PSNR: {restored_psnr:.1f} dB)")

    # Noise removed map (|Noisy - Restored|)
    noise_removed = np.abs(noisy - restored)
    axes[3].imshow(noise_removed, cmap='hot')
    axes[3].set_title("Noise Removed Map")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Visual evaluation plot saved to: '{out_path}'")

if __name__ == "__main__":
    evaluate_model()
