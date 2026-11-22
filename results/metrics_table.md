## Tabel IV. Perbandingan metrik pada Test Set

| Model | Dim Laten | Train time/epoch (s) | Latency (ms/batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID (N=2000) | Stabilitas |
|---|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 32 | 6.33 | 0.152 | 0.00858 | 21.75 | 0.7348 | N/A (Latent Voids) | Konvergen |
| Sparse Autoencoder | 64 | 6.49 | 0.162 | 0.00961 | 21.15 | 0.7127 | N/A (Latent Voids) | Konvergen |
| Variational Autoencoder (VAE) | 32 | 6.74 | 0.189 | 0.01330 | 19.60 | 0.6464 | 93.41 | Konvergen |
| GAN (MLP) | 100 | 8.46 | 0.236 | N/A | N/A | N/A | 112.69 | Stabil (dip diversitas hanya di epoch awal) |
| DCGAN | 100 | 11.79 | 0.656 | N/A | N/A | N/A | 31.71 | Stabil |
| DDPM | 28x28 (image) | 54.31 | 18534.5 | N/A | N/A | N/A | 18.26 | Stabil |

Catatan: latensi AE/VAE = rekonstruksi 64 citra; GAN/DCGAN = generator pada 64 noise; DDPM = sampling penuh T langkah untuk 64 citra. FID pada citra sintetis (prior sampling).

## Sparse AE: pengaruh penalti λ (Q1)

| Tag | Latent | λ | MSE | PSNR | SSIM | Fraksi unit laten ≈ 0 |
|---|---|---|---|---|---|---|
| sparse_ae_z64_lam1e-05 | 64 | 1e-05 | 0.00821 | 21.91 | 0.7440 | 0.371 |
| sparse_ae_z64_lam0.0001 | 64 | 0.0001 | 0.00961 | 21.15 | 0.7127 | 0.556 |
| sparse_ae_z64_lam0.001 | 64 | 0.001 | 0.08749 | 10.91 | 0.1341 | 1.000 |

## Eksperimen pembuktian batasan MLP Classifier (Q5)

Akurasi test classifier: 0.8850 (target ≥ 0,88: tercapai)

| Skenario | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| Tahap 3: Wᵀ pada z_MLP | 0.28057 | 5.57 | 0.0725 |
| Tahap 4: decoder pada z_MLP beku | 0.01816 | 18.13 | 0.5838 |
| Tahap 4: Vanilla AE (z=64) | 0.00757 | 22.26 | 0.7627 |

Tahap 2 (kelas → piksel): korelasi dengan rata-rata kelas yang cocok 0.231 vs tidak cocok 0.053; MSE terhadap rata-rata kelas 0.1036.

| Diagnostik | Nilai |
|---|---|
| ‖W1ᵀW1 − I‖_F / ‖I‖_F | 11.952 (cond=36.6) |
| ‖W2ᵀW2 − I‖_F / ‖I‖_F | 2.967 (cond=15.3) |
| ‖W3ᵀW3 − I‖_F / ‖I‖_F | 1.279 (cond=3.2) |
| Fraksi nol ReLU pada h1 | 0.693 |
| Within/between-class var: z_MLP vs z_AE | 0.456 vs 1.185 |
| Effective rank: z_MLP vs z_AE | 8.6 vs 17.2 |
| Unit mati z_MLP | 0.156 |
| Sampel Gaussian z_MLP di luar support (≥1 koordinat negatif) | 1.000 (koordinat negatif: 0.176; z_MLP asli bernilai tepat 0: 0.482) |
| FID sampling Gaussian: z_MLP vs z_AE (N=2000) | 146.4 vs 195.5 |
