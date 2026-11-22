"""Build the comparative metric tables (Section IV) from results/*.json.

Usage: python -m src.make_table
Outputs results/metrics_table.md and results/metrics_table.csv (+ lambda sweep and MLP-experiment tables inside the .md)

Stability labels are HEURISTICS computed from logged curves (thresholds below); always check them against Figure 5:
  AE family : "Konvergen" if val-MSE changed < 5% over the last 5 epochs.
  GAN/DCGAN : "Indikasi mode collapse" if mean diversity ratio over the LAST 10 epochs < 0.80 (generated pairwise
              distance / real); else "Osilatif" if std(loss_G, last 10) > 0.25 * mean(loss_G, last 10); else "Stabil"
              (annotated if min ratio < 0.80 earlier in training).
  DDPM      : "Stabil" if mean val loss of last 5 epochs differs < 5% from the mean of the 5 epochs before.
  (GAN and DDPM rules were revised after seeing the first run; see paper, Bagian IV.)
"""
import csv
import glob
import json
import os


def _load(p):
    with open(p) as f:
        return json.load(f)


def _f(v, nd=4):
    return "N/A" if v is None else f"{v:.{nd}f}"


def stab_ae(h):
    t = h["val_rec_mse"][-5:]
    return "Konvergen" if len(t) < 2 or (max(t) - min(t)) / max(min(t), 1e-12) < 0.05 else "Belum konvergen"


def stab_gan(r):
    """Judged on the END of training; an early warm-up dip alone is not mode collapse (revised after first run:
    the original global-minimum rule flagged epoch-1 diversity 0.46 although it recovered to ~1.0 by epoch ~6)."""
    d = r["history"]["diversity"][-10:]
    tail = sum(d) / len(d) / r["real_diversity_ref"]
    if tail < 0.80:
        return "Indikasi mode collapse"
    t = r["history"]["loss_G"][-10:]
    m = sum(t) / len(t)
    if r["loss_G_std_last10"] > 0.25 * m:
        return "Osilatif"
    return "Stabil (dip diversitas hanya di epoch awal)" if r["min_diversity_ratio"] < 0.80 else "Stabil"


def stab_ddpm(h):
    """Mean val loss of the last 5 epochs vs the 5 before (revised after first run: the max-min rule over 5 epochs
    reacted to Monte-Carlo noise of the random-t validation estimate, not to non-convergence)."""
    v = h["val_loss"]
    if len(v) < 10:
        return "Belum cukup epoch"
    a, b = sum(v[-5:]) / 5, sum(v[-10:-5]) / 5
    return "Stabil" if abs(a - b) / b < 0.05 else "Belum stabil"


def run(res_dir="results", sparse_tag="sparse_ae_z64_lam0.0001"):
    fid = _load(f"{res_dir}/fid.json") if os.path.exists(f"{res_dir}/fid.json") else {"fid": {}, "n": None}
    rows = []

    def ae_row(name, r, key):
        rows.append([name, str(r["latent_dim"]), _f(r["train_time_per_epoch_s"], 2), _f(r["latency_ms_batch64"], 3),
                     _f(r["test_mse"], 5), _f(r["test_psnr"], 2), _f(r["test_ssim"], 4),
                     _f(fid["fid"].get(key), 2) if key in fid["fid"] else "N/A (Latent Voids)", stab_ae(r["history"])])

    ae_row("Vanilla Autoencoder", _load(f"{res_dir}/ae_z32.json"), "ae")
    ae_row("Sparse Autoencoder", _load(f"{res_dir}/{sparse_tag}.json"), "sparse_ae")
    vae = _load(f"{res_dir}/vae_z32.json")
    ae_row("Variational Autoencoder (VAE)", vae, "vae")
    for key, name in (("gan", "GAN (MLP)"), ("dcgan", "DCGAN")):
        r = _load(f"{res_dir}/{key}.json")
        rows.append([name, str(r["latent_dim"]), _f(r["train_time_per_epoch_s"], 2), _f(r["latency_ms_batch64"], 3),
                     "N/A", "N/A", "N/A", _f(fid["fid"].get(key), 2), stab_gan(r)])
    r = _load(f"{res_dir}/ddpm.json")
    rows.append(["DDPM", "28x28 (image)", _f(r["train_time_per_epoch_s"], 2), _f(r["latency_ms_batch64"], 1),
                 "N/A", "N/A", "N/A", _f(fid["fid"].get("ddpm"), 2), stab_ddpm(r["history"])])

    head = ["Model", "Dim Laten", "Train time/epoch (s)", "Latency (ms/batch 64)", "Test MSE", "Test PSNR (dB)",
            "Test SSIM", f"FID (N={fid['n']})", "Stabilitas"]
    md = ["## Tabel IV. Perbandingan metrik pada Test Set", "", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)] + ["| " + " | ".join(x) + " |" for x in rows]
    md += ["", "Catatan: latensi AE/VAE = rekonstruksi 64 citra; GAN/DCGAN = generator pada 64 noise; "
           "DDPM = sampling penuh T langkah untuk 64 citra. FID pada citra sintetis (prior sampling).", ""]

    # ---- lambda sweep (Q1)
    sw = sorted(glob.glob(f"{res_dir}/sparse_ae_z*_lam*.json"), key=lambda p: (_load(p)["latent_dim"], _load(p)["lambda"]))
    if sw:
        md += ["## Sparse AE: pengaruh penalti λ (Q1)", "", "| Tag | Latent | λ | MSE | PSNR | SSIM | Fraksi unit laten ≈ 0 |",
               "|---|---|---|---|---|---|---|"]
        for p in sw:
            s = _load(p)
            md.append(f"| {s['tag']} | {s['latent_dim']} | {s['lambda']:g} | {_f(s['test_mse'], 5)} | {_f(s['test_psnr'], 2)} | "
                      f"{_f(s['test_ssim'], 4)} | {_f(s.get('latent_sparsity'), 3)} |")
        md.append("")

    # ---- MLP experiment (Q5)
    if os.path.exists(f"{res_dir}/mlp_experiment.json"):
        m = _load(f"{res_dir}/mlp_experiment.json")
        md += ["## Eksperimen pembuktian batasan MLP Classifier (Q5)", "",
               f"Akurasi test classifier: {m['step1']['test_acc']:.4f} (target ≥ 0,88: {'tercapai' if m['step1']['reached_88pct'] else 'TIDAK tercapai'})", "",
               "| Skenario | MSE | PSNR (dB) | SSIM |", "|---|---|---|---|"]
        for lab, k in (("Tahap 3: Wᵀ pada z_MLP", "step3"), ("Tahap 4: decoder pada z_MLP beku", "step4_mlp_frozen"),
                       ("Tahap 4: Vanilla AE (z=64)", "step4_vanilla_ae")):
            md.append(f"| {lab} | {_f(m[k]['mse'], 5)} | {_f(m[k]['psnr'], 2)} | {_f(m[k]['ssim'], 4)} |")
        s2 = m["step2"]
        md += ["", f"Tahap 2 (kelas → piksel): korelasi dengan rata-rata kelas yang cocok {s2['corr_matched_mean']:.3f} vs tidak cocok "
               f"{s2['corr_mismatched_mean']:.3f}; MSE terhadap rata-rata kelas {s2['mse_vs_class_mean']:.4f}.", ""]
        d = m["diagnostics"]
        md += ["| Diagnostik | Nilai |", "|---|---|"]
        for w in ("W1", "W2", "W3"):
            md.append(f"| ‖{w}ᵀ{w} − I‖_F / ‖I‖_F | {d[w]['WtW']:.3f} (cond={d[w]['cond']:.1f}) |")
        md += [f"| Fraksi nol ReLU pada h1 | {d['relu_zero_fraction_h1']:.3f} |",
               f"| Within/between-class var: z_MLP vs z_AE | {d['z_mlp']['within_over_between']:.3f} vs {d['z_ae']['within_over_between']:.3f} |",
               f"| Effective rank: z_MLP vs z_AE | {d['z_mlp']['effective_rank']:.1f} vs {d['z_ae']['effective_rank']:.1f} |",
               f"| Unit mati z_MLP | {d['z_mlp']['dead_units']:.3f} |"]
        if "step5_support" in m:
            sp = m["step5_support"]
            md.append(f"| Sampel Gaussian z_MLP di luar support (≥1 koordinat negatif) | {sp['frac_samples_with_negative_coord']:.3f} "
                      f"(koordinat negatif: {sp['frac_negative_coords']:.3f}; z_MLP asli bernilai tepat 0: {sp['real_zmlp_frac_exact_zero']:.3f}) |")
        if "step5_fid" in m:
            f5 = m["step5_fid"]
            md.append(f"| FID sampling Gaussian: z_MLP vs z_AE (N={f5['n']}) | {f5['mlp_gaussian_sampling']:.1f} vs {f5['ae_gaussian_sampling']:.1f} |")
        md.append("")

    with open(f"{res_dir}/metrics_table.md", "w") as f:
        f.write("\n".join(md))
    with open(f"{res_dir}/metrics_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    print("\n".join(md))
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sparse_tag", default="sparse_ae_z64_lam0.0001")
    run(sparse_tag=p.parse_args().sparse_tag)
