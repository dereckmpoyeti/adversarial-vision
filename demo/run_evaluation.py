import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from models.target_model import load_model, load_labels, preprocess, predict
from attacks.fgsm import fgsm_attack, fgsm_sweep
from attacks.pgd  import pgd_attack
from evaluation.metrics import full_report, l2_distance, compute_ssim

IMAGE_PATH = "images.jpg"

labels        = load_labels()
model, device = load_model()
tensor        = preprocess(IMAGE_PATH)

clean_preds     = predict(model, tensor, labels, device, top_k=1)
true_label_name = clean_preds[0][0]
true_label_idx  = labels.index(true_label_name)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

EPSILON = 0.05

# ── Attaques ──────────────────────────────────────────────────────────────────
fgsm_adv, _, _ = fgsm_attack(model, tensor.clone(), true_label_idx, EPSILON, device)
pgd_adv, _     = pgd_attack (model, tensor.clone(), true_label_idx, EPSILON, device)

fgsm_pred = predict(model, fgsm_adv, labels, device, top_k=1)[0][0]
pgd_pred  = predict(model, pgd_adv,  labels, device, top_k=1)[0][0]

# ── Rapports individuels ───────────────────────────────────────────────────────
r_fgsm = full_report(model, tensor, fgsm_adv, true_label_name,
                     fgsm_pred, "FGSM", EPSILON, device)
r_pgd  = full_report(model, tensor, pgd_adv,  true_label_name,
                     pgd_pred,  "PGD",  EPSILON, device)

# ── Sweep epsilon : ASR + L2 + SSIM ──────────────────────────────────────────
epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
asr_fgsm, asr_pgd = [], []
l2_fgsm,  l2_pgd  = [], []
ssim_fgsm, ssim_pgd = [], []

print("\n[+] Sweep epsilon en cours...")
for eps in epsilons:
    if eps == 0.0:
        asr_fgsm.append(0); asr_pgd.append(0)
        l2_fgsm.append(0);  l2_pgd.append(0)
        ssim_fgsm.append(1.0); ssim_pgd.append(1.0)
        continue

    fa, _, _ = fgsm_attack(model, tensor.clone(), true_label_idx, eps, device)
    pa, _    = pgd_attack (model, tensor.clone(), true_label_idx, eps, device)

    fp = predict(model, fa, labels, device, top_k=1)[0][0]
    pp = predict(model, pa, labels, device, top_k=1)[0][0]

    asr_fgsm.append(1 if fp != true_label_name else 0)
    asr_pgd.append (1 if pp != true_label_name else 0)
    l2_fgsm.append (l2_distance(tensor, fa))
    l2_pgd.append  (l2_distance(tensor, pa))
    ssim_fgsm.append(compute_ssim(tensor, fa))
    ssim_pgd.append (compute_ssim(tensor, pa))

# ── Figure : 3 subplots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Évaluation des attaques adversariales — FGSM vs PGD", 
             fontsize=13, fontweight="bold")

# ASR
axes[0].step(epsilons, asr_fgsm, where="post", color="#e67e22", 
             linewidth=2, label="FGSM", marker="o")
axes[0].step(epsilons, asr_pgd,  where="post", color="#e74c3c", 
             linewidth=2, label="PGD",  marker="s")
axes[0].set_title("Attack Success Rate")
axes[0].set_xlabel("Epsilon")
axes[0].set_ylabel("Succès (0/1)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# L2
axes[1].plot(epsilons, l2_fgsm, color="#e67e22", linewidth=2, 
             label="FGSM", marker="o")
axes[1].plot(epsilons, l2_pgd,  color="#e74c3c", linewidth=2, 
             label="PGD",  marker="s")
axes[1].set_title("Perturbation L2")
axes[1].set_xlabel("Epsilon")
axes[1].set_ylabel("Norme L2")
axes[1].legend(); axes[1].grid(alpha=0.3)

# SSIM
axes[2].plot(epsilons, ssim_fgsm, color="#e67e22", linewidth=2, 
             label="FGSM", marker="o")
axes[2].plot(epsilons, ssim_pgd,  color="#e74c3c", linewidth=2, 
             label="PGD",  marker="s")
axes[2].set_title("Similarité perceptuelle (SSIM)")
axes[2].set_xlabel("Epsilon")
axes[2].set_ylabel("SSIM (1 = identique)")
axes[2].set_ylim(0, 1.05)
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("demo/evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[+] Sauvegardé : demo/evaluation.png")

# ── Export JSON (utile pour le README) ────────────────────────────────────────
results = {"fgsm": r_fgsm, "pgd": r_pgd}
with open("report/results.json", "w") as f:
    json.dump(results, f, indent=2, default=lambda x: float(x))
print("[+] Sauvegardé : report/results.json")