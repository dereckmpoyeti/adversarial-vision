import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from models.target_model import load_model, load_labels, preprocess, predict
from attacks.fgsm import fgsm_attack
from attacks.pgd  import pgd_attack, pgd_targeted

IMAGE_PATH = "images.jpg"

labels        = load_labels()
model, device = load_model()
tensor        = preprocess(IMAGE_PATH)

clean_preds      = predict(model, tensor, labels, device, top_k=1)
true_label_name  = clean_preds[0][0]
true_label_idx   = labels.index(true_label_name)
print(f"[+] Label cible : {true_label_name} (idx {true_label_idx})")

EPSILON = 0.05

# ── Attaques ─────────────────────────────────────────────────────────────────
fgsm_adv, _, _  = fgsm_attack(model, tensor.clone(), true_label_idx, EPSILON, device)
pgd_adv, history = pgd_attack(model, tensor.clone(), true_label_idx, EPSILON, device)

# PGD ciblé : on force "Labrador Retriever" (idx 208)
target_idx  = 208
target_name = labels[target_idx]
pgd_targeted_adv = pgd_targeted(model, tensor.clone(), target_idx, EPSILON, device)

# ── Prédictions ──────────────────────────────────────────────────────────────
clean_p    = predict(model, tensor,            labels, device, top_k=3)
fgsm_p     = predict(model, fgsm_adv,          labels, device, top_k=3)
pgd_p      = predict(model, pgd_adv,           labels, device, top_k=3)
pgd_tgt_p  = predict(model, pgd_targeted_adv,  labels, device, top_k=3)

def print_preds(title, preds):
    print(f"\n[{title}]")
    for l, c in preds: print(f"  {l:<35} {c:.2f}%")

print_preds("ORIGINAL",        clean_p)
print_preds(f"FGSM  ε={EPSILON}",  fgsm_p)
print_preds(f"PGD   ε={EPSILON}",  pgd_p)
print_preds(f"PGD ciblé → {target_name}", pgd_tgt_p)

# ── Helper dénorm ─────────────────────────────────────────────────────────────
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def to_img(t):
    img = t.squeeze(0).cpu().detach() * std + mean
    return img.permute(1,2,0).numpy().clip(0,1)

# ── Figure 1 : comparaison 4 images ──────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle(f"FGSM vs PGD  —  ε = {EPSILON}", fontsize=13, fontweight="bold")

entries = [
    (tensor,           clean_p,   "#2ecc71", "Original"),
    (fgsm_adv,         fgsm_p,    "#e74c3c", "FGSM"),
    (pgd_adv,          pgd_p,     "#e74c3c", "PGD (untargeted)"),
    (pgd_targeted_adv, pgd_tgt_p, "#9b59b6", f"PGD ciblé\n→ {target_name}"),
]

for ax, (t, preds, color, method) in zip(axes, entries):
    ax.imshow(to_img(t))
    label, conf = preds[0]
    ax.set_title(f"{method}\n{label[:18]}\n{conf:.1f}%", fontsize=9)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("demo/pgd_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2 : courbe de convergence PGD ─────────────────────────────────────
iters, losses = zip(*history)
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(iters, losses, color="#e74c3c", linewidth=2)
ax2.set_xlabel("Itération")
ax2.set_ylabel("Loss (CrossEntropy)")
ax2.set_title("PGD — convergence de la loss au fil des itérations")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("demo/pgd_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[+] Sauvegardés : demo/pgd_comparison.png  |  demo/pgd_convergence.png")