import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from models.target_model import load_model, load_labels, preprocess, predict
from attacks.fgsm import fgsm_attack, fgsm_sweep

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
IMAGE_PATH = "images.jpg"   # adapte si besoin
EPSILON    = 0.05           # commence par ça

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
labels      = load_labels()
model, device = load_model()
tensor      = preprocess(IMAGE_PATH)

# Label correct = top-1 de l'image propre
clean_preds = predict(model, tensor, labels, device, top_k=1)
true_label_name, _ = clean_preds[0]
true_label_idx = labels.index(true_label_name)
print(f"[+] Label cible : {true_label_name} (idx {true_label_idx})")

# ──────────────────────────────────────────────
# Attaque FGSM
# ──────────────────────────────────────────────
adv_tensor, perturbation, loss = fgsm_attack(
    model, tensor.clone(), true_label_idx, EPSILON, device
)

clean_preds = predict(model, tensor,     labels, device, top_k=3)
adv_preds   = predict(model, adv_tensor, labels, device, top_k=3)

print(f"\n[ORIGINAL]")
for l, c in clean_preds: print(f"  {l:<35} {c:.2f}%")
print(f"\n[ADVERSARIAL  ε={EPSILON}]")
for l, c in adv_preds:   print(f"  {l:<35} {c:.2f}%")

# ──────────────────────────────────────────────
# Helper dénorm → image affichable
# ──────────────────────────────────────────────
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def to_img(t):
    img = t.squeeze(0).cpu() * std + mean
    return img.permute(1,2,0).numpy().clip(0,1)

# ──────────────────────────────────────────────
# Visualisation : Original | Perturbation | Adversarial
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.suptitle(f"FGSM Attack  —  ε = {EPSILON}", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(1, 3, wspace=0.35)

imgs   = [to_img(tensor), to_img(perturbation * 10 + 0.5), to_img(adv_tensor)]
titles = [
    f"Original\n{clean_preds[0][0]}  {clean_preds[0][1]:.1f}%",
    f"Perturbation × 10\n(invisible à ε={EPSILON})",
    f"Adversarial\n{adv_preds[0][0]}  {adv_preds[0][1]:.1f}%"
]
colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

for i, (img, title, color) in enumerate(zip(imgs, titles, colors)):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.savefig("demo/fgsm_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[+] Sauvegardé : demo/fgsm_result.png")

# ──────────────────────────────────────────────
# Sweep epsilon : montre la dégradation progressive
# ──────────────────────────────────────────────
print("\n[+] Sweep epsilon...")
epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
sweep    = fgsm_sweep(model, tensor.clone(), true_label_idx, device, epsilons)

fig2, axes = plt.subplots(1, len(epsilons), figsize=(18, 3))
fig2.suptitle("FGSM — Impact de ε sur la prédiction", fontsize=13, fontweight="bold")

for ax, (eps, adv_t, _, __) in zip(axes, sweep):
    preds = predict(model, adv_t, labels, device, top_k=1)
    label_pred, conf = preds[0]
    attack_success = label_pred != true_label_name

    ax.imshow(to_img(adv_t))
    ax.set_title(f"ε={eps}\n{label_pred[:14]}\n{conf:.1f}%", fontsize=8)
    ax.axis("off")
    color = "#e74c3c" if attack_success else "#2ecc71"
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("demo/fgsm_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print("[+] Sauvegardé : demo/fgsm_sweep.png")