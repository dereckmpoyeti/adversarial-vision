import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def l2_distance(original, adversarial):
    """Norme L2 de la perturbation — mesure l'amplitude globale."""
    delta = adversarial - original
    return delta.norm(p=2).item()

def linf_distance(original, adversarial):
    """Norme Linf — perturbation maximale sur un pixel."""
    delta = adversarial - original
    return delta.abs().max().item()

def compute_ssim(original, adversarial):
    """
    SSIM (Structural Similarity Index) — mesure la similarité perceptuelle.
    1.0 = images identiques, 0.0 = aucune similarité.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def to_np(t):
        img = t.squeeze(0).cpu().detach() * std + mean
        return img.permute(1,2,0).numpy().clip(0,1)

    orig_np = to_np(original)
    adv_np  = to_np(adversarial)

    return ssim(orig_np, adv_np, channel_axis=2, data_range=1.0)

def attack_success_rate(model, tensors, adv_tensors, true_labels, device):
    """
    ASR : proportion d'images où l'attaque a réussi (mauvaise prédiction).
    tensors et adv_tensors sont des listes de tensors [1,3,224,224].
    """
    success = 0
    for orig, adv, label in zip(tensors, adv_tensors, true_labels):
        adv = adv.to(device)
        with torch.no_grad():
            pred = model(adv).argmax(dim=1).item()
        if pred != label:
            success += 1
    return success / len(tensors)

def full_report(model, original, adversarial, true_label, pred_label, 
                attack_name, epsilon, device):
    """Génère un rapport complet pour une attaque."""
    l2   = l2_distance(original, adversarial)
    linf = linf_distance(original, adversarial)
    sim  = compute_ssim(original, adversarial)
    success = pred_label != true_label

    report = {
        "attack":    attack_name,
        "epsilon":   epsilon,
        "success":   success,
        "l2":        round(l2,   4),
        "linf":      round(linf, 4),
        "ssim":      round(sim,  4),
    }

    print(f"\n{'='*45}")
    print(f"  Rapport — {attack_name}")
    print(f"{'='*45}")
    print(f"  Attaque réussie  : {'✓ OUI' if success else '✗ NON'}")
    print(f"  Epsilon          : {epsilon}")
    print(f"  Perturbation L2  : {l2:.4f}")
    print(f"  Perturbation L∞  : {linf:.4f}")
    print(f"  SSIM             : {sim:.4f}  (1.0 = identique)")
    print(f"{'='*45}")

    return report