# Red Team Report — AdversarialVision
**Date :** 2025  
**Cible :** ResNet50 pré-entraîné ImageNet (torchvision)  
**Auteur :** [Ton nom]  
**Objectif :** Évaluer la robustesse d'un classificateur d'images face aux attaques adversariales

---

## 1. Résumé exécutif

Ce rapport documente deux attaques adversariales de type white-box conduites sur
ResNet50, un modèle de classification d'images atteignant 89% de confiance sur
l'image cible. Les deux attaques ont réussi à tromper le modèle avec une
perturbation imperceptible à l'œil humain (ε = 0.05, SSIM > 0.93).

**Résultat clé :** PGD (itératif) surpasse FGSM (one-shot) en efficacité tout
en produisant une perturbation plus faible — rendant l'attaque à la fois plus
puissante et plus discrète.

---

## 2. Environnement

| Paramètre       | Valeur                          |
|-----------------|---------------------------------|
| Modèle cible    | ResNet50 (ImageNet1K V1)        |
| Framework       | PyTorch 2.x                     |
| Image testée    | Golden Retriever (idx 207)      |
| Confiance init. | 89.14%                          |
| Device          | CPU / CUDA                      |

---

## 3. Méthodologie

### 3.1 FGSM — Fast Gradient Sign Method (Goodfellow et al., 2014)

Attaque one-shot : un seul pas dans la direction du gradient de la loss
par rapport à l'image.