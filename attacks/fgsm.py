import torch
import torch.nn as nn

def fgsm_attack(model, tensor, true_label, epsilon, device):
    """
    Génère un exemple adversarial via FGSM.

    Args:
        model    : le modèle cible (ResNet50)
        tensor   : image préprocessée [1, 3, 224, 224]
        true_label: label correct (int, index ImageNet)
        epsilon  : intensité de la perturbation [0.0 - 1.0]
        device   : cpu ou cuda

    Returns:
        adv_tensor     : image perturbée [1, 3, 224, 224]
        perturbation   : la perturbation brute (pour analyse)
        loss_value     : valeur de la loss sur l'image originale
    """
    tensor = tensor.to(device)
    tensor.requires_grad = True  # clé : on veut le gradient par rapport à l'image

    label = torch.tensor([true_label]).to(device)
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(tensor)
    loss = criterion(output, label)

    # Backward pass — gradient par rapport à l'IMAGE
    model.zero_grad()
    loss.backward()

    # Signe du gradient → perturbation
    gradient_sign = tensor.grad.sign()
    perturbation = epsilon * gradient_sign

    # Image adversariale + clamp pour rester dans [0,1] après dénorm
    adv_tensor = tensor + perturbation
    adv_tensor = adv_tensor.detach()

    return adv_tensor, perturbation.detach(), loss.item()


def fgsm_sweep(model, tensor, true_label, device, epsilons=None):
    """
    Teste FGSM sur plusieurs valeurs d'epsilon.
    Retourne une liste de (epsilon, adv_tensor, top1_label, top1_conf).
    """
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    results = []
    for eps in epsilons:
        if eps == 0.0:
            results.append((0.0, tensor.clone(), None, None))
            continue

        adv, _, _ = fgsm_attack(model, tensor.clone(), true_label, eps, device)
        results.append((eps, adv, None, None))

    return results