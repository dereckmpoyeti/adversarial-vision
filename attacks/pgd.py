import torch
import torch.nn as nn

def pgd_attack(model, tensor, true_label, epsilon, device,
               alpha=None, num_iter=40, random_start=True):
    """
    Génère un exemple adversarial via PGD (Madry et al., 2018).

    Args:
        model       : modèle cible
        tensor      : image préprocessée [1, 3, 224, 224]
        true_label  : index ImageNet correct
        epsilon     : rayon de la boule de perturbation
        device      : cpu ou cuda
        alpha       : step size (défaut = epsilon / 10)
        num_iter    : nombre d'itérations
        random_start: démarre avec un bruit aléatoire dans la boule ε

    Returns:
        adv_tensor  : image adversariale [1, 3, 224, 224]
        history     : liste de (iter, loss) pour visualiser la convergence
    """
    if alpha is None:
        alpha = epsilon / 10

    tensor = tensor.to(device)
    label  = torch.tensor([true_label]).to(device)
    criterion = nn.CrossEntropyLoss()

    # Démarrage : image originale + bruit aléatoire optionnel
    if random_start:
        delta = torch.empty_like(tensor).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(tensor)

    delta = delta.to(device)
    x_adv = (tensor + delta).clone()

    history = []

    for i in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)

        output = model(x_adv)
        loss   = criterion(output, label)
        loss.backward()

        # Petit pas dans la direction du gradient
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # Projection : on reste dans la boule ε autour de l'original
            delta = torch.clamp(x_adv - tensor, -epsilon, epsilon)
            x_adv = tensor + delta

        history.append((i, loss.item()))

    return x_adv.detach(), history


def pgd_targeted(model, tensor, target_label, epsilon, device,
                 alpha=None, num_iter=40):
    if alpha is None:
        alpha = epsilon / 10

    tensor = tensor.to(device)
    label  = torch.tensor([target_label]).to(device)
    criterion = nn.CrossEntropyLoss()

    x_adv = tensor.clone().detach()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)   # ← detach d'abord

        output = model(x_adv)
        loss   = criterion(output, label)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv - alpha * x_adv.grad.sign()
            delta  = torch.clamp(x_adv - tensor, -epsilon, epsilon)
            x_adv  = (tensor + delta).detach()

    return x_adv