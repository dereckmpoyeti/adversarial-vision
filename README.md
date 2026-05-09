# Adversarial Vision

Projet Python de vision par ordinateur qui explore les attaques adversariales contre des modèles de classification d'images.

Le projet montre comment une petite perturbation, parfois presque invisible pour un humain, peut modifier la prédiction d'un modèle de deep learning pré-entraîné.

## Objectif

L'objectif est de comprendre et démontrer le fonctionnement des attaques adversariales sur des modèles de vision, en particulier avec l'attaque FGSM sur un modèle ResNet50 pré-entraîné sur ImageNet.

## Fonctionnalités actuelles

- Chargement d'un modèle ResNet50 pré-entraîné avec PyTorch et torchvision
- Prétraitement standard des images ImageNet
- Prédiction des classes les plus probables
- Implémentation de l'attaque FGSM
- Comparaison entre image originale et image adversariale
- Visualisation des perturbations générées
- Test de plusieurs valeurs d'epsilon pour observer l'impact de l'attaque

## Architecture du projet

```txt
adversarial-vision/
|-- attacks/
|   |-- __init__.py
|   |-- fgsm.py
|   `-- pgd.py
|-- defenses/
|-- demo/
|   `-- run_fgsm.py
|-- evaluation/
|-- models/
|   `-- target_model.py
|-- report/
|-- architecture.txt
|-- images.jpg
|-- requirements.txt
`-- README.md
```

## Installation

Cloner le projet:

```bash
git clone https://github.com/dereckmpoyeti/adversarial-vision.git
cd adversarial-vision
```

Créer un environnement virtuel:

```bash
python -m venv .venv
```

Activer l'environnement virtuel:

Sur Windows:

```bash
.venv\Scripts\activate
```

Sur Linux ou macOS:

```bash
source .venv/bin/activate
```

Installer les dépendances:

```bash
pip install -r requirements.txt
```

## Lancement de la démonstration FGSM

```bash
python demo/run_fgsm.py
```

Le script charge l'image `images.jpg`, effectue une prédiction avec ResNet50, applique l'attaque FGSM, puis affiche les prédictions avant et après l'attaque.

Il génère aussi des visualisations dans le dossier `demo/`.

## Dépendances principales

- PyTorch
- torchvision
- Pillow
- NumPy
- Matplotlib

## Concepts utilisés

- Vision par ordinateur
- Deep learning
- Classification d'images
- Modèles pré-entraînés
- ResNet50
- ImageNet
- Attaques adversariales
- FGSM
- Gradients

## Prochaines améliorations possibles

- Implémenter l'attaque PGD
- Ajouter des défenses contre les attaques adversariales
- Ajouter une évaluation quantitative du taux de réussite des attaques
- Tester plusieurs images
- Comparer plusieurs modèles de classification
- Ajouter un rapport d'analyse dans le dossier `report/`

## Auteur

Projet réalisé par Dereck Mpoyeti.
