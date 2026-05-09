import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json

# Labels ImageNet
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

def load_labels():
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
        return json.load(f)

# Preprocessing standard ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(device=None):
    """Charge ResNet50 pré-entraîné sur ImageNet en mode évaluation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()  # IMPORTANT : désactive dropout/batchnorm stochastique
    return model, device

def preprocess(image_path):
    """Charge une image et retourne un tensor [1, 3, 224, 224]."""
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # batch dimension

def predict(model, tensor, labels, device, top_k=5):
    """Retourne les top-k prédictions (label, confidence)."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    
    top_probs, top_idxs = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        results.append((labels[idx.item()], round(prob.item() * 100, 2)))
    return results

# --- Test rapide ---
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    
    labels = load_labels()
    model, device = load_model()
    tensor = preprocess(image_path)
    preds = predict(model, tensor, labels, device)
    
    print(f"\n[+] Prédictions pour : {image_path}")
    for label, conf in preds:
        print(f"    {label:<30} {conf:.2f}%")