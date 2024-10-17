import torch
from torchvision import transforms
from PIL import Image
import timm

# Ścieżka do obrazka
image_path = '/home/admin2/Documents/repos/cct/data/appsilon/17flowers/jpg/image_0038.jpg'


# Funkcja do wczytania i przetworzenia obrazka
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    # Transformacje: Resize do 224x224 i normalizacja jak w ViT
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image).unsqueeze(0)  # Dodaj wymiar batcha
    return image


# Wczytaj i przetwórz obraz
image_tensor = preprocess_image(image_path)

# Stworzenie modelu Vision Transformer (ViT)
# TODO sprwadzic vit_base_patch16_384
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=17)

# Ustaw model w tryb ewaluacji (nie trenowania)
model.eval()

# Przepuść batch jednoelementowy przez model
with torch.no_grad():
    output = model(image_tensor)

# Wyświetl wynik
print(f"Output shape: {output.shape}")
print(f"Output: {output}")
