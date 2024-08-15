import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import numpy as np
import urllib.request
from PIL import Image
import torch
from torchvision import transforms

# Function to download an image from a URL
def download_image(url, filename):
    urllib.request.urlretrieve(url, filename)
    return filename

# Download a random image
image_url = "https://picsum.photos/224/224"  # Random image URL
image_path = download_image(image_url, "random_image.jpg")

# Load the image
image = Image.open(image_path)

# Display the original image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

# Normalize the image using ImageNet mean and std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformation
image_tensor = transform(image).numpy()

# Denormalize for visualization (bring back to [0, 1] range)
image_tensor = np.transpose(image_tensor, (1, 2, 0))  # CxHxW -> HxWxC
image_tensor = np.clip(image_tensor, 0, 1)

# Display the normalized image
plt.subplot(1, 2, 2)
plt.imshow(image_tensor)
plt.title("Normalized Image")

plt.show()
