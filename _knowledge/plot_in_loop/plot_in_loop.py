import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from PIL import Image
from pathlib import Path

# Path to the folder containing images
output_path = Path(__file__).parent / "files"

# List of image paths
list_image_paths = list(output_path.glob("*.jpg"))

# Function to plot a single image and wait for keypress or mouse click
def plot_image_standalone(image_path):
    img = Image.open(image_path)  # Open the image

    # Create a new figure for each image
    plt.figure()
    plt.imshow(img)
    plt.title(f"Image: {image_path.name}")
    plt.axis('off')  # Turn off axis

    # Display the plot and wait for a key or mouse button press
    plt.show(block=False)
    plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
    plt.close()  # Close the current figure after keypress/mouse click

# Loop through all image paths
for path in list_image_paths:
    plot_image_standalone(path)
    # The loop will proceed to the next image after the user presses a key or clicks
