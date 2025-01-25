import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

def generate_synthetic_image(size: Tuple[int, int] = (224, 224), text: str = None) -> Image.Image:
    """
    Generate a synthetic image with random noise and optional text.
    Args:
        size (Tuple[int, int]): Size of the image (width, height).
        text (str): Optional text to overlay on the image.
    Returns:
        Image.Image: Generated synthetic image.
    """
    # Create a random noise image
    random_noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(random_noise, mode='RGB')

    # Add text if provided
    if text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
        draw.text(position, text, font=font, fill=(255, 255, 255))

    return image

def generate_image_dataset(output_dir: str, categories: List[str], images_per_category: int = 100):
    """
    Generate a synthetic image dataset for classification.
    Args:
        output_dir (str): Directory to save the dataset.
        categories (List[str]): List of categories (e.g., ['safe', 'unsafe', 'adult']).
        images_per_category (int): Number of images to generate per category.
    """
    os.makedirs(output_dir, exist_ok=True)

    for category in categories:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for i in range(images_per_category):
            # Generate a synthetic image with category name as text
            image = generate_synthetic_image(text=category)
            image_path = os.path.join(category_dir, f"{category}_{i + 1}.jpg")
            image.save(image_path)

    print(f"Generated {images_per_category} images per category in {output_dir}")

# Example usage
if __name__ == "__main__":
    # Define categories and output directory
    categories = ['safe', 'unsafe', 'adult']
    output_dir = "path/to/generated_dataset"

    # Generate the dataset
    generate_image_dataset(output_dir, categories, images_per_category=100)