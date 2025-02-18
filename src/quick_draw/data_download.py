from quickdraw import QuickDrawDataGroup
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as transforms
from typing import Callable, Optional, Tuple, Union, List 
import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import logger

def load_data_categories(categories: List[str]):
    """
    Load data from QuickDrawDataGroup and save as .pt file.

    Args:
    -----
        categories (List[str]): A list of category names to download.

    Returns:
    --------
        None
    
    Saves:
    ------
        {category}.pt: The data (images for specific category) saved at .pt file.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.PILToTensor(), # More transformation will improve the model.
    ])

    for category in categories: 
        logger.info("Category: {category}")

        data = QuickDrawDataGroup(category, max_drawings=1000)
        images = []

        for i, drawing in enumerate(data.drawings):
            drawing_tensor = transform(drawing.image)
            drawing_tensor = torch.mean(drawing_tensor, axis=0, dtype=torch.float32)// 255
            images.append(drawing_tensor.unsqueeze(0))

        images_tensor = torch.stack(images)
        # Save as .pt file
        torch.save(images_tensor, f"data/raw/{category}.pt")

def check_loaded_data(category: str, index: int):
    """
    Check the loaded data from .pt file by plotting the image.
    
    Args:
    -----
        category (str): The category name.
        index (int): The index of the image to check.
        
    Returns:
    --------
        None
    
    Plots:
    ------
        The image of the category at the specified index
    """
    data = torch.load(f'data/raw/{category}.pt', weights_only=True)
    # Convert tensor to NumPy and transpose to (255, 255, 3) for plt.imshow
    image = data[index].permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title(category)
    plt.axis('off')  # Hide axis
    plt.show()
    

if __name__ == "__main__":
    categories = ['bear', 'broccoli', 'cake', 'cloud', 'bush', 'The Mona Lisa', 'The Great Wall of China', 'sea turtle', 'moustache', 'mouth']
    load_data_categories(categories)
    check_loaded_data('bear', 0)

