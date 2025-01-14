from quickdraw import QuickDrawDataGroup
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as transforms
from typing import Callable, Optional, Tuple, Union, List 

def load_data_categories(categories: List[str]):
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    for category in categories: 
        print(category)
        data = QuickDrawDataGroup(category)
        images = []

        for i, drawing in enumerate(data.drawings):
            drawing_tensor = transform(drawing.image)
            drawing_tensor = torch.mean(drawing_tensor, axis=0, dtype=float)// 255
            images.append(drawing_tensor)

        images_tensor = torch.stack(images)
        # Save as .pt file
        torch.save(images_tensor, f"data/raw/{category}.pt")

def check_loaded_data(category: str, index: int):
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

