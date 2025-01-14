from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
import typer
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, Union, List 
from torch.utils.data import TensorDataset


def split_data(tensor, train_ratio, val_ratio):
    train_val_data, test_data = train_test_split(tensor, test_size=1 - (train_ratio + val_ratio))
    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio / (train_ratio + val_ratio))
    return train_data, val_data, test_data


def preprocess(categories: List[str], train_ratio: float, val_ratio: float):
    final_data = {
            'all_train_images' : [],
            'all_val_images' : [],
            'all_test_images' : [],
            'all_train_targets' : [],
            'all_val_targets' : [],
            'all_test_targets' : []}
    
    category_to_label = {category: idx for idx, category in enumerate(categories)}

    for category in categories:
        print(category)
        data = torch.load(f'data/raw/{category}.pt', weights_only=True)
        
        train_images, val_images, test_images = split_data(data, train_ratio, val_ratio)

        label = category_to_label[category]
        train_targets = torch.full((len(train_images),), label, dtype=torch.long)
        val_targets = torch.full((len(val_images),), label, dtype=torch.long)
        test_targets = torch.full((len(test_images),), label, dtype=torch.long)

        final_data['all_train_images'].append(train_images)
        final_data['all_val_images'].append(val_images)
        final_data['all_test_images'].append(test_images)

        final_data['all_train_targets'].append(train_targets)
        final_data['all_val_targets'].append(val_targets)
        final_data['all_test_targets'].append(test_targets)

    # Merge all splits
    train_images_merged = torch.cat(final_data['all_train_images'])
    train_targets_merged = torch.cat(final_data['all_train_targets'])

    val_images_merged = torch.cat(final_data['all_val_images'])
    val_targets_merged = torch.cat(final_data['all_val_targets'])

    test_images_merged = torch.cat(final_data['all_test_images'])
    test_targets_merged = torch.cat(final_data['all_test_targets'])

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images_merged, train_targets_merged)
    val_dataset = TensorDataset(val_images_merged, val_targets_merged)
    test_dataset = TensorDataset(test_images_merged, test_targets_merged)

    # Save TensorDatasets
    torch.save(train_dataset, 'data/processed/train_dataset.pt')
    torch.save(val_dataset, 'data/processed/val_dataset.pt')
    torch.save(test_dataset, 'data/processed/test_dataset.pt')

def load_dataset(dataset_name: str):
    dataset = torch.load(f'data/processed/{dataset_name}_dataset.pt', weights_only=False)   
    return dataset   

if __name__ == "__main__":
    categories = ['bear', 'broccoli', 'cake', 'cloud', 'bush', 'The Mona Lisa', 'The Great Wall of China', 'sea turtle', 'moustache', 'mouth']
    preprocess(categories, 0.7, 0.15)
    load_dataset('train')
    #typer.run(preprocess)
