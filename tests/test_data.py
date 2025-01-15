import os

import pytest
from torch.utils.data import Dataset

from src.quick_draw.data import load_dataset
from tests import _PATH_DATA
import torch
from src.quick_draw.data import load_dataset, split_data, preprocess

# Define the path to the processed train dataset
train_dataset_path = os.path.abspath(os.path.join(_PATH_DATA, 'processed', 'train_dataset.pt'))

@pytest.mark.skipif(not os.path.exists(train_dataset_path), reason="Processed datasets not found")
def test_my_dataset():
    """Test the MyDataset class."""
    train_set = load_dataset("train")
    val_set = load_dataset("val")

    # Define the path to the processed train dataset
    train_dataset_path = os.path.abspath(os.path.join(_PATH_DATA, 'processed', 'train_dataset.pt'))

    @pytest.mark.skipif(not os.path.exists(train_dataset_path), reason="Processed datasets not found")
    def test_load_dataset():
        """Test the load_dataset function."""
        train_set = load_dataset("train")
        val_set = load_dataset("val")
        test_set = load_dataset("test")
        assert isinstance(train_set, Dataset)
        assert isinstance(val_set, Dataset)
        assert isinstance(test_set, Dataset)

    def test_split_data():
        """Test the split_data function."""
        tensor = torch.randn(100, 3, 28, 28)  # Example tensor with 100 samples
        train_ratio = 0.7
        val_ratio = 0.15
        train_data, val_data, test_data = split_data(tensor, train_ratio, val_ratio)
        assert len(train_data) == 70
        assert len(val_data) == 15
        assert len(test_data) == 15

    @pytest.mark.skipif(not os.path.exists(train_dataset_path), reason="Processed datasets not found")
    def test_preprocess():
        """Test the preprocess function."""
        categories = ['bear', 'broccoli', 'cake']
        train_ratio = 0.7
        val_ratio = 0.15
        preprocess(categories, train_ratio, val_ratio)
        
        train_set = load_dataset("train")
        val_set = load_dataset("val")
        test_set = load_dataset("test")
        
        assert isinstance(train_set, Dataset)
        assert isinstance(val_set, Dataset)
        assert isinstance(test_set, Dataset)
        assert len(train_set) > 0
        assert len(val_set) > 0
        assert len(test_set) > 0

