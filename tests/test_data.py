import os

import pytest
from torch.utils.data import Dataset

from src.quick_draw.data import load_dataset
from tests import _PATH_DATA

# Define the path to the processed train dataset
train_dataset_path = os.path.join(_PATH_DATA, 'processed', 'train_dataset.pt')

@pytest.mark.skipif(not os.path.exists(train_dataset_path), reason="Processed datasets not found")
def test_my_dataset():
    """Test the MyDataset class."""
    train_set = load_dataset("train")
    val_set = load_dataset("val")
    assert isinstance(train_set, Dataset)
    assert isinstance(val_set, Dataset)
