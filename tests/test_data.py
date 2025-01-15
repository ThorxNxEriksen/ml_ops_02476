from torch.utils.data import Dataset

from quick_draw.data import load_dataset

def test_my_dataset():
    """Test the MyDataset class."""
    train_set = load_dataset('train')
    val_set = load_dataset('val')
    assert isinstance(train_set, Dataset)
    assert isinstance(val_set, Dataset)
