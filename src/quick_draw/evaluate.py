import torch
import typer
from data import load_dataset
from model import QuickDrawModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """
    Evaluate a trained model.

    Args:
    -----
        model_checkpoint (str): The path to the model checkpoint.
    
    Returns:
    --------
        None
    """
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = QuickDrawModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    test_set = load_dataset("test")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
