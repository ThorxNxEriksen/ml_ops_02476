import matplotlib.pyplot as plt
import torch
import typer
from data import load_dataset
from model import QuickDrawModel
from tqdm import tqdm
import wandb

DEVICE = torch.device("mps" if torch.torch.backends.mps.is_built() else ("cuda" if torch.cuda.is_available() else "cpu"))

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5, gcp_bucket: bool = False, secret_manager: bool = False) -> None:
    """Train a model on the 'Quick, Draw!' dataset with validation."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    if secret_manager:
        wandb.login(key="4359ea2ef73a2790826a8f0b8fad581d23ca3b68")

    # Initialize wandb
    run = wandb.init(
        project="train_wandb",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # Load datasets using train_function() and validation_function()
    train_set = load_dataset('train', gcp_bucket)
    validation_set = load_dataset('val', gcp_bucket)

    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = QuickDrawModel().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track statistics
    statistics = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in tqdm(range(epochs)):
        print(epoch)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training Loop
        for i, (images, labels) in enumerate(train_dataloader):
            print(i)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_dataloader)
        train_accuracy = train_correct / train_total
        statistics["train_loss"].append(train_loss)
        statistics["train_accuracy"].append(train_accuracy)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in validation_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(validation_dataloader)
        val_accuracy = val_correct / val_total
        statistics["val_loss"].append(val_loss)
        statistics["val_accuracy"].append(val_accuracy)
        
        # Log statistics on wandb
        wandb.log({"train_loss": train_loss, 
                   "train_accuracy": train_accuracy, 
                   "val_loss": val_loss, 
                   "val_accuracy": val_accuracy})

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save model on wandb
    artifact = wandb.Artifact(
            name="efficient_net",
            type="model",
            description=f"Run: {wandb.run.id}"
        )
    # Save model
    print("Training complete")
    if gcp_bucket:
        torch.save(model.state_dict(), "/gcs/quickdraw-databucket/models/quickdraw_model.pth")
        artifact.add_file("/gcs/quickdraw-databucket/models/quickdraw_model.pth")
    else:
        torch.save(model.state_dict(), "models/quickdraw_model.pth")
        artifact.add_file("models/quickdraw_model.pth")
    
    wandb.log_artifact(artifact)

    # Plot training and validation statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="Train Loss", color="blue")
    axs[0].plot(statistics["val_loss"], label="Val Loss", color="orange")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(statistics["train_accuracy"], label="Train Accuracy", color="blue")
    axs[1].plot(statistics["val_accuracy"], label="Val Accuracy", color="orange")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    figure_path = 'reports/figures/training_and_validation_statistics.png'
    fig.savefig(figure_path)
    wandb.log({f"training_and_validation_statistics": wandb.Image(figure_path)})

if __name__ == "__main__":
    typer.run(train)