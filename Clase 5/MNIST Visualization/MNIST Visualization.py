import mlflow
import mlflow.pytorch
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the MNIST dataset for testing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_data = datasets.MNIST(root='./data', train=False,
                           download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Get the best model based on accuracy from MLFlow
best_accuracy = 0
best_model_uri = ""

# Search for all runs in the 'MNIST_Classification' experiment
experiment_name = "MNIST_Classification"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Identify the best model based on accuracy (checking the correct field for accuracy)
for run in runs.itertuples():
    accuracy = run._8  # Index for accuracy, you can print(run)
    if accuracy and accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_uri = run.artifact_uri + "/model"

# Load the best model from MLFlow
best_model = mlflow.pytorch.load_model(best_model_uri)

# Function to display images and predictions


def display_images_with_predictions(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Get the model's predictions
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    # Plot the first 5 images and their predictions
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        # Display prediction and true label
        ax.set_title(
            f"Pred: {predictions[i].item()}, True: {labels[i].item()}")
        ax.axis('off')
    plt.show()


# Display the images with predictions
display_images_with_predictions(best_model, test_loader)
