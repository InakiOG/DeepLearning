import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

# Define the transformation for the images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the STL10 dataset
STL10_data = STL10(root='DataLoader/data', split='train',
                   download=True, transform=transform)

# Create a DataLoader
data_loader = DataLoader(STL10_data, batch_size=32, shuffle=True)

# Function to display a batch of images


def show_batch(loader):
    images, labels = next(iter(loader))
    img = torchvision.utils.make_grid(images)
    plt.figure(figsize=(12, 12))
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Batch of Images from STL10")
    plt.show()


# Show a batch of images
show_batch(data_loader)
