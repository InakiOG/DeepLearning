import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

# Define a custom dataset class


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(self.classes)}
        self.images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append(
                    (os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def show_batch(loader):
    images, labels = next(iter(loader))
    img = torchvision.utils.make_grid(images)
    plt.figure(figsize=(12, 12))
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Batch of Images from dataset\nLabels: {labels.tolist()}")
    plt.show()


# Create an instance of the custom dataset
custom_data = CustomDataset(root_dir='./data',
                            transform=transform)
# Create a DataLoader
data_loader = DataLoader(dataset=custom_data, batch_size=32,
                         shuffle=True)

show_batch(data_loader)
