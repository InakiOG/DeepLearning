import torch
import matplotlib.pyplot as plt
from skimage import data, color
import numpy as np


def convolution2d(image, kernel):
    output_shape = (image.shape[0] - kernel.shape[0] + 1,
                    image.shape[1] - kernel.shape[1] + 1)
    output = torch.zeros(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[i, j] = torch.sum(image[i:i+kernel.shape[0],
                                           j:j+kernel.shape[1]] * kernel)
    return output


# Load astronaut image and convert to grayscale
image = data.astronaut()


def convolution3d(image, kernel):
    red_layer = convolution2d(image[:, :, 0], kernel)
    green_layer = convolution2d(image[:, :, 1], kernel)
    blue_layer = convolution2d(image[:, :, 2], kernel)
    return red_layer, green_layer, blue_layer


edge_kernel = torch.tensor([[0, -1, 0],
                            [-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]],
                           dtype=torch.float32)

image_tensor = torch.tensor(image, dtype=torch.float32)
edge_image = convolution3d(image_tensor, edge_kernel)

# Combine the edge-detected layers back into an RGB image
edge_image_combined = torch.stack(edge_image, dim=2).numpy()

# Clip values to be in the valid range [0, 255] and convert to uint8
edge_image_combined = np.clip(edge_image_combined, 0, 255).astype(np.uint8)

plt.imshow(edge_image_combined)
plt.title('Edge-detected RGB Image')
plt.axis('off')
plt.show()
