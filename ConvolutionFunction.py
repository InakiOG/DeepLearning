import torch


def convolution2d(image, kernel):
    output_shape = (image.shape[0] - kernel.shape[0] + 1,
                    image.shape[1] - kernel.shape[1] + 1)
    output = torch.zeros(output_shape)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[i, j] = torch.sum(image[i:i+kernel.shape[0],
                                           j:j+kernel.shape[1]] * kernel)
    return output


# Example Usage
image = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=torch.float32)
kernel = torch.tensor([[1, 0],
                       [0, -1]], dtype=torch.float32)

output = convolution2d(image, kernel)
print(output)

# Compute convolution manually
manual_result = convolution2d(image, kernel)
print("Manual Convolution Result:\n", manual_result)

# Compute convolution using PyTorch
image_torch = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
kernel_torch = kernel.unsqueeze(0).unsqueeze(0)
pytorch_result = torch.nn.functional.conv2d(
    image_torch, kernel_torch).squeeze()
print("PyTorch Convolution Result:\n", pytorch_result)
