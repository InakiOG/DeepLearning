import torch

tensor_7x7 = torch.rand(7, 7)
print(tensor_7x7)

tensor_1x7 = torch.rand(1, 7)
print(tensor_1x7)

result = torch.matmul(tensor_1x7, tensor_7x7)

print(result)
