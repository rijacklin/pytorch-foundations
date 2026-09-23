import torch

A = torch.tensor([[0, 1, 2], [3, 4, 5]])

sum_A = A.sum(1, True)

print(f"shape of A: {A.shape}")
print(f"shape of sum_A: {sum_A.shape}")
print(f"A / sum_A (broadcasting) = {A / sum_A}")
