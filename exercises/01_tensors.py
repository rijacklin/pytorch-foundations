# imports
import torch
from torch import nn

#################
#     DAY 1     #
#################

### Exercise 1

# float tensor
fl_tensor = torch.tensor(
    [
        [0.6114, 0.7254, 0.5923, 0.9191],
        [0.0314, 0.8007, 0.1688, 0.2715],
        [0.8353, 0.8090, 0.2216, 0.4080],
    ]
)
print(f"Shape of fl_tensor: {fl_tensor.shape}")
print(f"Datatype of fl_tensor: {fl_tensor.dtype}")
print(f"Tensor is stored on device: {fl_tensor.device}")
print()

# integer index tensors - dim: 0 = row; 1 col
indices = torch.tensor([0, 2])
int_index_0 = torch.index_select(fl_tensor, 0, indices)
assert int_index_0.equal(
    torch.tensor([[0.6114, 0.7254, 0.5923, 0.9191], [0.8353, 0.8090, 0.2216, 0.4080]])
), "incorrect"
print(f"Shape of indices: {indices.shape}")
print(f"Datatype of indices: {indices.dtype}")
print(f"Tensor is stored on device: {indices.device}")
print()

int_index_1 = torch.index_select(fl_tensor, 1, indices)
assert int_index_1.equal(
    torch.tensor([[0.6114, 0.5923], [0.0314, 0.1688], [0.8353, 0.2216]])
), "incorrect"
print(f"Shape of int_index_1: {int_index_1.shape}")
print(f"Datatype of int_index_1: {int_index_1.dtype}")
print(f"Tensor is stored on device: {int_index_1.device}")
print()

# slices
slice_0 = fl_tensor[0:2]
assert slice_0.equal(
    torch.tensor([[0.6114, 0.7254, 0.5923, 0.9191], [0.0314, 0.8007, 0.1688, 0.2715]])
), "incorrect"
print(f"Shape of slice_0: {slice_0.shape}")
print(f"Datatype of slice_0: {slice_0.dtype}")
print(f"Tensor is stored on device: {slice_0.device}")
print()

slice_1 = fl_tensor[:, 1:4:2]
assert slice_1.equal(
    torch.tensor([[0.7254, 0.9191], [0.8007, 0.2715], [0.8090, 0.4080]])
), "incorrect"
print(f"Shape of slice_1: {slice_1.shape}")
print(f"Datatype of slice_1: {slice_1.dtype}")
print(f"Tensor is stored on device: {slice_0.device}")
print()

# transposes
transpose_row_col = torch.transpose(fl_tensor, 0, 1)
assert transpose_row_col.equal(
    torch.tensor(
        [
            [0.6114, 0.0314, 0.8353],
            [0.7254, 0.8007, 0.8090],
            [0.5923, 0.1688, 0.2216],
            [0.9191, 0.2715, 0.4080],
        ]
    )
), "incorrect"
print(f"Shape of transpose_row_col: {transpose_row_col.shape}")
print(f"Datatype of transpose_row_col: {transpose_row_col.dtype}")
print(f"Tensor is stored on device: {slice_0.device}")
print()

transpose_col_row = torch.transpose(fl_tensor, 1, 0)
assert transpose_col_row.equal(
    torch.tensor(
        [
            [0.6114, 0.0314, 0.8353],
            [0.7254, 0.8007, 0.8090],
            [0.5923, 0.1688, 0.2216],
            [0.9191, 0.2715, 0.4080],
        ]
    )
), "incorrect"
print(f"Shape of transpose_col_row: {transpose_col_row.shape}")
print(f"Datatype of transpose_col_row: {transpose_col_row.dtype}")
print(f"Tensor is stored on device: {slice_0.device}")
print()

# reshapes
fl_tensor_reshape = fl_tensor.reshape(4, 3)
assert fl_tensor_reshape.equal(
    torch.tensor(
        [
            [0.6114, 0.7254, 0.5923],
            [0.9191, 0.0314, 0.8007],
            [0.1688, 0.2715, 0.8353],
            [0.8090, 0.2216, 0.4080],
        ]
    )
), "incorrect"
print(f"Shape of fl_tensor_reshape: {fl_tensor_reshape.shape}")
print(f"Datatype of fl_tensor_reshape: {fl_tensor_reshape.dtype}")
print(f"Tensor is stored on device: {slice_0.device}")
print()

### Exercise 2

print(f"Predicting that [4,3] + [3] = RuntimeError")
print(f"Predicting that [4,1] + [1,3] = [5,4]")
print(f"Predicting that [4,3] + [4] = RuntimeError")
print()

tensor_1a = torch.ones(4, 3)
tensor_1b = torch.ones(3)
print(f"[4,3] + [3] = {torch.add(tensor_1a, tensor_1b)}")
print()

tensor_2a = torch.ones(4, 1)
tensor_2b = torch.ones(1, 3)
print(f"[4,3] + [3] = {torch.add(tensor_2a, tensor_2b)}")
print()

tensor_3a = torch.ones(4, 3)
tensor_3b = torch.ones(4)
# Doesn't work because trailing 3 and 4 conflict
# print(f"[4,3] + [3] = {torch.add(tensor_3a, tensor_3b)}")

### Exercise 3

batch_size = 8
in_features = 4
out_features = 3

x = torch.randn(batch_size, in_features)
linear_projection = nn.Linear(in_features, out_features)
expected = linear_projection(x)
actual = x @ linear_projection.weight.T + linear_projection.bias

# compare
torch.testing.assert_close(actual, expected)

### Exercise 4

batch_size = 2
sequence_length = 5
in_features = 8
out_features = 12

x = torch.randn(batch_size, sequence_length, in_features)

linear_projection = nn.Linear(in_features, out_features)
expected = linear_projection(x)
actual = x @ linear_projection.weight.T + linear_projection.bias

# compare
torch.testing.assert_close(actual, expected)

"""
The first two axes (batches, sequence_length) are preserved. The final axis is changed from 8 to 12. PyTorch effectively applies the same 8 -> 12 projection to each of the 2 x 5 feature vectors independently.
"""
print(x.shape)
print(actual.shape)

### Exercise 5

batch_size = 64  # number of vectors
in_features = 10
out_features = 5

vectors = torch.randn(batch_size, in_features)  # [64, 10]
weight = torch.randn(in_features, out_features)  # [10, 5]

# this is very slow
loop_output = torch.empty(batch_size, out_features)
for i in range(batch_size):
    loop_output[i] = vectors[i] @ weight

# batched multiplication is fast
batched_output = vectors @ weight

# compare
torch.testing.assert_close(loop_output, batched_output)

"""
Summary: Batching lets you express all 64 independent projections in one operation.
"""

#################
#  EXIT CHECK   #
#################

B = 8
T = 4
D = 2
H = 3

x = torch.randn(B, T, D)
y = torch.randn(D, H)
result = torch.randn(B, T, H)
print(result.shape)
print((x @ y).shape)

"""
[B, T, D] @ [D, H] becomes [B, T, H] because of the shape rule: the shared dimension (i.e., D) disappears from the output shape because matrix multiplication multiples and sums along that dimension.
"""

f32_payload = torch.empty(8, 128, 256, dtype=torch.float32)
# numel gives total number of elements
f32_payload_size = f32_payload.numel() * f32_payload.element_size()
print(f32_payload_size)

"""
When calling `torch.transpose` on a tensor, PyTorch doesn't copy or rearrange the underlying data in memory. This causes the tensor to be non-contiguous. Thus, when trying to call view on the tensor, which requires the underlying tensor data memory to be stored contiguously, we get a RuntimeError. The recommended fix is to call reshape on the tensor, which automatically handles non-contiguous tensors.
"""

#################
#     DAY 2     #
#################

### Exercise 1
