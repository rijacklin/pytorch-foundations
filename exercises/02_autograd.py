# imports
import torch

#################
#   EXERCISE 1  #
#################

x = torch.tensor(2.0)  # input
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
target = torch.tensor(5.0)

prediction = w * x + b  # \hat{y} = wx + b
print(prediction)  # (3 * 2) + 1 = 7

loss = (prediction - target) ** 2  # L = (\hat{y} - y)^2
print(loss)  # (7 - 5) ** 2 == 4

# apply the chain rule through the computation graph to compute parameter gradients
loss.backward()

print(f"w.grad = {w.grad}")  # w should equal 8
print(f"b.grad = {b.grad}")  # b should equal 4

#################
#   EXERCISE 2  #
#################

"""
Compare autograd gradients from loss.backward() with centered finite-difference estimates.
"""

x = torch.tensor(2.0, dtype=torch.float64)  # input
w = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
b = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
target = torch.tensor(5.0, dtype=torch.float64)
h = 1e-6

loss = (w * x + b - target) ** 2
loss.backward()
assert w.grad is not None and b.grad is not None  # to get rid of lsp warning

with torch.no_grad():
    # create finite-difference gradient with respect to w
    grad_w_fd = (((w + h) * x + b - target) ** 2 - ((w - h) * x + b - target) ** 2) / (
        2 * h
    )
    grad_b_fd = ((w * x + (b + h) - target) ** 2 - (w * x + (b - h) - target) ** 2) / (
        2 * h
    )

print("w:", w.grad.item(), grad_w_fd.item())
print("b:", b.grad.item(), grad_b_fd.item())

#################
#   EXERCISE 3  #
#################

x = torch.tensor(2.0)  # input
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
target = torch.tensor(5.0)

loss = (w * x + b - target) ** 2
loss.backward()
assert w.grad is not None and b.grad is not None  # to get rid of lsp warning
print(f"After first call to .backward() - w: {w.grad.item()}, b: {b.grad.item()}")

# recompute the forward pass and backpropagate again, without clearing grads
loss = (w * x + b - target) ** 2
loss.backward()
assert w.grad is not None and b.grad is not None  # to get rid of lsp warning
print(f"After second call to .backward() - w: {w.grad.item()}, b: {b.grad.item()}")

# doing the same with gradient clearing
w.grad = None
b.grad = None

loss = (w * x + b - target) ** 2
loss.backward()
assert w.grad is not None and b.grad is not None  # to get rid of lsp warning
print(
    f"After first call to .backward() with clearing - w: {w.grad.item()}, b: {b.grad.item()}"
)

w.grad = None
b.grad = None

loss = (w * x + b - target) ** 2
loss.backward()
assert w.grad is not None and b.grad is not None  # to get rid of lsp warning
print(
    f"After second call to .backward() with clearing - w: {w.grad.item()}, b: {b.grad.item()}"
)

#################
#   EXERCISE 4  #
#################

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # input
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# set reproducible seed value
torch.manual_seed(0)

noise = 0.5 * torch.randn_like(x)  # noise with stddev 0.5
target = 3 * x + 2 + noise

lr = 0.01  # learning rate

# training loop
for step in range(500):
    loss = ((w * x + b - target) ** 2).mean()
    loss.backward()
    assert w.grad is not None and b.grad is not None  # to get rid of lsp warning

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # clear param gradients
    w.grad = None
    b.grad = None

with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()

# print(f"w: {w.item():.3f}")
# print(f"b: {b.item():.3f}")
# print(f"final loss: {final_loss.item():.3f}")

#################
#   EXERCISE 5  #
#################


def training_run(optimizer):
    # training loop
    for step in range(500):
        # reset the gradients
        optimizer.zero_grad()

        # calculate loss function and back propagate params
        loss = ((w * x + b - target) ** 2).mean()
        loss.backward()
        assert w.grad is not None and b.grad is not None  # to get rid of lsp warning

        # perform a single optimization step
        optimizer.step()


# set reproducible seed value
torch.manual_seed(0)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # input
noise = 0.5 * torch.randn_like(x)  # noise with stddev 0.5
target = 3 * x + 2 + noise

with torch.no_grad():
    initial_loss = ((3.0 * x + 1.0 - target) ** 2).mean()

print(f"Initial loss: {initial_loss.item():.3f}")

# SGD with learning_rate = 0.001
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.001)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"SGD w/ LR=0.001 -- final loss: {final_loss.item():.3f}")

# SGD with learning_rate = 0.01
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.01)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"SGD w/ LR=0.01 -- final loss: {final_loss.item():.3f}")
# SGD with learning_rate = 0.1

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.1)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"SGD w/ LR=0.1 -- final loss: {final_loss.item():.3f}")
# Adam with learning_rate = 0.001

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.001)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"Adam w/ LR=0.001 -- final loss: {final_loss.item():.3f}")

# Adam with learning_rate = 0.01
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.01)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"Adam w/ LR=0.01 -- final loss: {final_loss.item():.3f}")

# Adam with learning_rate = 0.1
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.Adam([w, b], lr=0.1)
training_run(optimizer)
with torch.no_grad():
    final_loss = ((w * x + b - target) ** 2).mean()
print(f"w: {w.item():.3f}")
print(f"b: {b.item():.3f}")
print(f"Adam w/ LR=0.1 -- final loss: {final_loss.item():.3f}")
