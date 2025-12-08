from demo3 import SmallMLP
import torch

my_model = SmallMLP(in_dim=28*28, hidden_dim=128, out_dim=10)

# Get a random set of input matrices
input = torch.randn(16, 1, 28, 28) # High-dimensional tensor: batch_size, channels, x-size, y-size
labels = torch.randint(0, 10, (16,)) # A 1D array of random integers (size 16) between 0 & 9

# Define a loss function
loss_fun = torch.nn.CrossEntropyLoss()

# Define an optimizer
opt = torch.optim.SGD(params=my_model.parameters(), lr=1e-3)

# Reset the gradients
opt.zero_grad()

# Simulate one step of forward flow of information
logits = my_model(input)

# Calculate the loss
my_loss = loss_fun(logits, labels)

# Go back assign credit to all parameters (autograd)
my_loss.backward()

# Use optimizer to adjust weights i.e. parameters of network
opt.step()

print("Loss: ", my_loss.item())