from demo3 import SmallMLP
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns

# Decide how I'm going to transform (convert images to tensors) each image
transform = torchvision.transforms.ToTensor()

# Or, you could do a more complex transformation:
other_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # Use mean and std_dev of MNIST images
])

# Load the MNIST dataset
my_mnist_dataset = torchvision.datasets.MNIST(
    root='./datasets', # This is where the dataset will be stored
    train=False, # We only want the test images here, so train=False
    download=True, # Download if you need to
    transform=transform # Apply my transform defined earlier
)

# Wrap the dataset into a DataLoader
my_data_loader = torch.utils.data.DataLoader(
    dataset=my_mnist_dataset,
    batch_size=16,
    shuffle=False
)

# Create a model (MLP)
my_model = SmallMLP(in_dim=28*28, hidden_dim=128, out_dim=10)

# Get one batch from the dataloader
images, labels = next(iter(my_data_loader))

# Example of iter
this_is_my_list = [1, 2, 3, 4, 5]
my_iterator = iter(this_is_my_list)
print(next(my_iterator)) # This should print 1
print(next(my_iterator)) # This should print 2
print(next(my_iterator)) # This should print 3

# Visualise this batch of images & labels
# plt.figure(figsize=(12,8))
# for ith_image in range(16):
#     plt.subplot(4, 4, ith_image+1)
#     plt.imshow(images[ith_image].squeeze(), cmap='gray')
#     plt.title(f"Label: {labels[ith_image]}")
#     plt.axis('off')

# plt.show()

# Shove the images through the network
my_model.eval()
logits = my_model(images)
probs = torch.softmax(logits, dim=1)
preds = torch.argmax(logits, dim=1)

# Visualise the same batch of images, now with predictions
# plt.figure(figsize=(12,8))
# for ith_image in range(16):
#     plt.subplot(4, 4, ith_image+1)
#     plt.imshow(images[ith_image].squeeze(), cmap='gray')
#     plt.title(f"Label: {labels[ith_image]}, Pred: {preds[ith_image]}")
#     plt.axis('off')

# plt.show()


# Load the MNIST training dataset
my_mnist_train_dataset = torchvision.datasets.MNIST(
    root='./datasets', # This is where the dataset will be stored
    train=True, # We load training images
    download=True, # Download if you need to
    transform=transform # Apply my transform defined earlier
)

# Wrap the dataset into a DataLoader
my_train_data_loader = torch.utils.data.DataLoader(
    dataset=my_mnist_train_dataset,
    batch_size=16,
    shuffle=True
)


# Define optimizer & loss function
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(my_model.parameters(), lr=1e-3)

# Training loop
my_model.train()
num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in my_train_data_loader:
        opt.zero_grad()
        logits = my_model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        running_acc += acc.item()
    avg_loss = running_loss / len(my_train_data_loader)
    avg_acc = running_acc / len(my_train_data_loader)
    print(f"Loss: {avg_loss}, Accuracy: {avg_acc}")


# Get one batch from the dataloader
images, labels = next(iter(my_data_loader))

# Shove the images through the network
my_model.eval()
logits = my_model(images)
probs = torch.softmax(logits, dim=1)
preds = torch.argmax(logits, dim=1)

# Visualise the same batch of images, now with predictions
plt.figure(figsize=(12,8))
for ith_image in range(16):
    plt.subplot(4, 4, ith_image+1)
    plt.imshow(images[ith_image].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[ith_image]}, Pred: {preds[ith_image]}")
    plt.axis('off')

plt.show()