# What is Auto Encoder?


An autoencoder is a type of neural network that is used for unsupervised learning of efficient data encodings. It learns to compress the input data into a lower-dimensional representation, and then reconstruct the output data from that representation. Autoencoders consist of two main parts: an encoder and a decoder. The encoder takes the input data and maps it to a lower-dimensional representation, while the decoder takes that representation and maps it back to the original data space. The network is trained by minimizing the difference between the input and the reconstructed output using backpropagation. Autoencoders have a wide range of applications, such as data compression, anomaly detection, and image denoising.

# Example

```

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the hyperparameters
input_size = 784  # 28x28 images
hidden_size = 256
latent_size = 64
epochs = 20
batch_size = 128
learning_rate = 1e-3

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create the autoencoder model and optimizer
model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the autoencoder
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten the input images
        data = data.view(data.size(0), -1)
        # Forward pass
        outputs = model(data)
        # Compute the loss
        loss = nn.BCELoss()(outputs, data)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the autoencoder
with torch.no_grad():
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (data, _) in enumerate(test_loader):
        # Flatten the input images
        data = data.view(data.size(0), -1)
        # Forward pass
        outputs = model(data)
        # Visualize the input and output images
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n], outputs[:n]])
        torchvision.utils.save_image(comparison.cpu(), f'reconstruction_{batch_idx+1}.png', nrow=n)
```