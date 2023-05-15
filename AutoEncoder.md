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

# elaborate how encoder is constructed?
In an autoencoder, the encoder is responsible for mapping the input data into a latent space representation. This is achieved through a series of linear and non-linear transformations applied to the input data. In Pytorch, the encoder can be constructed using the nn.Module class. Here is an example of an encoder with a single hidden layer:


```
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        z = self.fc2(hidden)
        return z
 ```
 
 In this example, the encoder takes an input of size input_dim and maps it to a hidden layer of size hidden_dim using a linear transformation defined by self.fc1. The output of this transformation is passed through a rectified linear unit (ReLU) activation function defined by self.relu. The output of the ReLU function is then passed through another linear transformation defined by self.fc2, which maps the hidden representation to a latent space representation of size latent_dim.

The forward method of the encoder takes an input x and applies the sequence of transformations defined in the __init__ method to produce the encoded latent representation z.
# What is VAE , how it different than AE? 
VAE stands for Variational Autoencoder, which is a type of neural network architecture that is designed to generate new data samples similar to the ones it was trained on. It is a type of autoencoder that is based on probabilistic modeling, which means it learns the distribution of the data and then generates new samples from that distribution.

The key difference between VAE and a traditional autoencoder (AE) is that VAEs learn a probability distribution over the latent space, whereas AEs map input data to a fixed-dimensional encoding. In other words, VAEs are generative models that learn the underlying structure of the data and can create new samples from that structure, whereas AEs are usually used for encoding data and not for generating new samples.

The encoding and decoding process in a VAE is similar to that of an AE, but with the addition of a latent variable z. The encoder maps the input data x to a probability distribution over z, and the decoder maps a sample from that distribution back to the original input space. During training, the VAE minimizes the reconstruction error between the original input and the decoded output, as well as the Kullback-Leibler divergence between the learned distribution and a prior distribution. This encourages the VAE to learn a compact and smooth latent representation of the data that can be used for generative purposes.

In summary, the main difference between VAE and AE is that VAEs learn a probability distribution over the latent space, which enables them to generate new samples, whereas AEs simply learn an encoding of the input data.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(z))
        return x

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

```
This code defines a VAE with an encoder and decoder, and includes the reparameterization trick to enable backpropagation through the stochastic sampling process. The encoder takes in an input of size input_size, applies a linear layer followed by a ReLU activation to get to a hidden layer of size hidden_size, and then produces two output layers representing the mean and log variance of the latent code, both of size latent_size. The decoder takes in a latent code of size latent_size, applies a linear layer followed by a ReLU activation to get to a hidden layer of size hidden_size, and then produces an output of size input_size through a sigmoid activation.

During training, the VAE minimizes a loss function consisting of a reconstruction loss and a KL divergence loss between the learned distribution of latent codes and a prior distribution (usually a standard normal). The reconstruction loss encourages the VAE to learn a good reconstruction of the input, while the KL divergence loss encourages the VAE to learn a latent space that is more disentangled and interpretable.

```
import torch
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)

# define loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# set up optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# train the model for multiple epochs
for epoch in range(1, epochs + 1):
    train(epoch)

```
In this example, we define a VAE with an encoder and decoder architecture, similar to the one we discussed earlier. The encode function takes the input image and maps it to the mean and variance of the latent distribution. The reparameterize function takes the mean and variance and samples a latent vector from the distribution. The decode function takes the latent vector and maps it to the reconstructed image.

The forward function is the main function of the model, which takes the input image, passes it through the encoder to get the mean and variance, samples a latent vector, and then passes it through the decoder to get the reconstructed image.

The loss_function takes the reconstructed image, the original image, the mean, and the variance and computes the reconstruction loss and the Kullback-Leibler (KL) divergence loss. The reconstruction loss is the binary cross-entropy between the reconstructed image and the original image. 






 
