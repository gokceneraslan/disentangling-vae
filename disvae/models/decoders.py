"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn


class DecoderConv2D(nn.Module):
    def __init__(self, img_size, latent_dim=10, kernel_size=4, num_layers=6, stride=2, padding=1, hidden_dim=256, hidden_channels=32):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super().__init__()

        # Layer parameters
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (self.hidden_channels, img_size[1]//(2**num_layers), img_size[2]//(2**num_layers))
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=stride, padding=padding)
        self.convt_layers = nn.ModuleList()
        
        for i in range(num_layers):
            target_channels = self.hidden_channels if i < (num_layers-1) else n_chan
            convt_layer = nn.ConvTranspose2d(self.hidden_channels, target_channels, self.kernel_size, **cnn_kwargs)
            self.convt_layers.append(convt_layer)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers
        for i, convt_layer in enumerate(self.convt_layers):
            if i < (self.num_layers-1):
                x = torch.relu(convt_layer(x))
            else:
                # Sigmoid activation for final conv layer
                x = torch.sigmoid(convt_layer(x))
        return x


class CondDecoderConv2D(nn.Module):
    def __init__(self, img_size, cond_dim, latent_dim=10, kernel_size=4, num_layers=6, stride=2, padding=1, hidden_dim=256, hidden_channels=32):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super().__init__()

        # Layer parameters
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (self.hidden_channels, img_size[1]//(2**num_layers), img_size[2]//(2**num_layers))
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=stride, padding=padding)
        self.convt_layers = nn.ModuleList()
        
        for i in range(num_layers):
            target_channels = self.hidden_channels if i < (num_layers-1) else n_chan
            convt_layer = nn.ConvTranspose2d(self.hidden_channels, target_channels, self.kernel_size, **cnn_kwargs)
            self.convt_layers.append(convt_layer)

    def forward(self, *, z, y):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(torch.cat([z, y], dim=-1)))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers
        for i, convt_layer in enumerate(self.convt_layers):
            if i < (self.num_layers-1):
                x = torch.relu(convt_layer(x))
            else:
                # Sigmoid activation for final conv layer
                x = torch.sigmoid(convt_layer(x))
        return x


class DecoderFC(nn.Module):
    def __init__(self, output_dim, latent_dim=10, hidden_dim=256, output_activation='linear'):
        super(DecoderFC, self).__init__()

        self.output_activation = output_activation
        # Layer parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, z):
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))

        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x
    
    
class CondDecoderFC(nn.Module):
    def __init__(self, output_dim, cond_dim, latent_dim=10, hidden_dim=256, output_activation='linear'):
        super().__init__()

        self.output_activation = output_activation
        # Layer parameters
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, *, z, y):
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(torch.cat([z, y], dim=-1)))
        x = torch.relu(self.lin2(x))

        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x

