"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn

class EncoderConv2D(nn.Module):
    def __init__(self, img_size, latent_dim=10, kernel_size=4, num_layers=6, stride=2, padding=1, hidden_dim=256, hidden_channels=32):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderConv2D, self).__init__()

        # Layer parameters
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_layers = num_layers

        # Shape required to start transpose convs
        assert len(img_size) == 3, 'img_size must be in CHW layout'
        assert (img_size[1] % 2) == 0, 'img height must be multiple of 2'
        assert (img_size[2] % 2) == 0, 'img width must be multiple of 2'
        assert (img_size[1]//(2**num_layers)) > 0, f'image height is too small for {num_layers} layers'
        assert (img_size[2]//(2**num_layers)) > 0, f'image width is too small for {num_layers} layers'
        
        # calculate the image size after convolutions
        self.reshape = (self.hidden_channels, img_size[1]//(2**num_layers), img_size[2]//(2**num_layers))
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=stride, padding=padding)
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_channel = n_chan if i == 0 else self.hidden_channels
            conv = nn.Conv2d(in_channel, self.hidden_channels, self.kernel_size, **cnn_kwargs)
            self.conv_layers.append(conv)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        for conv_layer in self.conv_layers:
            x = torch.relu(conv_layer(x))
        
        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class EncoderFC(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dim=256):
        super(EncoderFC, self).__init__()

        # Layer parameters
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Fully connected layers
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        x = torch.relu(self.lin1(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar
