"""
Module containing the main VAE class.
"""

from abc import ABC, abstractmethod, abstractproperty

import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import EncoderConv2D, EncoderFC
from .decoders import DecoderConv2D, DecoderFC


class VAE(ABC):
    @abstractproperty
    def encoder(self):
        pass

    @abstractproperty
    def decoder(self):
        pass    
        
    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class VAEConv2D(VAE, nn.Module):
    def __init__(self, img_size, latent_dim=10, kernel_size=4, stride=2, padding=1, hidden_dim=256, hidden_channels=32):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()

        self.img_size = self.input_size = img_size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.hidden_channels = hidden_channels
        
        self._encoder = EncoderConv2D(img_size,
                                      latent_dim=latent_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      hidden_dim=hidden_dim, 
                                      hidden_channels=hidden_channels)

        self._decoder = DecoderConv2D(img_size,
                                     latent_dim=latent_dim,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     hidden_dim=hidden_dim, 
                                     hidden_channels=hidden_channels)
        
        self.reset_parameters()

        
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder


class VAEFC(VAE, nn.Module):
    def __init__(self, input_size, latent_dim=10, hidden_dim=256):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self._encoder = EncoderFC(input_size,
                                  latent_dim=latent_dim,
                                  hidden_dim=hidden_dim)

        self._decoder = DecoderFC(input_size,
                                  latent_dim=latent_dim,
                                  hidden_dim=hidden_dim)
        
        self.reset_parameters()

        
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder