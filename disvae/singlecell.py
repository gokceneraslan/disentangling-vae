import argparse
import logging
import sys
import os
from configparser import ConfigParser

import torch
from torch import optim

import seaborn as sns
import numpy as np
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt

from pathlib import Path


from disvae import Trainer, Evaluator, VAEFC, CondVAEFC
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)

from utils.datasets import DisentangledDataset
from utils.visualize import GifTraversalsTraining

from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp


RES_DIR = "single_cell_results"
LOG_LEVELS = list(logging._levelToName.values())


class AnndataDataset(DisentangledDataset):
    files = {"train": "."}

    def __init__(self, adata, categorical_vars=None, unwanted_vars=None, scale_factor=1.0, **kwargs):
        super().__init__(root='.', **kwargs)
        self.adata = adata
        self.imgs = self.adata
        
        if categorical_vars is not None:
            assert np.all(pd.Series(categorical_vars).isin(adata.obs.columns))
            assert np.all([adata.obs[x].dtype.name == 'category' for x in categorical_vars])
            
            if unwanted_vars is None:
                self.mask = np.ones(len(categorical_vars))
            else:
                self.mask = 1.0 - np.isin(categorical_vars, unwanted_vars)
        else:
            self.mask = None
                
        self.unwanted_vars = unwanted_vars
        self.categorical_vars = categorical_vars
        self.scale_factor = scale_factor
        self.y = self.get_conditional_vectors(mask=False)

    def download(self):
        pass

    def __getitem__(self, idx):
        batch = self.adata.X[idx]
        return batch, (0 if self.categorical_vars is None else self.y[idx])
    
    def get_conditional_vectors(self, mask=False):
        if self.categorical_vars is None:
            return None
        
        if mask:
            assert self.mask is not None
            assert len(self.mask) == len(self.categorical_vars)
            mat = np.hstack([pd.get_dummies(self.adata.obs[x]).values*m for x, m in zip(self.categorical_vars, self.mask)])
        else:
            mat = np.hstack([pd.get_dummies(self.adata.obs[x]).values for x in self.categorical_vars])

        return mat.squeeze().astype('float32') * self.scale_factor

def fit_single_cell(adata, experiment,
                categorical_vars = None,
                unwanted_vars = None,
                pretrained_model = False,
                scale_factor = 1.0,

                # Training options
                epochs = 100,
                batch_size = 128,
                lr = 5e-4,
                checkpoint_every = 10,

                # Model Options
                output_activation = 'linear',
                hidden_dim = 256,
                latent_dim = 20,
                num_layers = 1,
                rec_dist = "gaussian",
                rec_coef = 1.0,
                reg_anneal = 10000,
 
                progress_bar = True,
                cuda = True,
                seed = 1234,
                log_level = "info",
                shuffle=True,
                pin_memory=True,

                # btcvae Options
                btcvae_A = 1,
                btcvae_G = 1,
                btcvae_B = 6,
                eval_batchsize = 1000):
    
    default_config = locals()
    
    if sp.issparse(adata.X): adata.X = adata.X.A

    ds = AnndataDataset(adata, categorical_vars=categorical_vars, unwanted_vars=unwanted_vars, scale_factor=scale_factor)
    train_loader = DataLoader(ds,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory)

    logger = logging.getLogger('vae')
    logger.info("Train VAE with {} samples".format(len(train_loader.dataset)))

    if categorical_vars is None:
        model = VAEFC(adata.n_vars, latent_dim=latent_dim, num_layers=num_layers, 
                      output_activation=output_activation, hidden_dim=hidden_dim) 
    else:
        cond_dim = sum([len(adata.obs[x].cat.categories) for x in categorical_vars])
        model = CondVAEFC(adata.n_vars, latent_dim=latent_dim, cond_dim=cond_dim, 
                          num_layers=num_layers, output_activation=output_activation, hidden_dim=hidden_dim)

    device = 'cuda' if cuda else 'cpu'
    model = model.to(device)

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)  # make sure trainer and viz on same device

    loss_f = get_loss_f(loss_name='btcvae', recons_coef=rec_coef,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **default_config)

    exp_dir = os.path.join(RES_DIR, experiment)

    if not pretrained_model:
        create_safe_directory(exp_dir, logger=logger)
        trainer = Trainer(model, optimizer, loss_f,
                          conditional=(categorical_vars is not None),
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=progress_bar)

        trainer(train_loader,
                epochs=epochs,
                checkpoint_every=checkpoint_every)
    else:
        trainer = None
        model = load_model(exp_dir)

    adata = forward_pass_in_batch(ds, model, get_corrected=(categorical_vars is not None))
    adata.write(exp_dir + '/adata.h5ad')
    
    return adata, model, trainer, train_loader


def forward_pass_in_batch(dataset, model, batch_size=2048, device='cpu', get_corrected=False):
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    recons_list = []
    mu_list  = []
    var_list = []
    samples_list = []
    
    model.eval()
    model = model.to(device)

    for i, (x, y) in enumerate(trainloader):
        if dataset.categorical_vars is not None:
            recons, (mu, var), samples = model.forward(x.to(device), y.to(device))
        else:
            recons, (mu, var), samples = model.forward(x.to(device))

        recons_list.append(recons.cpu().detach().numpy())
        mu_list.append(mu.cpu().detach().numpy())
        var_list.append(var.cpu().detach().numpy())
        samples_list.append(samples.cpu().detach().numpy())

    recons_list = np.vstack(recons_list)
    mu_list = np.vstack(mu_list)
    var_list = np.vstack(var_list)
    samples_list = np.vstack(samples_list)
    
    dataset.adata.obsm['X_vae_samples'] = samples_list
    dataset.adata.obsm['X_vae_mean'] = mu_list
    dataset.adata.obsm['X_vae_var'] = var_list
    dataset.adata.layers['VAE'] = recons_list
    
    if get_corrected:
        y = dataset.get_conditional_vectors(mask=True)
        y = torch.from_numpy(y).to(device)
        
        recons = model.decoder(z=torch.from_numpy(mu_list).to(device), y=y)
        dataset.adata.layers['VAE_corrected'] = recons.cpu().detach().numpy()
        
    return dataset.adata
    

def plot_correlation(mat, **kwargs):
    cmap = sns.color_palette("RdBu_r", 50)
    mat = mat.reshape(-1, np.prod(mat.shape[1:])).T
    with sns.plotting_context("notebook", font_scale=0.5):
        sns.clustermap(np.corrcoef(mat), cmap=cmap, vmin=-1, vmax=1, **kwargs)
        
        
def plot_losses(exp_dir, type='per_dim'):
    
    losses = pd.read_csv(Path(exp_dir) / 'train_losses.log')

    if type == 'per_dim':
        df = losses[losses.Loss.str.startswith('kl_loss_')]
        return (p9.ggplot(df, p9.aes(x='Epoch', y='Value', color='Loss')) + 
                p9.geom_point() + p9.geom_line() + p9.coord_trans(y='log') + p9.theme_minimal())
    else:
        df = losses[(~losses.Loss.str.startswith('kl_loss_')) & (losses.Loss != 'loss') & (losses.Loss != 'tc_loss')]
        return (p9.ggplot(df, p9.aes(x='Epoch', y='Value', color='Loss')) + 
                p9.geom_point() + p9.geom_line()+ p9.coord_trans(y='log') + p9.theme_minimal())
    
    
from scipy.stats import gaussian_kde

def plot_marginals(adata, rep='X_vae_mean'):

    latent_dim = adata.obsm[rep].shape[1]
    f, axs = plt.subplots(np.ceil(latent_dim/5).astype(int), 5, figsize=(16, 4))
    axs = axs.flatten()
    for i in range(latent_dim):
        x_mean = adata.obsm[rep][:, i]

        axs[i].hist(x_mean, bins=100, alpha=0.2, density=True)
        sx = sorted(x_mean)
        axs[i].plot(sx, gaussian_kde(sx)(sx))

    plt.subplots_adjust(hspace=0.4, wspace=0.4)