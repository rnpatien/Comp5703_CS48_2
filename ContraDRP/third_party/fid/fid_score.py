#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evaluate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

try:
    from third_party.fid.inception import InceptionV3
except ImportError:
    from inception import InceptionV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def compute_stats_from_G(G, model, size, batch_size):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    model.eval()

    n_batches = size // batch_size

    predictions = []
    for _ in tqdm(range(n_batches)):
        latent_samples = G.sample_latent(batch_size)

        with torch.no_grad():
            images = G(latent_samples)
            pred = model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        predictions.append(pred.view(images.size(0), -1).cpu())

    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.cpu().data.numpy()

    mu = np.mean(predictions, axis=0)
    sigma = np.cov(predictions, rowvar=False)

    return mu, sigma

def compute_stats_from_Images(imgs, model):
    model.eval()
    batch_size=imgs.shape[0]
    imgsTens=torch.tensor(imgs,device=device)
    predictions = []
    pred = model(imgsTens)[0]
    predictions.append(pred.view(imgsTens.size(0), -1).cpu())
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.cpu().data.numpy()
    mu = np.mean(predictions, axis=0)
    sigma = np.cov(predictions, rowvar=False)
    return mu, sigma


def compute_stats_from_dataloader(dataloader, model):
    """Calculates the FID of two paths"""
    model.eval()

    predictions = []
    for images, labels in tqdm(dataloader):
        images = images.to(device)

        with torch.no_grad():
            pred = model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        predictions.append(pred.view(images.size(0), -1).cpu())

    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.cpu().data.numpy()

    mu = np.mean(predictions, axis=0)
    sigma = np.cov(predictions, rowvar=False)

    return mu, sigma


def fid_score(path_base, G, size=10000, batch_size=50, model=None, dims=2048):
    """Calculates the FID between G and a pre-computed stats"""
    if not os.path.exists(path_base):
        raise RuntimeError('Invalid path: %s' % path_base)

    computed_stats_base = np.load(path_base)
    m1, s1 = computed_stats_base['mu'][:], computed_stats_base['sigma'][:]

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

    m2, s2 = compute_stats_from_G(G, model, size=size, batch_size=batch_size)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def fid_score_img(path_base,imgs,model, dims=2048):
    if not os.path.exists(path_base):
        raise RuntimeError('Invalid path: %s' % path_base)

    computed_stats_base = np.load(path_base)
    m1, s1 = computed_stats_base['mu'][:], computed_stats_base['sigma'][:]

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

    m2, s2 = compute_stats_from_Images(imgs, model)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, m1, s1, m2, s2  

def precompute_stats(dataset, save_path, model=None, dims=2048):
    from datasets import get_dataset_ref
    ref_dataset = get_dataset_ref(dataset)
    dataloader = DataLoader(ref_dataset, shuffle=False, batch_size=50)

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

    mu, sigma = compute_stats_from_dataloader(dataloader, model)
    np.savez(save_path, mu=mu, sigma=sigma)
