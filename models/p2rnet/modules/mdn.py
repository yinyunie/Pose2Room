#  Gaussian Mixture Model for probabilistic regression.
#  Copyright (c) 10.2021. Yinyu Nie
#  License: MIT

import math
import torch
from torch import nn
from torch.autograd import Variable
from typing import Optional
from models.p2rnet.modules.sub_modules import SingleConv
from torch.distributions.bernoulli import Bernoulli

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)

class MixtureDensityHead(nn.Module):
    def __init__(self, config, **kwargs):
        super(MixtureDensityHead, self).__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.pi = SingleConv(self.hparams.input_dim, self.hparams.num_gaussian, kernel_size=1, order='c', padding=0, ndim=1)

        self.log_sigma = nn.Parameter(torch.zeros(self.hparams.num_gaussian, self.hparams.out_dim))
        self.mu = nn.Parameter(self.hparams.mu_bias_init)

    def forward(self, x):
        pi = self.pi(x)
        pi = torch.sigmoid(pi)
        return pi

    def sample(self, num_samples, n_batch):
        """Draw samples from a MoG."""
        # get sigma
        sigma = torch.exp(self.log_sigma)  # 1*c*d
        sigma = sigma.unsqueeze(0).unsqueeze(2).repeat(n_batch, 1, num_samples, 1)

        # get mu
        mu = self.mu.unsqueeze(0).unsqueeze(2).repeat(n_batch, 1, num_samples, 1)

        # sample
        sample = Variable(mu.data.new(mu.size()).normal_())
        sample = sample * sigma + mu

        return sample

    def generate_samples(self, pi, n_samples=None, sample_pi=False):
        if n_samples is None:
            n_samples = self.hparams.n_samples

        n_batch, _, length = pi.size()

        pi_r = pi.transpose(1,2).contiguous().view(n_batch*length, -1)

        samples = self.sample(n_samples, pi_r.size(0))

        if sample_pi:
            bernoulli_dist = Bernoulli(pi_r)
            pi_r = bernoulli_dist.sample((n_samples,))
            pi_r = pi_r.permute(1, 2, 0)
            pi_r = pi_r.unsqueeze(-1).repeat(1, 1, 1, self.hparams.out_dim)
        else:
            pi_r = pi_r.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, n_samples, self.hparams.out_dim)

        samples = torch.sum(samples * pi_r, dim=1)

        samples = samples.view(n_batch, length, n_samples, -1)
        samples = samples.transpose(1,3).contiguous()

        return samples

    def generate_point_predictions(self, pi, n_samples=None, sample_pi=False):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, n_samples, sample_pi=sample_pi)
        if self.hparams.central_tendency == "mean":
            y_hat = torch.mean(samples, dim=2)
        elif self.hparams.central_tendency == "median":
            y_hat = torch.median(samples, dim=2).values
        else:
            raise NotImplementedError
        return y_hat

    def get_mean(self, pi):
        n_batch, _, length = pi.size()
        pi_r = pi.transpose(1,2).contiguous().view(n_batch*length, -1)

        # get mu
        mu = self.mu.unsqueeze(0).repeat(pi_r.size(0), 1, 1)

        pi_r = pi_r.unsqueeze(-1).repeat(1, 1, self.hparams.out_dim)

        samples = torch.sum(mu * pi_r, dim=1)

        samples = samples.view(n_batch, length, -1)
        samples = samples.transpose(1,2).contiguous()

        return samples

class BaseMDN(nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseMDN, self).__init__()
        self.config = config

    def forward(self, x):
        x = self.unpack_input(x)
        x = self.backbone(x)
        pi = self.mdn(x)
        return pi

    def predict(self, x):
        pi = self.forward(x)
        return self.mdn.generate_point_predictions(pi)

    def generate(self, x, return_pi=False, multi_modes=False, n_samples=10):
        pi = self.forward(x)
        if multi_modes:
            pred = self.mdn.generate_point_predictions(pi, n_samples=n_samples, sample_pi=True)
        else:
            pred = self.mdn.get_mean(pi)
        if return_pi:
            return pred, pi
        else:
            return pred

    def sample(self, x, n_samples: Optional[int] = None, ret_model_output=False):
        pi = self.forward(x)
        samples = self.mdn.generate_samples(pi, n_samples)
        if ret_model_output:
            return samples, pi
        else:
            return samples

    def test_step(self, x, y):
        pi = self(x)
        y_hat = self.mdn.generate_point_predictions(pi)
        return y_hat, y


class CategoryEmbeddingMDN(BaseMDN):
    def __init__(self, config, **kwargs):
        super(CategoryEmbeddingMDN, self).__init__(config, **kwargs)
        self.hparams = config
        self._build_network()

    def _build_network(self):
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Backbone
        self.backbone = SingleConv(self.hparams.continuous_dim,
                                   self.hparams.hidden_dim, kernel_size=1, order='cbr', padding=0, ndim=1)
        # Adding the last layer
        self.hparams.mdn_config.update(input_dim=self.hparams.hidden_dim)
        self.mdn = MixtureDensityHead(self.hparams.mdn_config)

    def unpack_input(self, x):
        if self.hparams.batch_norm_continuous_input:
            x = self.normalizing_batch_norm(x)
        return x

if __name__ == '__main__':
    import numpy as np
    class Struct(object):
        def __init__(self, **kwargs):
            self.update(**kwargs)

        def update(self, **kwargs):
            for key, val in kwargs.items():
                setattr(self, key, val)

    out_dim = 3
    num_gaussian = 8
    n_bins_per_dim_center = int(num_gaussian ** (1 / 3))

    bins_per_dim_center = np.linspace(-1, 1, n_bins_per_dim_center)
    init_center = np.array(np.meshgrid(bins_per_dim_center, bins_per_dim_center, bins_per_dim_center)).reshape(3, -1)
    init_center = torch.from_numpy(init_center.T).contiguous()

    mdn_config = Struct(num_gaussian=num_gaussian, out_dim=out_dim, mu_bias_init=init_center, n_samples=100,
                        central_tendency='mean')
    config = Struct(out_dim=out_dim, continuous_dim=128, batch_norm_continuous_input=False,
                    hidden_dim=128, mdn_config=mdn_config)
    gmm_model = CategoryEmbeddingMDN(config)
    x = torch.randn((4, 128, 12))
    y = torch.randn((4, out_dim, 12))
    gmm_model.test_step(x, y)
