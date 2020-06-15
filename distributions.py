import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical

TINY = 1e-8
class Gaussian():
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self._dim * 2

    @property
    def effective_dim(self):
        return self._dim

    def activate(self, distribution, fixed_std=False):
        size = distribution.size(1)//2
        mean = distribution[:,:size]
        if not fixed_std:
            std  = torch.exp(distribution[:,size:])
        else:
            std  = Variable(torch.ones((distribution.size(0),size))).to(distribution.device)
        return dict(mean=mean, std=std)

    def log_li(self, x_var, dist_info):
        mean   = dist_info["mean"]
        std    = dist_info["std"]
        epsilon = (x_var - mean) / (std + TINY)

        pi = Variable(torch.ones(1) * np.pi).to(x_var.device)
        #std should be squared for correct loglikelihood
        #logli = - 0.5 * torch.log(2 * pi) - torch.log(torch.pow(std,2) + TINY) - 0.5 * torch.pow(epsilon,2)
        logli = - 0.5 * torch.log(2 * pi) - torch.log(std + TINY) - 0.5 * torch.pow(epsilon,2)

        return logli.sum(1)

    def prior_dist_info(self, batch_size,device):
        mean   = torch.zeros([batch_size, self.dim]).to(device)
        std    = torch.ones([batch_size, self.dim]).to(device)
        return dict(mean=mean, std=std)

    def log_li_prior(self, x_var):
        b_size = x_var.size(0)
        prior_info = self.prior_dist_info(b_size,x_var.device)
        return self.log_li(x_var,prior_info)

    def sample(self, dist_info):
        mean    = dist_info["mean"]
        std     = dist_info["std"]
        epsilon = Variable(torch.randn(mean.shape)).to(mean.device)
        return mean + epsilon * std

    def kl_with_prior(self,dist_info):  
        raise "Not implemented"

    def sample_prior(self, batch_size, device='cuda:0'):
        return torch.randn((batch_size,self.dim)).to(device)


class Categorical():
    def __init__(self, dim):
        self._dim = dim
        self.softmax = nn.Softmax(1)

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    @property
    def effective_dim(self):
        return 1

    def log_li(self, x_var, dist_info):
        prob = dist_info["prob"]
        return torch.sum(torch.log(prob + TINY) * x_var,1)

    def log_li_prior(self,x_var):
        b_size = x_var.size(0)
        prior = self.prior_dist_info(b_size,x_var.device)
        return self.logli(x_var,prior)

    def prior_dist_info(self, batch_size,device='cuda:0'):
        prob = torch.ones([batch_size, self.dim]) / (self.dim * 1.)
        return dict(prob=prob.to(device))

    def compute_KL(self, p, q):
        """
        :param p: left dist info
        :param q: right dist info
        :return: KL(p||q)
        """
        p_prob = p["prob"]
        q_prob = q["prob"]
        kl = torch.sum(p_prob * (torch.log(p_prob + TINY) - torch.log(q_prob + TINY)), 1)
        return kl

    def sample(self, dist_info):
        prob = dist_info["prob"]
        sampler = OneHotCategorical(prob)
        return sampler.sample()

    def activate(self, flat_dist):
        return dict(prob=self.softmax(flat_dist))

    def logits_to_onehot(self,logits):
        y = torch.argmax(logits,1)
        eye = torch.eye(logits.size(-1)).to(logits.device)
        return eye[y]

    def entropy(self, dist_info):
        prob = dist_info["prob"]
        return -torch.sum(prob * torch.log(prob + TINY),1)

    def sample_prior(self, batch_size, device='cuda:0'):
        return self.sample(self.prior_dist_info(batch_size, device))

    @property
    def dist_info_keys(self):
        return None


class Gumbel():
    def __init__(self, dim):
        self._dim = dim 
        self.softmax = nn.Softmax(1)

    def sample_gumbel(self, logits, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits)
        return self.softmax( y / temperature)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
        """
        prob = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            sampler = OneHotCategorical(prob)
            prob = sampler.sample()
        return prob
