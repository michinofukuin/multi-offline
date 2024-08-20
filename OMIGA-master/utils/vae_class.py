import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributions as td


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(self, state_dim, action_dim, latent_dim, max_action=None, hidden_dim=750, dropout=0.0):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def elbo_loss(self, state, action, beta, num_samples=1):
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = self.encode(state, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = self.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss

    def iwae_loss(self, state, action, beta, num_samples=10):
        ll = self.importance_sampling_estimator(state, action, beta, num_samples)
        return -ll

    def elbo_estimator(self, state, action, beta, num_samples=1):
        mean, std = self.encode(state, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = math.sqrt(beta / 4)

        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).sum(-1)
        elbo = log_pxz.sum(-1).mean(-1) - KL_loss
        return elbo

    def importance_sampling_estimator(self, state, action, beta, na, num_samples=500):
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper, we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(1,num_samples,1)  # [B x S x D]
        std_enc = std.repeat(1,num_samples,1)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(1,num_samples,1)  # [B x S x C]
        action = action.repeat(1,num_samples,1)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = math.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z).to(self.device)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z).to(self.device)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action).to(self.device)
        if na==1:
            w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
            lll = torch.logsumexp(w, dim=-1, keepdim=True) - math.log(num_samples)
        else:
            log_pxz = torch.chunk(log_pxz, na, dim=1)
            log_pz = torch.chunk(log_pz, na, dim=1)
            log_qzx = torch.chunk(log_qzx, na, dim=1)
            lll=[]
            for i in range(len(log_pxz)):
                l_pxz = log_pxz[i]
                l_pz = log_pz[i]
                l_qzx = log_qzx[i]
                w = l_pxz.sum(-1) + l_pz.sum(-1) - l_qzx.sum(-1)
                ll = torch.logsumexp(w, dim=-1, keepdim=True) - math.log(num_samples)
                lll.append(ll)
            lll = torch.cat(lll, dim=-1)
        return lll

    def encode(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], state.shape[1], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        if self.max_action is not None:
            return self.max_action * torch.tanh(self.d3(a))
        else:
            return self.d3(a)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']