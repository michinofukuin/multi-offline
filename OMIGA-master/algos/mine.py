import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
from networks.network import Actor, V_critic, Q_critic, MixNet
import torch.nn.init as init
from copy import deepcopy
torch.autograd.set_detect_anomaly(True)

class mine(object):
    def __init__(self, observation_spec, action_spec, num_agent, eval_env, config):
        self._gamma = config['gamma']
        self._tau = config['tau']
        self.adap_total_alpha_start= 2
        self._hidden_sizes = config['hidden_sizes']
        self._mix_hidden_sizes = config['mix_hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self.is_mix = config["mix"]
        self._grad_norm_clip = config['grad_norm_clip']
        self._num_agent = num_agent
        self._device = config['device']
        self._quality = config['quality']
        self._eval_env = eval_env
        self._iteration = 0
        self._optimizers = dict()
        self._o_dim = observation_spec
        self._v_network = V_critic(observation_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._v_target_network = copy.deepcopy(self._v_network)
        self._optimizers['v'] = torch.optim.Adam(self._v_network.parameters(), self._lr)
        self._q_network = Q_critic(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._q_target_network = copy.deepcopy(self._q_network)
        self._mix_network = MixNet(observation_spec, action_spec, num_agent, self._mix_hidden_sizes, self._device).to(self._device)
        self._mix_target_network = copy.deepcopy(self._mix_network)
        self._q_param = list(self._q_network.parameters()) + list(self._mix_network.parameters())
        self._optimizers['q'] = torch.optim.Adam(self._q_param, self._lr)
        self._policy_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)
        self.log_alpha = nn.Parameter(torch.ones(self._num_agent, device=self._device) * self.adap_total_alpha_start)
        self.target_value = self.log_alpha.data
        # init.normal_(self.log_alpha, mean=self.adap_total_alpha_start, std=0.1)
        self._optimizers['alpha'] = torch.optim.Adam([self.log_alpha], lr=config['clr'])
        # self.hi = (config['h_tot'] * torch.ones(self._num_agent)).to(self._device)
        vv = [-0.1, -0.1, -0.1]
        self.hi = torch.tensor(vv).to(self._device)
        self.cql_sample_noise_level = 0.2
        self.adap_total_alpha_tau = config['adap_total_alpha_tau']
        self.behaviour_network = Actor(observation_spec, action_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._optimizers['behaviour'] = torch.optim.Adam(self.behaviour_network.parameters(), self._lr)
        self._v_behaviour = V_critic(observation_spec, num_agent, self._hidden_sizes, self._device).to(self._device)
        self._v_target_behaviour = copy.deepcopy(self._v_behaviour)
        self._optimizers['v_behaviour'] = torch.optim.Adam(self._v_behaviour.parameters(), self._lr)
     
    def train_behaviour(self, o, a, r, o_next, mask):
        loss_result = {}
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
        policy_loss = -(self.behaviour_network.get_log_density(o_with_id, a)).mean()
        self._optimizers['behaviour'].zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.behaviour_network.parameters(), self._grad_norm_clip)
        self._optimizers['behaviour'].step()
        # td_v = self._v_target_behaviour(o_next_with_id)
        # v_now = self._v_behaviour(o_with_id)
        # v_loss = torch.mean(F.mse_loss(r + self._gamma * mask * td_v.detach(), v_now))
        # self._optimizers['v_behaviour'].zero_grad()
        # v_loss.backward()
        # nn.utils.clip_grad_norm_(self._v_behaviour.parameters(), self._grad_norm_clip)
        # self._optimizers['v_behaviour'].step()
        # for param, target_param in zip(self._v_behaviour.parameters(), self._v_target_behaviour.parameters()):
        #     target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        loss_result.update({'policy_loss':policy_loss})
        return loss_result

    def policy_loss(self, exp_a, a, o_with_id, result={}):
        log_probs = self._policy_network.get_log_density(o_with_id, a)
        policy_loss = -(exp_a * log_probs).mean()
        result.update({'policy_loss': policy_loss})
        return result

    def q_loss(self, o_with_a_id, s, o_next_with_id, s_next, r, mask, result={}):
        q_values = self._q_network(o_with_a_id)
        if self.is_mix:
            w, b = self._mix_network(s)
            q_total = (w * q_values).sum(dim=-2) + b.squeeze(dim=-1)
        else:
            q_total = 1
        v_next_target_values = self._v_target_network(o_next_with_id)
        if self.is_mix:
            w_next_target, b_next_target = self._mix_target_network(s_next)
            v_next_target_total = (w_next_target * v_next_target_values).sum(dim=-2) + b_next_target.squeeze(dim=-1)
        else:
            v_next_target_total = 1
        expected_q_total = r + self._gamma * mask * v_next_target_total.detach()
        q_loss = ((q_total - expected_q_total.detach()) ** 2).mean()
        result.update({'q_loss': q_loss, 'q_total': q_total.mean()})
        return result
    
    def compute_contribution(self, exp_a, o_with_id, a, pi, s):
        contributions = []
        for i in range(self._num_agent):
            o_i = o_with_id[:, i, :].unsqueeze(1)
            a_i = a[:, i, :].unsqueeze(1)
            old_pi = self.little_policy_loss(deepcopy(pi), exp_a[:, i], a_i, o_i)
            with torch.no_grad():
                loss_1 = ((old_pi.get_log_density(o_i, a_i) - pi.get_log_density(o_i, a_i)) * exp_a[:, i].unsqueeze(-1)).mean()
                action_pi = old_pi.get_deterministic_action(o_i)
                kl_1 = torch.exp(-self.vae.importance_sampling_estimator(o_i, action_pi, 0.5, 1, num_samples=5)).mean()
                action_pi_2 = pi.get_deterministic_action(o_i)
                kl_2 = torch.exp(-self.vae.importance_sampling_estimator(o_i, action_pi_2, 0.5, 1, num_samples=5)).mean()
                contributions.append(-(loss_1) / (kl_1 - kl_2 + 1e-3))
        contributions = torch.tensor(contributions).to(self._device)
        mean_contributions = torch.mean(contributions)
        std_contributions = torch.std(contributions) + 1e-8
        contributions = (contributions - mean_contributions) / std_contributions
        return F.softmax(contributions, dim=-1)

    def little_policy_loss(self, policy, exp_a, a, o_with_id):
        small_step_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
        log_probs = policy.get_log_density(o_with_id, a)
        policy_loss = (- torch.exp(log_probs) * log_probs).mean()
        small_step_optimizer.zero_grad()
        policy_loss.backward()
        small_step_optimizer.step()
        return policy

    def check_nan_inf(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            print(f"{tensor_name} is nan")
        if torch.isinf(tensor).any():
            print(f"{tensor_name} is inf")

    def alpha_loss_3(self, h, exp_a, result={}):
        loss_1 = h.detach() * self.log_alpha.exp()
        loss_2 = self.log_alpha.exp() * torch.log(torch.sum(exp_a, dim=0))
        loss_1 = loss_1.mean()
        loss = -loss_1 + loss_2.mean()
        result.update({
            'alpha_loss': loss,
        })
        return result
    
    def alpha_loss_2(self, h, exp_a, result={}):
        loss_1 = h.detach() * self.log_alpha.exp()
        loss_2 = self.log_alpha.exp() * torch.log(torch.mean(exp_a, dim=0))
        loss = (-loss_1 + loss_2).mean()
        result.update({
            'alpha_loss': loss,
        })
        return result
    
    def alpha_loss(self, h, o_with_id, result={}):
        al_loss= 0.0
        zz =[]
        for i in range (self._num_agent):
            o_i = o_with_id[:, i, :].unsqueeze(1)
            # a_i = a[:, i, :].unsqueeze(1)
            dis_1 = self._policy_network.get_dis(o_i)
            dis_2 = self.behaviour_network.get_dis(o_i)
            kl = D.kl_divergence(dis_2, dis_1)
            zz.append(kl.mean())
            # gap = torch.sum(h)
            # h_cl = torch.clamp(h, min= 0.2 * gap, max = 0.8 * gap)
            h_cl = h
            al = (self.log_alpha[i].exp()) * (-kl.mean().detach() - h_cl[i].detach())
            # al = (self.log_alpha[i].exp()) * (kl.mean().detach() + h_cl[i].detach())
            self.check_nan_inf(kl, 'kl')
            self.check_nan_inf(h[i], 'h')
            al_loss = al_loss + al.mean()
        loss = al_loss / self._num_agent
        result.update({
            'alpha_loss': loss,
            'kl1' : zz[0],
            'kl2' : zz[1],
            'kl3' : zz[2],
        })
        return result

    def v_loss(self, z, w_target, v_values, result={}):
        max_z = torch.max(z)
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).to(self._device), max_z)
        cc = self.log_alpha.exp().view(1, self._num_agent, 1).detach()
        if self.is_mix:
            v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target * v_values / cc
        else:
            v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * v_values / self.alpha
        v_loss = v_loss.mean()
        result.update({'v_loss': v_loss})
        return result

    def train_step(self, o, s, a, r, mask, s_next, o_next, a_next):
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        o_with_a_id = torch.cat((o, a, one_hot_agent_id), dim=-1)
        o_next_with_id = torch.cat((o_next, one_hot_agent_id), dim=-1)
        loss_result = self.q_loss(o_with_a_id, s, o_next_with_id, s_next, r, mask, result={})
        q_target_values = self._q_target_network(o_with_a_id)
        w_target, b_target = self._mix_target_network(s)
        v_values = self._v_network(o_with_id)
        v_values.clamp_(min=-10, max=50)
        if self.is_mix:
            cc = self.log_alpha.exp().view(1, self._num_agent, 1).detach()
            z = 1 / cc * (w_target.detach() * q_target_values.detach() - w_target.detach() * v_values).to(self._device)
        else:
            z = 1
        z = torch.clamp(z, min=-5, max=5)
        exp_a = torch.exp(z).detach().squeeze(-1)
        if self.is_mix:
            cc = self.log_alpha.exp().view(1, self._num_agent, 1)
            z_2 = 1 / cc * (w_target.detach() * q_target_values.detach() - w_target.detach() * v_values.detach()).to(self._device)
        else:
            z_2 = 1
        z_2 = torch.clamp(z_2, min=-5, max=5)
        exp_a_2 = torch.exp(z_2).squeeze(-1)
        z_new = z.squeeze(-1)
        tot_kl = - (exp_a * z_new).sum(-1).mean()
        loss_result = self.v_loss(z, w_target.detach(), v_values, result=loss_result)
        loss_result = self.policy_loss(exp_a, a, o_with_id, result=loss_result)
        self._optimizers['policy'].zero_grad()
        loss_result['policy_loss'].backward()
        nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._grad_norm_clip)
        self._optimizers['policy'].step()
        self._optimizers['q'].zero_grad()
        loss_result['q_loss'].backward()
        nn.utils.clip_grad_norm_(self._q_param, self._grad_norm_clip)
        self._optimizers['q'].step()
        self._optimizers['v'].zero_grad()
        loss_result['v_loss'].backward()
        nn.utils.clip_grad_norm_(self._v_network.parameters(), self._grad_norm_clip)
        self._optimizers['v'].step()
        # with torch.no_grad():
        #     dis_1 = self._policy_network.get_dis(o_with_id)
        #     dis_2 = self.behaviour_network.get_dis(o_with_id)
        #     kll = D.kl_divergence(dis_1, dis_2)
        #     self.hi = kll.mean() * self.h
        # h_new = self.compute_contribution(exp_a, o_with_id, a, self._policy_network, s).to(self._device)
        # self.hi = (1 - self.adap_total_alpha_tau) * h_new.detach() * torch.sum(self.hi) + self.adap_total_alpha_tau * self.hi
        # loss_result = self.alpha_loss(self.hi, o_with_id, result=loss_result)
        loss_result = self.alpha_loss(self.hi, o_with_id, result=loss_result)
        self._optimizers['alpha'].zero_grad()
        loss_result['alpha_loss'].backward()
        nn.utils.clip_grad_norm_([self.log_alpha], self._grad_norm_clip)
        self._optimizers['alpha'].step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=-0.5, max=2.5)
        for param, target_param in zip(self._q_network.parameters(), self._q_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._mix_network.parameters(), self._mix_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._v_network.parameters(), self._v_target_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        self._iteration += 1
        loss_result.update({'v_values1': v_values[:,0,:].mean(), 'v_values2': v_values[:,1,:].mean(), 'v_values3': v_values[:,2,:].mean(), 'alpha': self.log_alpha.exp()[0], 'alpha2': self.log_alpha.exp()[1], 'alpha3':self.log_alpha.exp()[2]})
        return loss_result

    def step(self, o):
        o = torch.from_numpy(o).to(self._device)
        one_hot_agent_id = torch.eye(self._num_agent).expand(o.shape[0], -1, -1).to(self._device)
        o_with_id = torch.cat((o, one_hot_agent_id), dim=-1)
        action = self._policy_network.get_deterministic_action(o_with_id)
        return action.detach().cpu()

    def save_models(self, path):
        torch.save(self._policy_network.state_dict(), "{}/policy.th".format(path))
        torch.save(self._q_network.state_dict(), "{}/q.th".format(path))
        torch.save(self._v_network.state_dict(), "{}/v.th".format(path))
        torch.save(self._mix_network.state_dict(), "{}/mixer.th".format(path))

    def load_models(self, path):
        self._policy_network.load_state_dict(torch.load("{}/policy.th".format(path), map_location=lambda storage, loc: storage))
        self._q_network.load_state_dict(torch.load("{}/q.th".format(path), map_location=lambda storage, loc: storage))
        self._v_network.load_state_dict(torch.load("{}/v.th".format(path), map_location=lambda storage, loc: storage))
        self._mix_network.load_state_dict(torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

