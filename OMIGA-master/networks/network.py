import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Actor, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)

        action = torch.tanh(action)
        return action, logp_pi, a_distribution, log_sigma
    
    def get_sigma(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        return log_sigma

    def get_log_density(self, x, y):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=-1)
        return logp_pi

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action
    
    def get_dis(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)
        a_distribution = Normal(mu, sigma)
        return a_distribution

    def get_deterministic_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)
        return mu


class V_critic(nn.Module):
    def __init__(self, num_state, num_agent, num_hidden, device):
        super(V_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v
    
    def v(self, obs, agent_id):
        x = torch.cat([obs, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class Q_critic(nn.Module):
    def __init__(self, num_state, num_action, num_agent, num_hidden, device):
        super(Q_critic, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state + num_action + num_agent, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q

    def q(self, obs, action, agent_id):
        x = torch.cat([obs, action, agent_id], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.state_value(x)
        return q


class MixNet(nn.Module):
    def __init__(self, num_state, num_agent, num_hidden, device):
        super(MixNet, self).__init__()
        self.device = device
        self.state_shape = num_state * num_agent  # concat state from agents
        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden

        self.f_v = nn.Linear(self.state_shape, num_hidden)
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)

    def forward(self, states):
        batch_size = states.size(0)
        states = torch.cat([states[:, j, :] for j in range(self.n_agents)], dim=-1)
        states = states.reshape(-1, self.state_shape)
        x = self.f_v(states)
        w = self.w_v(F.relu(x)).reshape(batch_size, self.n_agents, 1)
        b = self.b_v(F.relu(x)).reshape(batch_size, 1, 1)
        
        return torch.abs(w), b
    
class MixNet_2(nn.Module):
    def __init__(self, num_state, num_agent, num_hidden, device):
        super(MixNet_2, self).__init__()
        self.device = device
        self.state_shape = num_state  # state shape for each agent
        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden

        self.f_v = nn.Linear(self.state_shape + num_agent, num_hidden)
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)

    def forward(self, states):
        batch_size = states.size(0)

        # Create one-hot encoding for agent IDs
        agent_ids = torch.eye(self.n_agents).to(self.device)  # (n_agents, n_agents)

        w_list = []
        b_list = []

        for i in range(self.n_agents):
            # Expand agent ID to batch size and concat with state
            agent_id_expanded = agent_ids[i].unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_agents)
            state_with_id = torch.cat([states[:, i, :], agent_id_expanded], dim=-1)  # (batch_size, state_shape + n_agents)
            
            x = self.f_v(state_with_id)
            w = self.w_v(F.relu(x)).reshape(batch_size, self.n_agents)
            b = self.b_v(F.relu(x)).reshape(batch_size, 1)
            w_list.append(w)
            b_list.append(b)

        w = torch.stack(w_list, dim=1)  # (batch_size, n_agents, n_agents)
        b = torch.stack(b_list, dim=1)  # (batch_size, n_agents, 1)
        
        return torch.abs(w), b

class MixNet_3(nn.Module):
    def __init__(self, num_state, num_agent, num_hidden, device):
        super(MixNet_3, self).__init__()
        self.device = device
        self.state_shape = num_state * num_agent  # concat state from agents
        self.n_agents = num_agent
        self.hyper_hidden_dim = num_hidden

        self.f_v = nn.Linear(self.state_shape + self.n_agents, num_hidden)  # Adding space for the one-hot index and q values
        self.w_v = nn.Linear(num_hidden, num_agent)
        self.b_v = nn.Linear(num_hidden, 1)  # Change output dimension to num_agent

    def forward(self, global_state):
        batch_size = global_state.size(0)
        w_list = []
        b_list = []
        for agent_idx in range(self.n_agents):
            # Create one-hot vector for agent index
            one_hot_idx = F.one_hot(torch.tensor(agent_idx), self.n_agents).float().to(self.device)
            one_hot_idx = one_hot_idx.unsqueeze(0).expand(batch_size, -1)  # Expand to batch size
            # Concatenate global state with the one-hot index of the agent
            agent_inputs = torch.cat([global_state, one_hot_idx], dim=-1)
            # Compute f(s)
            x = self.f_v(agent_inputs)
            x = F.relu(x)
            w = self.w_v(x).reshape(batch_size, 1, self.n_agents)
            b = self.b_v(x).reshape(batch_size, 1)
            w_list.append(w)
            b_list.append(b)
        w_stack = torch.cat(w_list, dim=1)  # Stack w along the agent dimension
        b_stack = torch.cat(b_list, dim=1)  # Stack b along the agent dimension
        return torch.abs(w_stack), b_stack

