import torch
import torch.nn as nn

from ..modules.distributions import SquashedNormal
from ..modules.subnetworks import MLP

LOG_SIG_MAX = 2
LOG_SIG_MIN = -4


class DeterministicActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, max_action=1.):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )
        self.max_action = max_action

    def forward(self, state):
        a = self.trunk(state)
        return self.max_action * torch.tanh(a)


class StochasticActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # print(mu.shape)
        # print(log_std.shape)
        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (
            LOG_SIG_MAX - LOG_SIG_MIN
        ) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

    def sample_n(self, obs, n_samples):
        return self.forward(obs).sample((n_samples,))


class DeterministicActorRNN(nn.Module):
    def __init__(self, obs_dim,goal_dim, out_dim, hidden_dim=256, max_action=1.,time_frame = 5,rnn_type="RNN",num_layers = 2 , device = "cuda"):
        super().__init__()

        self.max_action = max_action
        self.obs_dim = obs_dim
        self.device = device
        self.goal_dim = goal_dim
        self.time_frame = time_frame
        self.obs_per_frame = obs_dim // time_frame
        # self.num_layers = 2
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        print(f"Using {rnn_type} as RNN ")


        # MLP takes concatenated RNN output and goal
        self.trunk = MLP(
            in_dim=self.obs_per_frame + goal_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, state):
        # print("================>",state.shape)
        obs , goal = self.split_tensor(state)
        # print(obs)
        # print(goal)
        batch_size = obs.size(0)
        # print(obs.shape)
        if isinstance(self.rnn, nn.LSTM):

            rnn_output, (hn, cn) = self.rnn(obs, None)
            last_output = hn[-1]  # Get last layer's hidden state
        else:

            rnn_output, hn = self.rnn(obs, None)
            last_output = hn[-1]  # Get last layer's hidden state


        combined = torch.cat([last_output, goal], dim=1)

        # Pass through MLP
        a = self.trunk(combined)
        # print("===================>" ,a.shape)
        return self.max_action * torch.tanh(a)


    def split_tensor(self, inputs):
        """
        Split a tensor of shape (batch, obs) into two tensors:
        (batch, obs_dim) and (batch, goal_dim).

        Returns:
            o_tensor (torch.Tensor): Tensor of shape (batch, obs_dim).
            g_tensor (torch.Tensor): Tensor of shape (batch, goal_dim).
        """
        # Validate the sizes
        # print(inputs.shape)
        # print(inputs.size(1))
        assert int(inputs.shape[1]) == self.goal_dim + self.obs_dim
        # Split the tensor
        o_tensor = inputs[:, :self.obs_dim]
        o_tensor = self.reshape_stacked_obs(o_tensor)
        g_tensor = inputs[:, self.obs_dim:self.obs_dim + self.goal_dim]
        
        return o_tensor, g_tensor
    def reshape_stacked_obs(self,stacked_obs):
        """
        Reshape 1D stacked observations into a 2D array.

        Args:
            stacked_obs (np.ndarray): The 1D array of stacked observations.


        Returns:
            tensor  2D array of stacked observations with shape (k, observation_size).
        """
        assert stacked_obs.shape[1] == self.obs_dim, f"Stacked_obs size does not match expected dimensions. {stacked_obs.shape[1]} ==  * {self.obs_dim},"
        assert self.obs_dim % self.time_frame == 0 ,f"Not divisible"
        stacked_obs_3d = stacked_obs.reshape(stacked_obs.shape[0], self.time_frame, self.obs_dim // self.time_frame)
        # print(stacked_obs_3d.shape)
        stacked_obs_3d = stacked_obs_3d.flip(1)
        # print(stacked_obs_3d.shape)
        return stacked_obs_3d

import torch
import torch.nn as nn



class StochasticActorRNN(nn.Module):
    def __init__(self, obs_dim, goal_dim, out_dim, hidden_dim=256, time_frame=5, 
                 rnn_type="RNN", num_layers=2, device="cuda"):
        super().__init__()

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.time_frame = time_frame
        self.obs_per_frame = obs_dim // time_frame
        self.num_layers = num_layers
        self.device = device

        # Choose RNN type
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.obs_per_frame,
                hidden_size=self.obs_per_frame,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        # Output of RNN + goal => MLP will produce mu and log_std
        # We need 2 * out_dim for mu and log_std
        self.trunk = MLP(
            in_dim=self.obs_per_frame + goal_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, state):
        obs, goal = self.split_tensor(state)
        if isinstance(self.rnn, nn.LSTM):
            rnn_output, (hn, cn) = self.rnn(obs, None)
            last_output = hn[-1]
        else:
            rnn_output, hn = self.rnn(obs, None)
            last_output = hn[-1]

        combined = torch.cat([last_output, goal], dim=1)

        mu, log_std = self.trunk(combined).chunk(2, dim=-1)
        # print(mu.shape)
        # print(log_std.shape)
        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (
            LOG_SIG_MAX - LOG_SIG_MIN
        ) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
        
    def sample_n(self, state, n_samples):
        dist = self.forward(state)
        return dist.sample((n_samples,))

    def split_tensor(self, inputs):
        assert int(inputs.shape[1]) == self.goal_dim + self.obs_dim
        o_tensor = inputs[:, :self.obs_dim]
        o_tensor = self.reshape_stacked_obs(o_tensor)
        g_tensor = inputs[:, self.obs_dim:self.obs_dim + self.goal_dim]
        return o_tensor, g_tensor

    def reshape_stacked_obs(self, stacked_obs):
        assert stacked_obs.shape[1] == self.obs_dim, \
            f"Stacked_obs size does not match expected dimensions: {stacked_obs.shape[1]} != {self.obs_dim}"
        assert self.obs_dim % self.time_frame == 0, "Obs dimension must be divisible by time_frame"
        stacked_obs_3d = stacked_obs.reshape(stacked_obs.shape[0], self.time_frame, self.obs_dim // self.time_frame)
        stacked_obs_3d = stacked_obs_3d.flip(1)
        return stacked_obs_3d

# Example usage
