# new_critics.py
import torch
import torch.nn as nn

from ..modules.subnetworks import MLP

class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.q = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q = self.q(sa)
        return q


class DoubleCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.q1 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )
        self.q2 = MLP(
            in_dim=in_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class CriticLastFrame(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim=256, time_frame=5):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_frame = time_frame
        self.obs_per_frame = obs_dim // time_frame

        self.q = MLP(
            in_dim=self.obs_per_frame + goal_dim + action_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        # Split state into obs and goal
        obs = state[:, :self.obs_dim]
        goal = state[:, self.obs_dim:self.obs_dim + self.goal_dim]

        # Extract current observation (last frame)
        current_obs = obs[:, :self.obs_per_frame]

        # Concatenate current_obs, goal, and action
        sa = torch.cat([current_obs, goal, action], dim=-1)

        q = self.q(sa)
        return q


class DoubleCriticLastFrame(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim=256, time_frame=5):
        super().__init__()
        print("Initializing DoubleCriticLastFrame...")
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_frame = time_frame
        self.obs_per_frame = obs_dim // time_frame

        self.q1 = MLP(
            in_dim=self.obs_per_frame + goal_dim + action_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )

        self.q2 = MLP(
            in_dim=self.obs_per_frame + goal_dim + action_dim,
            out_dim=1,
            hidden_dim=hidden_dim
        )
        print("DoubleCriticLastFrame initialized successfully.")

    def forward(self, state, action):
        # Split state into obs and goal
        obs = state[:, :self.obs_dim]
        goal = state[:, self.obs_dim:self.obs_dim + self.goal_dim]

        # Extract current observation (last frame)
        current_obs = obs[:, :self.obs_per_frame]

        # Concatenate current_obs, goal, and action
        sa = torch.cat([current_obs, goal, action], dim=-1)

        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
