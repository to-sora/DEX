# dex_new.py
import copy
import torch
import torch.nn.functional as F

from ..utils.general_utils import AttrDict
from .ddpgbc import DDPGBC
from ..modules.policies import  DeterministicActorRNN
# from .new_networks import NewDeterministicActor, NewCritic  # Import the new networks
from ..modules.critics import CriticLastFrame
class DEXLSTM(DDPGBC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(self.time_frame)
        print()
        # Override the actor and critic with new architectures
        print(f"dimo {self.dimo} + dimg {self.dimg}  act {self.dima}")
        self.actor = DeterministicActorRNN(
            obs_dim=self.dimo ,goal_dim= self.dimg,
            out_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame = self.time_frame,
            rnn_type="LSTM",
            num_layers = 2 ,
            device = agent_cfg.device
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)

        self.critic = CriticLastFrame(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            action_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)
        
        # Re-initialize optimizers with new parameters
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        
        # If there are any additional initializations specific to DEX, include them here

    # If the new actor or critic require additional methods or overrides, define them here
    # Otherwise, all other methods will inherit from DEX and function as before
