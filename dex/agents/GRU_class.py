# agents_gru.py

import copy
import torch
import torch.nn.functional as F
from ..utils.general_utils import AttrDict
from ..modules.policies import DeterministicActorRNN , StochasticActorRNN 
from ..modules.critics import CriticLastFrame , DoubleCriticLastFrame 

from .ddpg import DDPG
from .ddpgbc import DDPGBC
from .col import CoL
from .dex import DEX
from .sac import SAC
from .sqil import SQIL
from .amp import AMP
from .awac import AWAC

class DDPGGRU(DDPG):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing DDPGGRU with timeframe: {self.time_frame}")
        
        self.actor = DeterministicActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
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
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        print("DDPGGRU initialized successfully.")

class DDPGBCGRU(DDPGBC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing DDPGBCGRU with timeframe: {self.time_frame}")
        
        self.actor = DeterministicActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
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

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        print("DDPGBCGRU initialized successfully.")

class CoLGRU(CoL):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing CoLGRU with timeframe: {self.time_frame}")
        
        self.actor = DeterministicActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
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
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        print("CoLGRU initialized successfully.")

class DEXGRU2(DEX):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing DEXGRU2 with timeframe: {self.time_frame}")
        
        self.actor = StochasticActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
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
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_cfg.critic_lr
        )
        print("DEXGRU2 initialized successfully.")

class SACGRU(SAC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing SACGRU with timeframe: {self.time_frame}")
        
        self.actor = StochasticActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima*2,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
 
        # For SAC, normally we have two critics (Q-networks). We use the same RNN critic here.
        self.critic = DoubleCriticLastFrame(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            action_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=agent_cfg.critic_lr
        )
        print("SACGRU initialized successfully.")

class SQILGRU(SQIL):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing SQILGRU with timeframe: {self.time_frame}")
        
        self.actor = StochasticActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima*2,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
 
        # For SAC, normally we have two critics (Q-networks). We use the same RNN critic here.
        self.critic = DoubleCriticLastFrame(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            action_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=agent_cfg.critic_lr
        )
        print("SQILGRU initialized successfully.")

class AMPGRU(AMP):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing AMPGRU with timeframe: {self.time_frame}")
        
        self.actor = StochasticActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima*2,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
 
        # For SAC, normally we have two critics (Q-networks). We use the same RNN critic here.
        self.critic = DoubleCriticLastFrame(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            action_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=agent_cfg.critic_lr
        )
        print("AMPGRU initialized successfully.")

class AWACGRU(AWAC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.time_frame = agent_cfg.timeframe
        print(f"Initializing AWACGRU with timeframe: {self.time_frame}")
        
        self.actor = StochasticActorRNN(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            out_dim=self.dima*2,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame,
            rnn_type="GRU",
            num_layers=2,
            device=agent_cfg.device
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
 
        # For SAC, normally we have two critics (Q-networks). We use the same RNN critic here.
        self.critic = DoubleCriticLastFrame(
            obs_dim=self.dimo,
            goal_dim=self.dimg,
            action_dim=self.dima,
            hidden_dim=agent_cfg.hidden_dim,
            time_frame=self.time_frame
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=agent_cfg.critic_lr
        )
        print("AWACGRU initialized successfully.")
