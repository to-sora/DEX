from .ddpg import DDPG
from .ddpgbc import DDPGBC
from .col import CoL
from .dex import DEX

from .sac import SAC
from .sqil import SQIL
from .amp import AMP
from .awac import AWAC
from .dex_RNN import DEXRNN
from .dex_GRU import DEXGRU
from .dex_LSTM import DEXLSTM
from .GRU_class import DDPGGRU, DDPGBCGRU, CoLGRU, DEXGRU2, SACGRU, SQILGRU, AMPGRU, AWACGRU





AGENTS = {
    'DDPG': DDPG,
    'DDPGBC': DDPGBC,
    'CoL': CoL,
    'DEX': DEX,
    'SAC': SAC,
    'SQIL': SQIL,
    'AMP': AMP,
    'AWAC': AWAC,
    "DEXRNN2": DEXRNN,
    "DEXGRU" : DEXGRU,
    "DEXLSTM" : DEXLSTM,
    "DDPGGRU": DDPGGRU,
    "DDPGBCGRU": DDPGBCGRU,
    "CoLGRU": CoLGRU,
    "DEXGRU2": DEXGRU2,
    "SACGRU": SACGRU,
    "SQILGRU": SQILGRU,
    "AMPGRU": AMPGRU,
    "AWACGRU": AWACGRU
}


def make_agent(env_params, sampler, cfg):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        return AGENTS[cfg.name](
            env_params=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )
