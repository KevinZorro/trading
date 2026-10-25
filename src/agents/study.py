"""La configuracion del estudio: ``C*`` = R3 a 120 000 timesteps. ADR 0011.

Un solo lugar del que la importan el Agente A, el Agente B, el placebo y toda
corrida nueva (ADR 0011 seccion 2: B no se calibra aparte). La fija la
confirmacion del 2026-10-25 (``results/confirmacion/summary.json``).

No la usan las corridas ya selladas: el walk-forward de C3 (ADR 0006) y los
Agentes A congelados de la sombra (ADR 0009) corrieron con 60 000 timesteps y
quedan como registro (ADR 0010 seccion 8).
"""

from __future__ import annotations

from agents import confirmation as conf
from agents import regularization as reg
from agents.ppo import PPOConfig
from envs.observation import ObservationSpec

STUDY_PPO: PPOConfig = reg.rung(conf.CONFIRMED_RUNG).ppo
STUDY_OBSERVATION: ObservationSpec = reg.rung(conf.CONFIRMED_RUNG).observation
