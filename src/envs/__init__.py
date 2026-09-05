"""Entornos Gymnasium sobre el simulador.

Wrapper delgado por diseno: el entorno rutea, no calcula. La ejecucion, la
contabilidad, los costos y el dimensionamiento viven en ``sim/``; los
indicadores y la normalizacion en ``features/``; los limites de riesgo en
``risk/``. Un entorno que reimplementara cualquiera de esas cosas divergiria del
simulador validado y el backtest dejaria de corresponder a lo que se midio.
"""

from envs.observation import ObservationBuilder, ObservationSpec, fit_scaler_on_train
from envs.rewards import (
    REWARDS,
    DifferentialSharpeReward,
    NetReturnReward,
    RewardFn,
    make_reward,
)
from envs.trading_env import ClipEvent, EnvConfig, TradingEnv

__all__ = [
    "REWARDS",
    "ClipEvent",
    "DifferentialSharpeReward",
    "EnvConfig",
    "NetReturnReward",
    "ObservationBuilder",
    "ObservationSpec",
    "RewardFn",
    "TradingEnv",
    "fit_scaler_on_train",
    "make_reward",
]
