"""Agentes: baselines heuristicos y el Agente A (PPO sobre el entorno).

El adaptador de PPO (``agents.ppo``) **no** se re-exporta desde aca a proposito:
importarlo arrastra ``stable_baselines3`` y ``torch``, que son un grupo opcional.
Si estuviera en este ``__init__``, ``import agents`` fallaria en cualquier
entorno sin el grupo instalado, incluido el job principal de CI. Se importa por
su ruta completa: ``from agents.ppo import train_ppo``.
"""

from agents.baselines import BuyAndHold, MovingAverageCross, RandomAgent
from agents.experiment import ExperimentLog, ExperimentRecord, jsonable
from agents.policy import (
    ConstantWeightPolicy,
    Policy,
    WarmupDelay,
)
from agents.protocol import (
    ArmResult,
    LevelResult,
    MultiPathResult,
    ProtocolReport,
    ProtocolThresholds,
    SeedRun,
    Verdict,
    assemble_protocol,
    run_arm,
    run_multipath_arm,
)
from agents.runner import (
    EpisodeOutcome,
    build_env,
    run_baselines,
    run_policy,
    run_strategy,
    scaler_for,
)

__all__ = [
    "ArmResult",
    "BuyAndHold",
    "ConstantWeightPolicy",
    "EpisodeOutcome",
    "ExperimentLog",
    "ExperimentRecord",
    "LevelResult",
    "MovingAverageCross",
    "MultiPathResult",
    "Policy",
    "ProtocolReport",
    "ProtocolThresholds",
    "RandomAgent",
    "SeedRun",
    "Verdict",
    "WarmupDelay",
    "assemble_protocol",
    "build_env",
    "jsonable",
    "run_arm",
    "run_baselines",
    "run_multipath_arm",
    "run_policy",
    "run_strategy",
    "scaler_for",
]
