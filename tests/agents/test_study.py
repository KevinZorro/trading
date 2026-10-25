"""La configuracion del estudio es la que paso la confirmacion (ADR 0011).

Los valores se escriben a mano, no se leen de ``regularization``: si alguien
cambia un peldano, este test falla en vez de arrastrar el cambio al estudio.
"""

from __future__ import annotations

from agents.ppo import PPOConfig
from agents.study import STUDY_OBSERVATION, STUDY_PPO
from agents.walkforward import WalkForwardConfig
from envs.observation import ObservationSpec


def test_es_r3_a_120k() -> None:
    assert (
        PPOConfig(
            total_timesteps=120_000,
            learning_rate=1e-4,
            n_epochs=5,
            net_arch=(32, 32),
            gamma=0.9,
        )
        == STUDY_PPO
    )
    assert ObservationSpec(lookback=1) == STUDY_OBSERVATION


def test_el_walk_forward_sellado_de_c3_no_cambia() -> None:
    """ADR 0010 seccion 8: C3 corrio con 60k y queda como registro."""
    ppo = WalkForwardConfig().ppo_config()
    assert ppo.total_timesteps == 60_000
    assert (ppo.learning_rate, ppo.gamma, ppo.net_arch) == (3e-4, 0.99, (64, 64))
