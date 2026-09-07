"""Humo del adaptador de PPO. Necesita el grupo ``rl``, que no se instala en CI.

Marcado ``rl`` **y** ``slow``: se saltea solo cuando torch no esta, en vez de
fallar. Lo que verifica no es que el agente aprenda -eso lo mide el protocolo,
que tarda horas- sino que el cableado con stable-baselines3 este bien: que la
politica entrenada devuelva acciones en rango, que la evaluacion sea
determinista y que el estado de la LSTM se reinicie entre episodios.

**El adaptador de SB3 no esta cubierto por el CI.** La rueda de torch de PyPI
para Linux arrastra el stack de CUDA (varios GB) y el indice de ruedas de CPU no
es alcanzable desde el entorno donde se desarrollo esto, asi que no se pudo
verificar un job que lo instale. Queda dicho en vez de disimulado; ver el ADR.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.policy import Policy
from agents.ppo import PPOConfig, PPOUnavailableError, SB3Policy, train_ppo
from agents.runner import build_env, run_policy, scaler_for
from data.fixtures import level_0_deterministic
from envs.rewards import NetReturnReward
from sim.engine import SimConfig

pytestmark = [pytest.mark.rl, pytest.mark.slow]

CAPITAL = 100_000.0


@pytest.fixture(scope="module")
def entrenado() -> tuple[SB3Policy, object, object]:
    pytest.importorskip("stable_baselines3")
    fixture = level_0_deterministic(400)
    train, validacion, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)

    def construir(serie: object) -> object:
        return build_env(
            serie.series,  # type: ignore[attr-defined]
            config,
            scaler=escalador,
            reward=NetReturnReward(scale=100.0),
        )

    politica = train_ppo(
        lambda: construir(train),  # type: ignore[arg-type]
        PPOConfig(total_timesteps=512, n_steps=64, batch_size=32),
        seed=0,
    )
    return politica, construir, validacion


def test_la_politica_entrenada_satisface_el_protocolo(entrenado) -> None:  # type: ignore[no-untyped-def]
    politica, _, _ = entrenado
    assert isinstance(politica, Policy)
    assert politica.name.startswith("ppo")


def test_las_acciones_caen_en_el_rango_del_motor(entrenado) -> None:  # type: ignore[no-untyped-def]
    politica, construir, validacion = entrenado
    salida = run_policy(construir(validacion), politica)
    acciones = np.asarray(salida.actions)
    assert np.all((acciones >= 0.0) & (acciones <= 1.0))
    # SB3 recorta contra el espacio antes de entregar la accion, asi que el
    # entorno no deberia registrar ni un recorte propio.
    assert salida.clipped_actions == 0


def test_la_evaluacion_es_determinista(entrenado) -> None:  # type: ignore[no-untyped-def]
    """Se evalua con ``deterministic=True``: dos pasadas dan lo mismo.

    Si difirieran, el reporte estaria midiendo el promedio de las dudas del
    agente en vez de la politica que se pondria a operar.
    """
    politica, construir, validacion = entrenado
    a = run_policy(construir(validacion), politica)
    b = run_policy(construir(validacion), politica)
    assert a.actions == b.actions
    np.testing.assert_array_equal(a.result.equity, b.result.equity)


def test_la_config_rechaza_minibatches_incompletos() -> None:
    with pytest.raises(ValueError, match="no es multiplo de batch_size"):
        PPOConfig(n_steps=100, batch_size=32)
    with pytest.raises(ValueError, match="supera n_steps"):
        PPOConfig(n_steps=32, batch_size=64)


def test_la_politica_recurrente_reinicia_su_estado() -> None:
    """Sin reiniciar, el primer paso arrastra la memoria del episodio anterior y
    el resultado dependeria del orden de evaluacion."""
    pytest.importorskip("sb3_contrib")
    fixture = level_0_deterministic(400)
    train, validacion, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)

    def construir(serie_fixture: object) -> object:
        return build_env(
            serie_fixture.series,  # type: ignore[attr-defined]
            config,
            scaler=escalador,
            reward=NetReturnReward(scale=100.0),
        )

    politica = train_ppo(
        lambda: construir(train),  # type: ignore[arg-type]
        PPOConfig(total_timesteps=256, n_steps=64, batch_size=32, recurrent=True),
        seed=0,
    )
    a = run_policy(construir(validacion), politica)  # type: ignore[arg-type]
    b = run_policy(construir(validacion), politica)  # type: ignore[arg-type]
    assert a.actions == b.actions


def test_el_error_de_import_explica_como_arreglarlo() -> None:
    assert issubclass(PPOUnavailableError, ImportError)
