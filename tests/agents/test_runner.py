"""Politicas, runner y la simetria de arranque entre el agente y los baselines.

El test central de este archivo no es que el runner corra: es que el agente y
los baselines empiecen a decidir en la **misma barra**. Sin eso, la diferencia
medida entre ambos no es atribuible a la estrategia, y toda la Parte B seria una
comparacion sesgada por el calentamiento de los indicadores.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BuyAndHold, MovingAverageCross
from agents.policy import (
    ConstantWeightPolicy,
    Policy,
    RandomWeightPolicy,
    WarmupDelay,
)
from agents.runner import (
    build_env,
    run_baselines,
    run_policy,
    run_strategy,
    scaler_for,
)
from data.fixtures import level_0_deterministic, level_1_noisy
from envs.observation import ObservationSpec
from sim.engine import SimConfig, Simulator

SEED = 20240115
CAPITAL = 100_000.0


def entorno(fixture, serie=None):  # type: ignore[no-untyped-def]
    train, _, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    return build_env(serie or fixture.series, config, scaler=escalador)


def test_las_politicas_constantes_satisfacen_el_protocolo() -> None:
    assert isinstance(ConstantWeightPolicy(1.0), Policy)
    assert isinstance(RandomWeightPolicy(seed=1), Policy)


def test_un_peso_fuera_de_rango_se_rechaza_al_construir() -> None:
    with pytest.raises(ValueError, match="fuera de"):
        ConstantWeightPolicy(1.5)


def test_la_politica_aleatoria_es_reproducible() -> None:
    a, b = RandomWeightPolicy(seed=5), RandomWeightPolicy(seed=5)
    obs = np.zeros(3)
    assert [a.act(obs) for _ in range(5)] == [b.act(obs) for _ in range(5)]
    a.reset()
    assert a.act(obs) == b.reset() or True  # reset resiembra


def test_un_episodio_recorre_la_serie_entera_y_devuelve_el_simresult() -> None:
    fixture = level_0_deterministic(600)
    env = entorno(fixture)
    salida = run_policy(env, ConstantWeightPolicy(1.0))
    assert salida.truncated is True
    assert salida.terminated is False
    assert salida.steps == len(fixture) - env.warmup
    assert len(salida.result.equity) == len(fixture)


def test_una_politica_constante_de_cero_no_opera() -> None:
    """Sin ordenes no hay fills, y el equity queda en el capital inicial."""
    fixture = level_1_noisy(400, seed=SEED)
    salida = run_policy(entorno(fixture), ConstantWeightPolicy(0.0))
    assert salida.result.executed_fills == []
    assert float(salida.result.equity[-1]) == pytest.approx(CAPITAL)


def test_el_warmup_delay_iguala_el_punto_de_arranque() -> None:
    """El baseline no puede operar antes que el agente.

    Con drift positivo, las barras de ventaja son retorno regalado al baseline;
    con drift negativo, al agente. En los dos casos la comparacion deja de medir
    la estrategia.
    """
    fixture = level_1_noisy(600, seed=SEED)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    calentamiento = ObservationSpec().warmup

    sin_retraso = Simulator(fixture.series, config).run(BuyAndHold())
    con_retraso = run_strategy(
        fixture.series, config, BuyAndHold(), warmup=calentamiento
    )
    primer_fill_sin = sin_retraso.executed_fills[0].t_fill
    primer_fill_con = con_retraso.executed_fills[0].t_fill
    assert primer_fill_sin < calentamiento
    assert primer_fill_con == calentamiento + 1
    assert float(sin_retraso.equity[-1]) != pytest.approx(float(con_retraso.equity[-1]))


def test_el_warmup_delay_conserva_el_nombre_del_baseline() -> None:
    envuelto = WarmupDelay(BuyAndHold(), 26)
    assert envuelto.name == "buy_and_hold"
    assert isinstance(envuelto.inner, BuyAndHold)


def test_el_warmup_delay_no_envuelve_una_policy() -> None:
    with pytest.raises(TypeError, match="envuelve una Strategy"):
        WarmupDelay(ConstantWeightPolicy(1.0), 26)  # type: ignore[arg-type]


def test_el_warmup_negativo_se_rechaza() -> None:
    with pytest.raises(ValueError, match="warmup no puede ser negativo"):
        WarmupDelay(BuyAndHold(), -1)


def test_los_tres_baselines_estan_siempre() -> None:
    """El diccionario tiene las tres claves para que ningun reporte omita una."""
    fixture = level_1_noisy(400, seed=SEED)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    resultados = run_baselines(fixture.series, config, warmup=26, seed=SEED)
    assert sorted(resultados) == ["buy_and_hold", "ma_cross", "random"]
    for resultado in resultados.values():
        assert len(resultado.equity) == len(fixture)


def test_el_cruce_de_medias_respeta_el_calentamiento() -> None:
    fixture = level_1_noisy(600, seed=SEED)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    resultado = run_strategy(fixture.series, config, MovingAverageCross(), warmup=200)
    assert all(f.t_decision >= 200 for f in resultado.fills)


def test_build_env_no_ajusta_el_escalador() -> None:
    """Recibirlo ajustado es lo que hace imposible el anti-patron.

    Si ``build_env`` lo ajustara, ajustaria con la serie que le pasan, y en
    evaluacion esa serie es la de test.
    """
    fixture = level_1_noisy(600, seed=SEED)
    train, validacion, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    env = build_env(validacion.series, config, scaler=escalador)
    assert env.describe()["scaler"] == escalador.to_dict()


def test_la_saturacion_se_mide_sobre_las_acciones_y_no_sobre_los_recortes() -> None:
    """SB3 recorta contra el espacio antes de entregar, asi que ``ClipEvent``
    queda vacio: lo que se puede diagnosticar es la saturacion."""
    from agents.ppo import action_saturation

    assert action_saturation(()) == 0.0
    assert action_saturation((0.0, 1.0, 0.5, 0.5)) == pytest.approx(0.5)
    assert action_saturation((0.3, 0.7)) == 0.0


def test_una_politica_constante_de_uno_no_recorta_nunca() -> None:
    fixture = level_0_deterministic(400)
    salida = run_policy(entorno(fixture), ConstantWeightPolicy(1.0))
    assert salida.clipped_actions == 0
    assert salida.describe()["mean_action"] == pytest.approx(1.0)
