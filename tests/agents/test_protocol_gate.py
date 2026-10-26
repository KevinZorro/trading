"""Evaluadores de la puerta del Agente B (ADR 0012, 5): niveles 5a y 5b.

Los resultados son sinteticos, con las cifras escritas a mano; el cableado de
``blind`` se prueba con una politica constante, sin torch.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents.policy import ConstantWeightPolicy
from agents.ppo import PPOConfig
from agents.protocol import (
    ArmResult,
    MultiPathResult,
    ProtocolThresholds,
    Verdict,
    evaluate_level_5a,
    evaluate_level_5b,
    run_arm,
)
from data.fixtures import level_5a_exog_signal, level_5b_exog_noise
from envs.observation import ObservationSpec
from eval.distribution import decompose_variance, summarize
from features.news import NEWS_EXOG

from .test_protocol import _METRICAS, SEMILLAS, brazo, corrida

UMBRALES = ProtocolThresholds()


def multi(
    label: str,
    *,
    capture: list[float],
    exceso: list[float] | None = None,
    dispersion: float = 0.05,
    path_seeds: tuple[int, ...] | None = None,
) -> MultiPathResult:
    """Un camino por entrada de ``capture``; las semillas varian +-0.01 alrededor."""
    exceso = exceso if exceso is not None else [0.0] * len(capture)
    brazos = []
    for k, (c, e) in enumerate(zip(capture, exceso, strict=True)):
        corridas = tuple(
            corrida(
                s,
                capture=c + 0.01 * ((i % 3) - 1),
                excess_log_growth_vs_always_long=e + 0.001 * ((i % 3) - 1),
                time_invested=0.95,
                action_std=dispersion,
            )
            for i, s in enumerate(SEMILLAS)
        )
        base = brazo(f"{label}:path{k}")
        brazos.append(
            ArmResult(
                label=base.label,
                fixture_name=base.fixture_name,
                fixture_config=base.fixture_config,
                ppo_config=base.ppo_config,
                runs=corridas,
                distributions={
                    m: summarize(m, SEMILLAS, [getattr(c_, m) for c_ in corridas])
                    for m in _METRICAS
                },
                ceiling=base.ceiling,
                baselines=base.baselines,
            )
        )
    tupla = tuple(brazos)
    return MultiPathResult(
        label=label,
        path_seeds=path_seeds or tuple(range(len(capture))),
        agent_seeds=tuple(SEMILLAS),
        arms=tupla,
        decompositions={
            m: decompose_variance(m, [[getattr(c_, m) for c_ in a.runs] for a in tupla])
            for m in (
                "capture",
                "excess_log_growth_vs_always_long",
                "time_invested",
                "action_std",
                "turnover_annualized",
            )
        },
        drift_t_by_path=tuple([5.5] * len(capture)),
        ppo_config={"total_timesteps": 1},
    )


CAMINOS = [0.02 * k for k in range(10)]  # dispersion entre caminos


def test_5a_aprueba_si_b_captura_y_supera_a_a() -> None:
    b = multi("b", capture=[0.60 + c for c in CAMINOS])
    a = multi("a", capture=[0.00 + 0.01 * (k % 3) for k in range(10)])
    r = evaluate_level_5a(b, a, UMBRALES)
    assert r.verdict is Verdict.PASS
    assert "capture de B: mediana entre caminos 0.69" in r.finding


def test_5a_falla_si_b_no_alcanza_el_capture_minimo() -> None:
    b = multi("b", capture=[0.20 + c for c in CAMINOS])
    a = multi("a", capture=[0.0 + 0.01 * (k % 3) for k in range(10)])
    r = evaluate_level_5a(b, a, UMBRALES)
    assert r.verdict is Verdict.FAIL
    assert "NO captura lo suficiente" in r.finding


def test_5a_falla_si_b_no_se_distingue_de_a() -> None:
    mismo = [0.60 + c for c in CAMINOS]
    r = evaluate_level_5a(
        multi("b", capture=mismo), multi("a", capture=mismo), UMBRALES
    )
    assert r.verdict is Verdict.FAIL
    assert "no se distingue de A" in r.finding


def test_5a_falla_si_b_es_peor_que_a() -> None:
    b = multi("b", capture=[0.55 + c for c in CAMINOS])
    a = multi("a", capture=[0.80 + c for c in CAMINOS])
    assert evaluate_level_5a(b, a, UMBRALES).verdict is Verdict.FAIL


def test_5a_exige_diez_caminos() -> None:
    b = multi("b", capture=[0.7 + 0.01 * k for k in range(5)])
    a = multi("a", capture=[0.0 + 0.01 * k for k in range(5)])
    r = evaluate_level_5a(b, a, UMBRALES)
    assert r.verdict is Verdict.FAIL
    assert "solo hay 5 caminos" in r.finding


def test_5a_exige_las_mismas_semillas_de_camino() -> None:
    b = multi("b", capture=[0.7 + c for c in CAMINOS])
    a = multi("a", capture=[0.0 + c for c in CAMINOS], path_seeds=tuple(range(50, 60)))
    r = evaluate_level_5a(b, a, UMBRALES)
    assert r.verdict is Verdict.FAIL
    assert "no comparten las semillas de camino" in r.finding


def test_5b_se_mide_y_no_bloquea_ni_cuando_b_rota_con_el_ruido() -> None:
    exc = [0.01 * k for k in range(10)]
    b = multi("b", capture=exc, exceso=exc, dispersion=0.30)
    a = multi("a", capture=exc, exceso=[e + 0.02 for e in exc], dispersion=0.05)
    r = evaluate_level_5b(b, a, UMBRALES)
    assert r.verdict is Verdict.MEASURED
    assert "HALLAZGO: B rota con el ruido" in r.finding
    assert "Exceso sobre estar invertido" in r.finding


def test_5b_sin_rotacion_no_levanta_hallazgo() -> None:
    exc = [0.01 * k for k in range(10)]
    b = multi("b", capture=exc, exceso=exc, dispersion=0.05)
    a = multi("a", capture=exc, exceso=exc)
    r = evaluate_level_5b(b, a, UMBRALES)
    assert r.verdict is Verdict.MEASURED
    assert "HALLAZGO" not in r.finding


def test_el_umbral_del_5a_viaja_con_los_umbrales() -> None:
    assert UMBRALES.describe()["level_5a_min_capture"] == 0.50


# -- cableado de `blind` ---------------------------------------------------


def _corre(fixture, *, blind: bool, observation=None):  # type: ignore[no-untyped-def]
    return run_arm(
        fixture,
        label="x",
        seeds=[1],
        ppo_config=PPOConfig(total_timesteps=1),
        policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0),
        observation=observation,
        blind=blind,
        allow_fewer_seeds=True,
    )


def test_a_ciego_corre_sobre_el_mismo_mercado_con_exogenas() -> None:
    fx = level_5b_exog_noise(600, seed=3)
    r = _corre(fx, blind=True)
    assert r.fixture_config["series"]["exog_names"]  # el fixture las tiene
    assert len(r.runs) == 1


def test_a_sin_blind_rechaza_una_serie_con_exogenas_que_no_ve() -> None:
    with pytest.raises(ValueError, match="la observacion pide"):
        _corre(level_5b_exog_noise(600, seed=3), blind=False)


def test_b_usa_las_siete_columnas_y_el_techo_del_5a_esta_definido() -> None:
    spec = replace(ObservationSpec(), exog=NEWS_EXOG)
    fx = level_5a_exog_signal(1_000, seed=3)
    b = _corre(fx, blind=False, observation=spec)
    a = _corre(fx, blind=True)
    # Una politica constante en 1.0 es estar siempre invertido: captura cero,
    # pero el capture existe (el techo con canal no coincide con always_long).
    assert b.runs[0].capture is not None
    assert a.runs[0].capture == pytest.approx(b.runs[0].capture)
