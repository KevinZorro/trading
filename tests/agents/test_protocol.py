"""El protocolo de validacion: que juzgue bien y que **pare** cuando debe.

Un criterio que solo se ejercita con agentes que pasan no esta testeado. Casi
todos los tests de aca construyen resultados que **fallan** a proposito, porque
lo que hay que garantizar es que el protocolo detecte el fallo, lo explique y no
siga adelante produciendo numeros que habria que descartar despues.

Nada de esto necesita torch: los criterios se aplican sobre resultados ya
corridos, y la unica corrida de punta a punta usa una politica constante.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agents.policy import ConstantWeightPolicy
from agents.ppo import PPOConfig
from agents.protocol import (
    ArmResult,
    ProtocolThresholds,
    SeedRun,
    Verdict,
    assemble_protocol,
    evaluate_level_0,
    evaluate_level_1,
    evaluate_level_2,
    evaluate_level_3,
    evaluate_level_4,
    run_arm,
)
from data.fixtures import level_0_deterministic, level_1_noisy
from eval.distribution import summarize

SEMILLAS = list(range(10))
UMBRALES = ProtocolThresholds()

_METRICAS = (
    "capture",
    "log_growth",
    "excess_log_growth_vs_always_long",
    "total_return_mark",
    "total_return_liquidation",
    "friction_gap",
    "sharpe_mark",
    "max_drawdown_mark",
    "turnover_annualized",
    "time_invested",
    "saturation",
    "n_round_trips",
    "win_rate",
)


def corrida(seed: int, **campos: object) -> SeedRun:
    base: dict[str, object] = {
        "capture": 0.9,
        "log_growth": 0.5,
        "excess_log_growth_vs_always_long": 0.4,
        "total_return_mark": 0.6,
        "total_return_liquidation": 0.59,
        "friction_gap": 0.01,
        "sharpe_mark": 1.2,
        "max_drawdown_mark": 0.1,
        "turnover_annualized": 30.0,
        "time_invested": 0.5,
        "saturation": 0.9,
        "clipped_actions": 0,
        "n_round_trips": 40,
        "win_rate": 0.6,
        "costs_paid": 100.0,
        "gap": 0.0,
        "ruined": False,
    }
    base.update(campos)
    return SeedRun(seed=seed, **base)  # type: ignore[arg-type]


def brazo(label: str, *, r_squared: float = 0.09, **campos: object) -> ArmResult:
    corridas = tuple(corrida(s, **campos) for s in SEMILLAS)
    distribuciones = {
        m: summarize(m, SEMILLAS, [getattr(c, m) for c in corridas]) for m in _METRICAS
    }
    return ArmResult(
        label=label,
        fixture_name=label,
        fixture_config={"name": label, "spec": {"r_squared": r_squared}},
        ppo_config={"total_timesteps": 1},
        runs=corridas,
        distributions=distribuciones,
        ceiling={"informed_total_return": 1.0},
        baselines={
            "buy_and_hold": {
                "total_return_mark": 0.2,
                "turnover_annualized": 0.05,
            },
            "random": {"total_return_mark": 0.05, "turnover_annualized": 12.0},
            "ma_cross": {"total_return_mark": 0.1, "turnover_annualized": 3.0},
        },
    )


# ---------------------------------------------------------------------------
# Nivel 0
# ---------------------------------------------------------------------------


def test_nivel_0_aprueba_cuando_el_agente_se_acerca_al_optimo() -> None:
    resultado = evaluate_level_0(brazo("level_0", capture=0.95), UMBRALES)
    assert resultado.verdict is Verdict.PASS
    assert "0.950" in resultado.finding


def test_nivel_0_falla_y_manda_a_mirar_el_pipeline() -> None:
    """El mensaje importa tanto como el veredicto.

    Un fallo en el nivel 0 que se lea como "faltan hiperparametros" hace que
    alguien pase la tarde barriendo learning rates sobre un bug de la
    observacion.
    """
    resultado = evaluate_level_0(brazo("level_0", capture=0.4), UMBRALES)
    assert resultado.verdict is Verdict.FAIL
    assert "pipeline" in resultado.finding
    assert "No pasar al nivel 1" in resultado.finding


def test_nivel_0_falla_si_el_capture_es_indefinido() -> None:
    """``capture`` indefinido en el nivel 0 delata un fixture mal armado."""
    resultado = evaluate_level_0(brazo("level_0", capture=None), UMBRALES)
    assert resultado.verdict is Verdict.FAIL
    assert "fixture mal armado" in resultado.finding


# ---------------------------------------------------------------------------
# Nivel 1
# ---------------------------------------------------------------------------


def brazos_snr(capturas: dict[float, float]) -> list[ArmResult]:
    return [
        brazo(f"level_1_beta{-(r**0.5):.1f}", r_squared=r, capture=c)
        for r, c in capturas.items()
    ]


def test_nivel_1_aprueba_una_degradacion_suave() -> None:
    resultado = evaluate_level_1(
        brazos_snr({0.25: 0.90, 0.09: 0.80, 0.04: 0.65, 0.01: 0.40}), UMBRALES
    )
    assert resultado.verdict is Verdict.PASS


def test_nivel_1_falla_si_el_capture_sube_al_bajar_el_snr() -> None:
    """Capturar mas fraccion del techo con menos senal no es "ruido favorable":
    es que la normalizacion o el techo estan mal."""
    resultado = evaluate_level_1(
        brazos_snr({0.25: 0.40, 0.09: 0.80, 0.04: 0.85, 0.01: 0.90}), UMBRALES
    )
    assert resultado.verdict is Verdict.FAIL
    assert "Inversiones fuera de tolerancia" in resultado.finding


def test_nivel_1_tolera_inversiones_dentro_del_ruido_muestral() -> None:
    """Con 10 semillas el error estandar de la mediana es del orden de 0.05.

    Exigir monotonia estricta haria fallar el nivel por ruido, y el resultado
    seria un protocolo que nadie puede pasar dos veces seguidas.
    """
    resultado = evaluate_level_1(
        brazos_snr({0.25: 0.80, 0.09: 0.85, 0.04: 0.70, 0.01: 0.50}), UMBRALES
    )
    assert resultado.verdict is Verdict.PASS


def test_nivel_1_falla_si_ni_con_el_snr_mas_alto_captura() -> None:
    resultado = evaluate_level_1(
        brazos_snr({0.25: 0.30, 0.09: 0.25, 0.04: 0.20, 0.01: 0.10}), UMBRALES
    )
    assert resultado.verdict is Verdict.FAIL
    assert "por debajo del minimo" in resultado.finding


# ---------------------------------------------------------------------------
# Nivel 2
# ---------------------------------------------------------------------------


def test_nivel_2_aprueba_cuando_rota_menos_con_costos() -> None:
    resultado = evaluate_level_2(
        brazo("level_2", turnover_annualized=20.0),
        brazo("level_1", turnover_annualized=40.0),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.PASS
    assert "caida del 50" in resultado.finding


def test_nivel_2_falla_si_opera_igual_con_y_sin_costos() -> None:
    """El criterio esta sobre el comportamiento, no sobre el retorno.

    Rendir menos con costos es automatico -se restan del equity aunque el agente
    los ignore-. Lo que prueba que la penalizacion llego al reward es rotar menos.
    """
    resultado = evaluate_level_2(
        brazo("level_2", turnover_annualized=39.0),
        brazo("level_1", turnover_annualized=40.0),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.FAIL
    assert "no esta llegando al reward" in resultado.finding


def test_nivel_2_falla_si_la_referencia_no_rota() -> None:
    resultado = evaluate_level_2(
        brazo("level_2", turnover_annualized=10.0),
        brazo("level_1", turnover_annualized=0.0),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.FAIL


# ---------------------------------------------------------------------------
# Nivel 3
# ---------------------------------------------------------------------------


def test_nivel_3_se_mide_y_no_se_aprueba() -> None:
    """Adaptarse y memorizar son dos hallazgos validos.

    Poner un umbral aca seria inventar una hipotesis despues del hecho. El
    veredicto es MEASURED y el reporte lleva los numeros.
    """
    resultado = evaluate_level_3(
        [brazo("level_3", capture=0.7), brazo("level_3_lstm", capture=0.85)]
    )
    assert resultado.verdict is Verdict.MEASURED
    assert "level_3_lstm" in resultado.finding
    assert "sin criterio de aprobacion" in resultado.statement


# ---------------------------------------------------------------------------
# Nivel 4
# ---------------------------------------------------------------------------


def test_nivel_4_aprueba_al_converger_a_estar_invertido() -> None:
    resultado = evaluate_level_4(
        brazo(
            "level_4",
            time_invested=0.97,
            excess_log_growth_vs_always_long=0.001,
            turnover_annualized=0.08,
        ),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.PASS


def test_nivel_4_falla_si_el_agente_le_gana_al_ruido() -> None:
    """Ganarle a una serie sin senal es sobreajuste, no habilidad."""
    resultado = evaluate_level_4(
        brazo(
            "level_4",
            time_invested=0.95,
            excess_log_growth_vs_always_long=0.30,
            turnover_annualized=0.1,
        ),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.FAIL
    assert "le gana a una serie sin senal" in resultado.finding


def test_nivel_4_falla_si_no_converge_a_estar_invertido() -> None:
    """Sobre Heston estar afuera cuesta drift y no compra nada."""
    resultado = evaluate_level_4(
        brazo(
            "level_4",
            time_invested=0.30,
            excess_log_growth_vs_always_long=-0.2,
            turnover_annualized=5.0,
        ),
        UMBRALES,
    )
    assert resultado.verdict is Verdict.FAIL
    assert "No converge a estar invertido" in resultado.finding


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------


def test_el_protocolo_para_en_el_nivel_0_y_saltea_el_resto() -> None:
    """Si el nivel 0 falla, seguir solo produce numeros que hay que descartar."""
    reporte = assemble_protocol(
        UMBRALES,
        level_0=brazo("level_0", capture=0.2),
        level_1=brazos_snr({0.25: 0.9, 0.09: 0.8, 0.04: 0.7, 0.01: 0.6}),
        level_2=brazo("level_2", turnover_annualized=10.0),
        level_2_reference=brazo("level_1", turnover_annualized=40.0),
        level_3=[brazo("level_3")],
        level_4=brazo("level_4"),
    )
    assert reporte.stopped_at == 0
    veredictos = [n.verdict for n in reporte.levels]
    assert veredictos[0] is Verdict.FAIL
    assert all(v is Verdict.SKIPPED for v in veredictos[1:])
    assert "PROTOCOLO DETENIDO en el nivel 0" in reporte.render()


def test_los_niveles_salteados_aparecen_en_el_reporte() -> None:
    """Un reporte con tres niveles y sin explicacion se lee como si el protocolo
    tuviera tres niveles."""
    reporte = assemble_protocol(UMBRALES, level_0=brazo("level_0", capture=0.1))
    assert [n.level for n in reporte.levels] == [0, 1, 2, 3, 4]
    assert all("el nivel 0 fallo" in n.finding for n in reporte.levels[1:])


def test_el_protocolo_completo_no_reporta_parada() -> None:
    reporte = assemble_protocol(
        UMBRALES,
        level_0=brazo("level_0", capture=0.95),
        level_1=brazos_snr({0.25: 0.9, 0.09: 0.8, 0.04: 0.7, 0.01: 0.6}),
        level_2=brazo("level_2", turnover_annualized=10.0),
        level_2_reference=brazo("level_1", turnover_annualized=40.0),
        level_3=[brazo("level_3")],
        level_4=brazo(
            "level_4",
            time_invested=0.95,
            excess_log_growth_vs_always_long=0.0,
            turnover_annualized=0.08,
        ),
    )
    assert reporte.stopped_at is None
    assert [str(n.verdict) for n in reporte.levels] == [
        "PASS",
        "PASS",
        "PASS",
        "MEASURED",
        "PASS",
    ]


def test_los_umbrales_viajan_con_el_resultado() -> None:
    """Un umbral elegido despues de ver los numeros no es un criterio."""
    reporte = assemble_protocol(UMBRALES, level_0=brazo("level_0", capture=0.95))
    serializado = json.dumps(reporte.describe())
    assert "level_0_min_capture" in serializado
    assert json.loads(serializado)["thresholds"]["level_0_min_capture"] == 0.8


def test_el_brazo_se_reconstruye_desde_su_json() -> None:
    """Volver a aplicar los criterios no puede exigir reentrenar 10 semillas."""
    original = brazo("level_1_beta-0.3", capture=0.77)
    copia = ArmResult.from_dict(json.loads(json.dumps(original.describe())))
    assert copia.label == original.label
    assert copia.median("capture") == pytest.approx(0.77)
    assert copia.runs == original.runs


def test_beats_cuenta_semillas_y_no_compara_medianas() -> None:
    corridas = [corrida(s, total_return_mark=0.1 if s < 6 else 0.5) for s in SEMILLAS]
    arm = brazo("x")
    arm = ArmResult(
        label=arm.label,
        fixture_name=arm.fixture_name,
        fixture_config=arm.fixture_config,
        ppo_config=arm.ppo_config,
        runs=tuple(corridas),
        distributions={
            m: summarize(m, SEMILLAS, [getattr(c, m) for c in corridas])
            for m in _METRICAS
        },
        ceiling=arm.ceiling,
        baselines=arm.baselines,
    )
    # buy_and_hold rinde 0.2: lo superan las 4 semillas que dan 0.5.
    assert arm.beats("buy_and_hold") == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Punta a punta, sin torch
# ---------------------------------------------------------------------------


def test_run_arm_funciona_con_una_politica_inyectada() -> None:
    """Verifica el cableado: escalador en train, techo con el mismo safety.

    Se inyecta una politica constante en vez de PPO para que el test corra en el
    job principal de CI, que no instala torch.
    """
    fixture = level_1_noisy(500, seed=20240115)
    resultado = run_arm(
        fixture,
        label="prueba",
        seeds=[1, 2],
        ppo_config=PPOConfig(total_timesteps=1),
        policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0),
        allow_fewer_seeds=True,
    )
    assert len(resultado.runs) == 2
    assert sorted(resultado.baselines) == ["buy_and_hold", "ma_cross", "random"]
    assert resultado.distributions["capture"].below_minimum_seeds is True
    # Una politica constante de peso 1.0 es exactamente `always_long`, asi que
    # captura cero del camino entre esa referencia y el techo.
    assert resultado.median("capture") == pytest.approx(0.0, abs=0.02)
    assert resultado.median("time_invested") == pytest.approx(1.0)


def test_run_arm_falla_el_nivel_0_con_una_politica_que_no_aprende() -> None:
    """La verificacion de que el criterio del nivel 0 muerde de verdad."""
    fixture = level_0_deterministic(600)
    resultado = run_arm(
        fixture,
        label="level_0",
        seeds=[1, 2],
        ppo_config=PPOConfig(total_timesteps=1),
        policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0),
        allow_fewer_seeds=True,
    )
    veredicto = evaluate_level_0(resultado, UMBRALES)
    assert veredicto.verdict is Verdict.FAIL


def test_el_techo_del_brazo_usa_el_mismo_safety_que_el_sizer() -> None:
    """Si no, el 2% del margen de seguridad se leeria como fallo de aprendizaje."""
    fixture = level_1_noisy(500, seed=20240115)
    resultado = run_arm(
        fixture,
        label="prueba",
        seeds=[1],
        ppo_config=PPOConfig(total_timesteps=1),
        policy_factory=lambda _f, _s: ConstantWeightPolicy(1.0),
        allow_fewer_seeds=True,
        safety=0.98,
    )
    equity = np.asarray(resultado.runs[0].total_return_mark)
    referencia = resultado.ceiling["always_long_total_return"]
    assert float(equity) == pytest.approx(float(referencia), rel=1e-3)
