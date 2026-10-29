"""Criterio del estudio A vs B vs placebo (ADR 0012 §6 y §7), con oraculos a mano."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agents.policy import ConstantWeightPolicy
from agents.study_assemble import assemble_study, k3_verdict, run_assemble
from agents.study_run import StudyRunConfig, run_study
from agents.walkforward import FoldSeedRecord, RegimeClass, WalkForwardError

RAIZ = Path(__file__).resolve().parents[2]
POS, NEG = [0.1] * 10, [-0.1] * 10
ALTO, BAJO = [0.99] * 10, [0.5] * 10


def test_aporta_si_se_cumplen_las_seis() -> None:
    r = k3_verdict(POS, POS, POS, POS, ALTO, ALTO)
    assert r["verdict"] == "APORTA"
    assert r["label"] is None
    assert all(r["conditions"].values())


def test_la_mediana_manda_no_el_mejor_seed() -> None:
    una_buena = [5.0, *NEG[:9]]
    r = k3_verdict(una_buena, POS, POS, POS, ALTO, ALTO)
    assert r["verdict"] == "NO APORTA"
    assert r["conditions"]["d_a_mark"] is False


def test_efecto_de_forma_gana_a_a_pero_no_al_placebo() -> None:
    r = k3_verdict(POS, POS, NEG, NEG, ALTO, BAJO)
    assert (r["verdict"], r["label"]) == ("NO APORTA", "efecto de forma")


def test_b_peor_si_pierde_contra_a_en_las_dos_series() -> None:
    r = k3_verdict(NEG, NEG, POS, POS, BAJO, ALTO)
    assert r["label"] == "B peor"


def test_plano_en_cualquier_otro_caso() -> None:
    r = k3_verdict(POS, NEG, POS, POS, BAJO, BAJO)
    assert r["label"] == "plano"


def test_un_dsr_indefinido_cuenta_como_no_significativo() -> None:
    r = k3_verdict(POS, POS, POS, POS, [None] * 10, ALTO)
    assert r["verdict"] == "NO APORTA"
    assert r["conditions"]["dsr_b_a"] is False


def registro(fold: int, seed: int, e: float, exceso: float) -> FoldSeedRecord:
    metricas: dict[str, Any] = {
        "e_mark": e,
        "e_liq": e,
        "t_mark": 0.0,
        "t_liq": 0.0,
        "exposure": 0.5,
        "n_orders": 1,
        "n_rejected_min_notional": 0,
        "n_rejected_other": 0,
        "rejected_by_reason": {},
        "baselines": {},
        "excess_daily": [exceso, -exceso, exceso * 2, 0.0],
    }
    return FoldSeedRecord(
        fold=fold,
        seed=seed,
        exposure_first=0.5,
        exposure_rest=0.5,
        action_std=0.1,
        n_fills=1,
        n_outliers=0,
        e_mark_sin_outliers=e,
        sensitivity=metricas,
        **{k: v for k, v in metricas.items() if k not in ("baselines",)},
        baselines={},
    )


def mundo(e: dict[str, float], semillas: int = 3, folds: int = 2) -> Any:
    registros = [
        (arm, registro(f, s, e[arm], e[arm] * (s + 1)))
        for arm in ("A", "B", "placebo")
        for s in range(semillas)
        for f in range(folds)
    ]
    regimenes = [RegimeClass(f, 0.0, 0.0, 0.0, "ambigua") for f in range(folds)]
    return registros, regimenes


def test_d_es_la_diferencia_de_x_por_semilla() -> None:
    """X = suma sobre 2 folds de e. B 0.30 vs A 0.10 por fold: X_B - X_A = 0.4."""
    registros, regimenes = mundo({"A": 0.10, "B": 0.30, "placebo": 0.20})
    r = assemble_study(registros, regimenes, n_folds=2, allow_fewer_seeds=True)
    d = r["criterion"]["d"]
    assert d["d_a_mark"]["median"] == pytest.approx(0.4)
    assert d["d_p_mark"]["median"] == pytest.approx(0.2)
    assert d["d_a_liq"]["median"] == pytest.approx(0.4)
    assert r["arms"]["B"]["x_mark"]["median"] == pytest.approx(0.6)
    assert r["agent_a_secondary"]["n_trials_honest"] == 2


def test_una_corrida_incompleta_falla_en_vez_de_juzgar() -> None:
    registros, regimenes = mundo({"A": 0.1, "B": 0.1, "placebo": 0.1})
    with pytest.raises(WalkForwardError, match="incompleta"):
        assemble_study(registros[:-1], regimenes, n_folds=2, allow_fewer_seeds=True)


def test_semillas_distintas_entre_brazos_fallan() -> None:
    registros, regimenes = mundo({"A": 0.1, "B": 0.1, "placebo": 0.1})
    sin_b = [x for x in registros if not (x[0] == "B" and x[1].seed == 2)]
    with pytest.raises(WalkForwardError, match="mismas semillas"):
        assemble_study(sin_b, regimenes, n_folds=2, allow_fewer_seeds=True)


def test_ensamblado_sobre_una_corrida_real_escribe_resumen_y_manifest(
    tmp_path: Path,
) -> None:
    """Tres brazos con la misma politica constante: D = 0, NO APORTA plano."""
    base = StudyRunConfig()
    config = StudyRunConfig(walk=replace(base.walk, seeds=(1, 2)))
    run_study(
        RAIZ,
        tmp_path,
        lambda _o: lambda _make_env, _seed: ConstantWeightPolicy(1.0),
        config=config,
        run_commit="abc",
        live_latency_p99_min=5.0,
    )
    reporte = run_assemble(RAIZ, tmp_path)
    assert reporte["criterion"]["verdict"] == "NO APORTA"
    assert reporte["criterion"]["label"] == "plano"
    for clave in ("d_a_mark", "d_a_liq", "d_p_mark", "d_p_liq"):
        assert reporte["criterion"]["d"][clave]["median"] == 0.0
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert set(manifest) == {"SEAL.json", "records.jsonl", "summary.json"}
    resumen = json.loads((tmp_path / "summary.json").read_text())
    assert resumen["provenance"]["seal"]["commit"] == "abc"
    assert resumen["provenance"]["n_records"] == 66
