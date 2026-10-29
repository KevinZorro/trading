"""Extras del ADR 0012 §8: cobertura de noticias, fold del apagon y sigma."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.policy import ConstantWeightPolicy
from agents.ppo import SB3Policy
from agents.study_extras import blackout_fold, news_coverage, policy_sigma
from agents.study_run import StudyRunConfig, Unit, policy_path, run_study
from agents.walkforward import load_dataset
from data.schema import BarSeries
from eval.walkforward import Window
from tests.agents.test_study_assemble import registro

RAIZ = Path(__file__).resolve().parents[2]
NOMBRES = ("coverage_ok", "no_title")


def serie_con(coverage: list[float], titulo: list[float]) -> BarSeries:
    base, _, _ = load_dataset(RAIZ)
    n = len(base)
    exog = np.zeros((n, 2))
    exog[: len(coverage), 0] = coverage
    exog[: len(titulo), 1] = titulo
    return replace(base, exog=exog, exog_names=NOMBRES)


def test_cobertura_es_la_fraccion_de_barras_de_test_a_mano() -> None:
    # Test del fold 0 = barras 4..7 (4 barras): 1 de 4 sin cobertura, 2 de 4 sin titulo.
    ventana = Window(index=0, train=(0, 2), validation=(2, 4), test=(4, 8))
    serie = serie_con(
        [1.0] * 4 + [1.0, 0.0, 1.0, 1.0], [0.0] * 4 + [1.0, 1.0, 0.0, 0.0]
    )
    r = news_coverage({"B": serie, "placebo": serie}, [ventana])
    assert r["B"]["total"] == {"coverage_ok_0": 0.25, "no_title_1": 0.5}
    assert r["placebo"]["per_fold"][0] == {"coverage_ok_0": 0.25, "no_title_1": 0.5}


def test_el_fold_del_apagon_con_y_sin_las_barras_sin_cobertura() -> None:
    """Test = 5 barras; el exceso diario cubre las ultimas 4. Cobertura 0 en la
    penultima barra: esa barra sale de la segunda suma."""
    ventana = Window(index=9, train=(0, 2), validation=(2, 4), test=(4, 9))
    serie = serie_con([1.0] * 4 + [1.0, 1.0, 1.0, 0.0, 1.0], [0.0] * 9)
    registros = []
    for brazo, exceso in (("A", [0.0, 0.0, 0.0, 0.0]), ("B", [0.1, 0.2, 0.3, 0.4])):
        for semilla in range(3):
            r = registro(9, semilla, 0.0, 0.0)
            registros.append((brazo, replace(r, excess_daily=tuple(exceso))))
    d = blackout_fold(registros, serie, ventana, allow_fewer_seeds=True)
    assert d["n_bars"] == 4
    assert d["n_bars_without_coverage"] == 1
    assert d["sum_b_minus_a_all"]["median"] == pytest.approx(1.0)
    # Las barras alineadas son 5..8; la 7 (cobertura 0) es el tercer valor: 0.3.
    assert d["sum_b_minus_a_covered"]["median"] == pytest.approx(0.7)


def test_sigma_vacio_si_la_politica_no_la_tiene() -> None:
    registros = [("B", registro(0, 1, 0.0, 0.0))]
    assert policy_sigma(registros, allow_fewer_seeds=True) == {}


def test_sigma_es_la_mediana_entre_folds_de_cada_semilla() -> None:
    registros = [
        ("B", replace(registro(f, s, 0.0, 0.0), policy_sigma=v))
        for s in range(3)
        for f, v in enumerate((1.0, 3.0, 2.0))
    ]
    r = policy_sigma(registros, allow_fewer_seeds=True)
    assert r["B"]["median"] == pytest.approx(2.0)
    assert set(r) == {"B"}


class _Parametro:
    def __init__(self, valores: list[float]) -> None:
        self._v = np.array(valores)

    def detach(self) -> _Parametro:
        return self

    def cpu(self) -> _Parametro:
        return self

    def numpy(self) -> Any:
        return self._v


class _ModeloFalso:
    """Lo minimo de un modelo SB3: ``predict``, ``save`` y ``policy.log_std``."""

    def __init__(self) -> None:
        self.policy = type("P", (), {"log_std": _Parametro([0.0])})()

    def predict(self, observation: Any, deterministic: bool) -> Any:
        return np.array([1.0]), None

    def save(self, path: str) -> None:
        Path(path + ".zip").write_text("politica")


def test_run_study_guarda_solo_las_politicas_de_b_de_las_semillas_elegidas(
    tmp_path: Path,
) -> None:
    base = StudyRunConfig()
    config = StudyRunConfig(
        walk=replace(base.walk, seeds=(1, 2)), saved_policy_seeds=(1,)
    )
    run_study(
        RAIZ,
        tmp_path,
        lambda _o: lambda _m, _s: SB3Policy(_ModeloFalso(), recurrent=False),
        config=config,
        run_commit="abc",
        live_latency_p99_min=5.0,
    )
    guardadas = sorted(p.name for p in (tmp_path / "policies").iterdir())
    assert guardadas == sorted(f"B-fold{f}-seed1.zip" for f in range(11))
    assert policy_path(tmp_path, Unit("B", 0, 1)).with_suffix(".zip").exists()
    assert not policy_path(tmp_path, Unit("A", 0, 1)).with_suffix(".zip").exists()
    assert not policy_path(tmp_path, Unit("B", 0, 2)).with_suffix(".zip").exists()
    assert (tmp_path / "SEAL.json").exists()


def test_sin_politica_sb3_no_se_crea_la_carpeta_de_politicas(tmp_path: Path) -> None:
    base = StudyRunConfig()
    config = StudyRunConfig(walk=replace(base.walk, seeds=(1,)))
    run_study(
        RAIZ,
        tmp_path,
        lambda _o: lambda _m, _s: ConstantWeightPolicy(1.0),
        config=config,
        run_commit="abc",
        live_latency_p99_min=5.0,
    )
    assert not (tmp_path / "policies").exists()
