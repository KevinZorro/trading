"""Anti-leakage del estudio A vs B (ADR 0012): lo que cada brazo ve en la barra t."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agents.study_run import (
    ARM_A,
    ARM_B,
    ARM_PLACEBO,
    StudyRunConfig,
    arm_series,
    load_news_table,
)
from agents.walkforward import load_dataset
from data.schema import BarSeries
from features.news import NewsFeatureTable

pytestmark = pytest.mark.leakage

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mundo() -> tuple[BarSeries, NewsFeatureTable]:
    serie, _, _ = load_dataset(RAIZ)
    tabla, _ = load_news_table(RAIZ, StudyRunConfig())
    return serie, tabla


def test_a_no_trae_columnas_de_noticias(
    mundo: tuple[BarSeries, NewsFeatureTable],
) -> None:
    serie, tabla = mundo
    assert arm_series(ARM_A, serie, tabla, StudyRunConfig()).exog is None


def test_el_placebo_es_exactamente_b_364_dias_antes(
    mundo: tuple[BarSeries, NewsFeatureTable],
) -> None:
    """Oraculo a mano: la fila ``i`` del placebo es la fila ``i - 364`` de B (la
    serie es diaria sin huecos). Solo mira pasado: no hay forma de que adelante."""
    serie, tabla = mundo
    config = StudyRunConfig()
    b = arm_series(ARM_B, serie, tabla, config)
    p = arm_series(ARM_PLACEBO, serie, tabla, config)
    assert b.exog is not None
    assert p.exog is not None
    np.testing.assert_array_equal(p.exog[364:], b.exog[:-364])
    assert p.exog_names == b.exog_names
    # Sobre las barras en que ambos tienen valor, B y placebo no son lo mismo.
    assert not np.array_equal(p.exog[364:], b.exog[364:], equal_nan=True)
