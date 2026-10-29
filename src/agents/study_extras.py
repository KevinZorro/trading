"""Lo que el ADR 0012 §8 pide ademas del criterio: cobertura de noticias en el test,
el fold del apagon de 2025 con y sin esos dias, y el sigma de la politica.

Todo es descriptivo y sale de los registros guardados y de la tabla de features; no
decide el veredicto.

ASSUMPTION: "el resultado del fold 9 con y sin los 17 dias del apagon" se mide como la
suma del exceso diario ``B - A`` del fold (retorno simple del agente menos el de
buy-and-hold, barra a barra), con todas las barras y sin las de ``coverage_ok`` = 0.
No es el ``X`` log del criterio: quitar barras de una serie de retornos no define un
crecimiento compuesto, y la suma es lo que sobrevive a quitarlas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from agents.study_run import ARM_A, ARM_B, ARM_PLACEBO
from agents.walkforward import FoldSeedRecord
from data.schema import BarSeries, FloatArray
from eval.distribution import summarize
from eval.walkforward import Window

#: El fold cuyo test contiene el apagon de 2025-06-14 a 2025-07-02 (ADR 0012 §2.5).
BLACKOUT_FOLD = 9


def _columna(series: BarSeries, nombre: str) -> FloatArray:
    if series.exog is None or nombre not in series.exog_names:
        raise ValueError(f"la serie no trae la columna {nombre!r}")
    return np.asarray(series.exog[:, series.exog_names.index(nombre)], dtype=np.float64)


def news_coverage(
    brazos: dict[str, BarSeries], ventanas: Sequence[Window]
) -> dict[str, Any]:
    """Fraccion de barras de test con ``coverage_ok`` = 0 y con ``no_title`` = 1,
    por brazo con noticias (B y placebo) y por fold."""
    salida: dict[str, Any] = {}
    for brazo in (ARM_B, ARM_PLACEBO):
        cobertura = _columna(brazos[brazo], "coverage_ok")
        sin_titulo = _columna(brazos[brazo], "no_title")
        por_fold = {
            v.index: {
                "coverage_ok_0": float(np.mean(cobertura[slice(*v.test)] == 0.0)),
                "no_title_1": float(np.mean(sin_titulo[slice(*v.test)] == 1.0)),
            }
            for v in ventanas
        }
        tests = np.concatenate([np.arange(*v.test) for v in ventanas])
        salida[brazo] = {
            "total": {
                "coverage_ok_0": float(np.mean(cobertura[tests] == 0.0)),
                "no_title_1": float(np.mean(sin_titulo[tests] == 1.0)),
            },
            "per_fold": por_fold,
        }
    return salida


def blackout_fold(
    registros: Sequence[tuple[str, FoldSeedRecord]],
    b_series: BarSeries,
    ventana: Window,
    *,
    allow_fewer_seeds: bool = False,
) -> dict[str, Any]:
    """Suma del exceso diario ``B - A`` del fold, con todas las barras y sin las de
    ``coverage_ok`` = 0 de B. Secundario: no decide."""
    cobertura = _columna(b_series, "coverage_ok")[slice(*ventana.test)]
    sanas = cobertura != 0.0
    por_brazo = {
        a: {r.seed: np.asarray(r.excess_daily) for b, r in registros if b == a}
        for a in (ARM_A, ARM_B)
    }
    # ``excess_daily`` cubre las barras de test salvo la primera (retorno de la barra
    # ``test.start`` en adelante): se alinea con las ultimas ``len`` de la ventana.
    n = len(next(iter(por_brazo[ARM_B].values())))
    sanas = sanas[-n:]
    semillas = sorted(por_brazo[ARM_B])
    diario = {s: por_brazo[ARM_B][s] - por_brazo[ARM_A][s] for s in semillas}

    def dist(nombre: str, valores: list[float | None]) -> dict[str, object]:
        return summarize(
            nombre, semillas, valores, allow_fewer_seeds=allow_fewer_seeds
        ).describe()

    return {
        "fold": ventana.index,
        "n_bars": int(n),
        "n_bars_without_coverage": int((~sanas).sum()),
        "sum_b_minus_a_all": dist("todas", [float(diario[s].sum()) for s in semillas]),
        "sum_b_minus_a_covered": dist(
            "con_cobertura", [float(diario[s][sanas].sum()) for s in semillas]
        ),
    }


def policy_sigma(
    registros: Sequence[tuple[str, FoldSeedRecord]],
    *,
    allow_fewer_seeds: bool = False,
) -> dict[str, Any]:
    """Sigma de la politica por brazo: mediana entre folds de cada semilla, y luego
    la distribucion entre semillas. ``{}`` si la politica no la tiene (tests)."""
    salida: dict[str, Any] = {}
    for brazo in (ARM_A, ARM_B, ARM_PLACEBO):
        por_semilla: dict[int, list[float]] = {}
        for b, r in registros:
            if b == brazo and r.policy_sigma is not None:
                por_semilla.setdefault(r.seed, []).append(r.policy_sigma)
        if por_semilla:
            semillas = sorted(por_semilla)
            salida[brazo] = summarize(
                f"policy_sigma_{brazo}",
                semillas,
                [float(np.median(por_semilla[s])) for s in semillas],
                allow_fewer_seeds=allow_fewer_seeds,
            ).describe()
    return salida
