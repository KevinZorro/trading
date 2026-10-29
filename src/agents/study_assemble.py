"""Criterio del estudio A vs B vs placebo sobre los registros guardados (ADR 0012).

Secciones 6 y 7 del ADR.

No entrena nada: lee ``records.jsonl`` y ``SEAL.json`` de ``agents.study_run`` y aplica
las seis condiciones de APORTA, las tres etiquetas de lectura de un NO APORTA y el
veredicto secundario de A contra buy-and-hold (K1, "segundo toque del test").

``D_A(s,q) = X_B(s,q) - X_A(s,q)`` y ``D_P(s,q) = X_B(s,q) - X_placebo(s,q)``
por semilla; buy-and-hold se cancela. El DSR de ``B - A`` y de ``B - placebo``
usa la serie diaria
de excesos de cada brazo (las 2 200 barras de test concatenadas por fold), ``n_trials``
= 10 y la varianza del Sharpe entre semillas, como el ADR 0006 §2. Con ``n_trials`` = 1
se informa sin decidir. Un DSR indefinido cuenta como 0.0 (lectura conservadora).

ASSUMPTION: la serie diaria ``B - A`` es ``excess_daily`` de B menos ``excess_daily`` de
A, barra a barra y misma semilla: ambos son retorno del agente menos el de buy-and-hold
sobre las mismas barras, asi que el buy-and-hold se cancela.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agents.study_extras import (
    BLACKOUT_FOLD,
    blackout_fold,
    news_coverage,
    policy_sigma,
)
from agents.study_run import (
    ARM_A,
    ARM_B,
    ARM_PLACEBO,
    ARMS,
    StudyRunConfig,
    arm_series,
    load_news_table,
    read_records,
    record_files,
)
from agents.walkforward import (
    DSR_N_TRIALS,
    DSR_UMBRAL,
    FoldSeedRecord,
    WalkForwardError,
    _dsr,
    _sharpe,
    assemble,
    classify_regime,
    folds,
    load_dataset,
)
from eval.distribution import summarize

#: Configuraciones evaluadas sobre estos dias para A (C3 y R3): el numero honesto
#: del veredicto secundario de A (ADR 0012 §7).
N_TRIALS_A_HONESTO = 2
CONDITIONS: tuple[str, ...] = (
    "d_a_mark",
    "d_a_liq",
    "d_p_mark",
    "d_p_liq",
    "dsr_b_a",
    "dsr_b_placebo",
)


def _efectivo(valores: Sequence[float | None]) -> list[float]:
    return [0.0 if v is None or not np.isfinite(v) else float(v) for v in valores]


def k3_verdict(
    d_a_mark: Sequence[float],
    d_a_liq: Sequence[float],
    d_p_mark: Sequence[float],
    d_p_liq: Sequence[float],
    dsr_b_a: Sequence[float | None],
    dsr_b_placebo: Sequence[float | None],
) -> dict[str, Any]:
    """Las seis condiciones del ADR 0012 §6 y la etiqueta de lectura."""
    cumple = {
        "d_a_mark": float(np.median(d_a_mark)) > 0.0,
        "d_a_liq": float(np.median(d_a_liq)) > 0.0,
        "d_p_mark": float(np.median(d_p_mark)) > 0.0,
        "d_p_liq": float(np.median(d_p_liq)) > 0.0,
        "dsr_b_a": float(np.median(_efectivo(dsr_b_a))) >= DSR_UMBRAL,
        "dsr_b_placebo": float(np.median(_efectivo(dsr_b_placebo))) >= DSR_UMBRAL,
    }
    if all(cumple.values()):
        return {"verdict": "APORTA", "label": None, "conditions": cumple}
    if cumple["d_a_mark"] and cumple["d_a_liq"] and cumple["dsr_b_a"]:
        etiqueta = "efecto de forma"  # B le gana a A pero no al placebo
    elif float(np.median(d_a_mark)) < 0.0 and float(np.median(d_a_liq)) < 0.0:
        etiqueta = "B peor"
    else:
        etiqueta = "plano"
    return {"verdict": "NO APORTA", "label": etiqueta, "conditions": cumple}


def _por_semilla(
    registros: Sequence[FoldSeedRecord], n_folds: int
) -> dict[int, list[FoldSeedRecord]]:
    salida: dict[int, list[FoldSeedRecord]] = {}
    for r in sorted(registros, key=lambda r: (r.seed, r.fold)):
        salida.setdefault(r.seed, []).append(r)
    for semilla, filas in salida.items():
        if [r.fold for r in filas] != list(range(n_folds)):
            raise WalkForwardError(
                f"la semilla {semilla} no tiene los {n_folds} folds: "
                "la corrida esta incompleta"
            )
    return salida


def _x(filas: Sequence[FoldSeedRecord], campo: str) -> float:
    return float(sum(getattr(r, campo) for r in filas))


def _serie(filas: Sequence[FoldSeedRecord]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([x for r in filas for x in r.excess_daily], dtype=np.float64)


def assemble_study(
    records: Sequence[tuple[str, FoldSeedRecord]],
    regimes: Sequence[Any],
    *,
    n_folds: int,
    allow_fewer_seeds: bool = False,
) -> dict[str, Any]:
    """Aplica el criterio sellado sobre los registros de los tres brazos."""
    por_brazo = {
        a: _por_semilla([r for brazo, r in records if brazo == a], n_folds)
        for a in ARMS
    }
    semillas = sorted(por_brazo[ARM_A])
    for a in ARMS:
        if sorted(por_brazo[a]) != semillas:
            raise WalkForwardError(f"el brazo {a} no tiene las mismas semillas que A")

    def dist(nombre: str, valores: Sequence[float | None]) -> dict[str, object]:
        return summarize(
            nombre, semillas, list(valores), allow_fewer_seeds=allow_fewer_seeds
        ).describe()

    def x(brazo: str, campo: str) -> list[float]:
        return [_x(por_brazo[brazo][s], campo) for s in semillas]

    def diferencia(otro: str, campo: str) -> list[float]:
        return [b - o for b, o in zip(x(ARM_B, campo), x(otro, campo), strict=True)]

    def dsr_contra(otro: str) -> dict[str, list[float | None]]:
        excesos = {
            s: _serie(por_brazo[ARM_B][s]) - _serie(por_brazo[otro][s])
            for s in semillas
        }
        sharpes = [_sharpe(v.tolist()) for v in excesos.values()]
        finitos = [v for v in sharpes if np.isfinite(v)]
        varianza = float(np.var(finitos, ddof=1)) if len(finitos) > 1 else 0.0
        return {
            "n10": [
                _dsr(excesos[s].tolist(), varianza, DSR_N_TRIALS) for s in semillas
            ],
            "n1": [_dsr(excesos[s].tolist(), varianza, 1) for s in semillas],
        }

    dsr_a, dsr_p = dsr_contra(ARM_A), dsr_contra(ARM_PLACEBO)
    d = {
        "d_a_mark": diferencia(ARM_A, "e_mark"),
        "d_a_liq": diferencia(ARM_A, "e_liq"),
        "d_p_mark": diferencia(ARM_PLACEBO, "e_mark"),
        "d_p_liq": diferencia(ARM_PLACEBO, "e_liq"),
    }
    veredicto = k3_verdict(
        d["d_a_mark"],
        d["d_a_liq"],
        d["d_p_mark"],
        d["d_p_liq"],
        dsr_a["n10"],
        dsr_p["n10"],
    )
    arms = {
        a: assemble(
            [r for brazo, r in records if brazo == a],
            regimes,
            allow_fewer_seeds=allow_fewer_seeds,
        )
        for a in ARMS
    }
    return {
        "criterion": {
            **veredicto,
            "d": {k: dist(k, v) for k, v in d.items()},
            "dsr_b_a_n10": dist("dsr_b_a_n10", dsr_a["n10"]),
            "dsr_b_a_n1": dist("dsr_b_a_n1", dsr_a["n1"]),
            "dsr_b_placebo_n10": dist("dsr_b_placebo_n10", dsr_p["n10"]),
            "dsr_b_placebo_n1": dist("dsr_b_placebo_n1", dsr_p["n1"]),
            "threshold_dsr": DSR_UMBRAL,
            "n_trials": DSR_N_TRIALS,
        },
        # K1: segundo toque del test para el Agente A. Decide con n_trials = 10,
        # como C3; el numero honesto es 2 (C3 y R3).
        "agent_a_secondary": {
            "label": "segundo toque del test para el Agente A",
            "n_trials_honest": N_TRIALS_A_HONESTO,
            "main": arms[ARM_A]["main"],
            "sensitivity_commission_only": arms[ARM_A]["sensitivity_commission_only"],
        },
        "arms": {
            a: {
                "x_mark": dist(f"x_mark_{a}", x(a, "e_mark")),
                "x_liq": dist(f"x_liq_{a}", x(a, "e_liq")),
                "x_exposure_matched": dist(f"x_exp_{a}", x(a, "t_mark")),
                "action_std": dist(
                    f"action_std_{a}",
                    [
                        float(np.mean([r.action_std for r in por_brazo[a][s]]))
                        for s in semillas
                    ],
                ),
                "per_fold": arms[a]["per_fold"],
                "sensitivity_commission_only": arms[a]["sensitivity_commission_only"],
            }
            for a in ARMS
        },
    }


def run_assemble(root: Path, out_dir: Path) -> dict[str, Any]:
    """Lee la corrida sellada, aplica el criterio y escribe ``summary.json``."""
    sello = json.loads((out_dir / "SEAL.json").read_text(encoding="utf-8"))
    archivos = record_files(out_dir)
    registros = read_records(out_dir)
    serie, _, dataset_sha = load_dataset(root)
    ventanas = folds(len(serie), StudyRunConfig().walk)
    reporte = assemble_study(
        registros,
        [classify_regime(serie, v) for v in ventanas],
        n_folds=len(ventanas),
        allow_fewer_seeds=len({r.seed for _, r in registros}) < 10,
    )
    pocas = len({r.seed for _, r in registros}) < 10
    tabla, _ = load_news_table(root, StudyRunConfig())
    brazos = {a: arm_series(a, serie, tabla, StudyRunConfig()) for a in ARMS}
    reporte["news_coverage"] = news_coverage(brazos, ventanas)
    reporte["blackout_fold"] = blackout_fold(
        registros,
        brazos[ARM_B],
        ventanas[BLACKOUT_FOLD],
        allow_fewer_seeds=pocas,
    )
    reporte["policy_sigma"] = policy_sigma(registros, allow_fewer_seeds=pocas)
    reporte["provenance"] = {
        "seal": sello,
        "dataset_sha256": dataset_sha,
        "records_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in archivos
        },
        "n_records": len(registros),
        "limitations": (
            "N = 1 camino real: CONSISTENTE es el maximo (ADR 0012 §0 y §11)"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(reporte, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (out_dir / "SEAL.json", *archivos, out_dir / "summary.json")
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return reporte


def describe_conditions(resultado: Mapping[str, Any]) -> str:
    """Una linea por condicion, para imprimir en la consola."""
    return "\n".join(
        f"  {'OK ' if resultado['conditions'][c] else 'NO '} {c}" for c in CONDITIONS
    )
