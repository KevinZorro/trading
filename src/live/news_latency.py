"""Retraso real de los lotes de GDELT segun el grabador en vivo (ADR 0012, 2.2).

El sello de la corrida del estudio exige el p99 del retraso entre el rotulo de un
lote y su recepcion, medido con los lotes que tienen ``last_404_at`` (los que el
grabador vio ausentes antes de verlos presentes: su recepcion queda acotada por el
sondeo, no por una caida del proceso).

ASSUMPTION: el retraso es ``received_at - label``, el techo del retraso de
disponibilidad: el lote aparecio como mucho ``poll_seconds`` antes de recibirse.
Los lotes sin ``last_404_at`` se excluyen: pueden ser recuperaciones tras una caida.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


class LatencyError(RuntimeError):
    """No hay datos para medir."""


def delays_minutes(paths: Iterable[Path]) -> list[float]:
    """``received_at - label`` en minutos, de los lotes ``ok`` con ``last_404_at``."""
    retrasos: list[float] = []
    for ruta in paths:
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            registro = json.loads(linea)
            if (
                registro.get("type") != "batch"
                or registro.get("status") != "ok"
                or registro.get("last_404_at") is None
            ):
                continue
            diferencia = pd.Timestamp(registro["received_at"]) - pd.Timestamp(
                registro["label"]
            )
            retrasos.append(diferencia.total_seconds() / 60.0)
    return retrasos


def p99_minutes(directory: Path) -> tuple[float, int]:
    """p99 del retraso y cuantos lotes lo sustentan, sobre ``news-*.jsonl``."""
    retrasos = delays_minutes(sorted(directory.glob("news-*.jsonl")))
    if not retrasos:
        raise LatencyError(
            f"no hay lotes con last_404_at en {directory}: no se puede medir el p99"
        )
    return float(np.percentile(retrasos, 99)), len(retrasos)
