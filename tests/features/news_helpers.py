"""Agregados por lote sinteticos con contenido conocido, para los tests de N4."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grilla(dias: int, inicio: str = "2020-01-01", **cols: object) -> pd.DataFrame:
    """``dias`` dias de lotes ``ok``: total=100, tema=2, tono_sum=-4, 1 titulo."""
    rotulos = pd.date_range(inicio, periods=dias * 96, freq="15min")
    frame = pd.DataFrame(
        {
            "label": rotulos,
            "status": "ok",
            "rows_total": 100,
            "rows_malformed": 0,
            "rows_theme": 2,
            "tone_sum": -4.0,
            "n_titled": 1,
            "last_modified_offset_s": -600,
        }
    )
    for nombre, valor in cols.items():
        frame[nombre] = valor
    return frame


def cierre(dia: str) -> np.datetime64:
    return np.datetime64(dia, "ns")
