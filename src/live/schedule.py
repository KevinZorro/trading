"""Cadencia de la sombra (ADR 0007, seccion 4). Funciones puras: sin reloj ni I/O.

Un snapshot cada ``snapshot_minutes`` alineado a la hora (``:00``, ``:15``,
``:30``, ``:45`` UTC) y uno dedicado a las ``daily_snapshot_utc`` (00:00:30), el
comparable directo con el backtest. El segundo nunca coincide con la grilla de
minutos enteros, asi que cada instante tiene un unico ``kind``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Kind = Literal["periodic", "daily"]
_DIA = np.timedelta64(1, "D")


@dataclass(frozen=True)
class Due:
    at: np.datetime64
    kind: Kind


def parse_daily(texto: str) -> np.timedelta64:
    """``HH:MM:SS`` -> segundos desde las 00:00 UTC."""
    try:
        h, m, s = (int(x) for x in texto.split(":"))
    except ValueError as exc:
        raise ValueError(f"daily_snapshot_utc debe ser HH:MM:SS, es {texto!r}") from exc
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"daily_snapshot_utc fuera de rango: {texto!r}")
    return np.timedelta64(h * 3600 + m * 60 + s, "s")


def next_due(now: np.datetime64, snapshot_minutes: int, daily_snapshot_utc: str) -> Due:
    """Primer instante **estrictamente posterior** a ``now``.

    Si un snapshot se demora mas que el paso, los instantes salteados no se
    recuperan: quedan como hueco, visible en los datos.
    """
    ahora = np.datetime64(now, "ns")
    paso = np.timedelta64(snapshot_minutes, "m").astype("timedelta64[ns]")
    periodico = (ahora.astype("int64") // paso.astype("int64") + 1) * paso.astype(
        "int64"
    )
    periodica = np.datetime64(int(periodico), "ns")
    diaria = ahora.astype("datetime64[D]").astype("datetime64[ns]") + parse_daily(
        daily_snapshot_utc
    )
    if diaria <= ahora:
        diaria = diaria + _DIA
    return Due(diaria, "daily") if diaria < periodica else Due(periodica, "periodic")
