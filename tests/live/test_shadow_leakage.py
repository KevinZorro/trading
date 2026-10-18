"""El modelo del snapshot solo ve barras cerradas hasta la hora del libro."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live.schedule import Due
from sim.clock import SimulatedClock

from .test_shadow import armar, t

pytestmark = pytest.mark.leakage


def test_la_barra_en_curso_no_entra_al_modelo(tmp_path: Path) -> None:
    # 17:30 del 17: la fila del 17 sigue abierta; su timestamp (cierre) es el 18.
    runner, _, _ = armar(tmp_path, "2026-10-17T17:30:00")
    r, _ = runner.take_snapshot(Due(t("2026-10-17T17:30:00"), "periodic"))
    assert r.valid
    assert r.bar_close == "2026-10-17T00:00:00.000Z"
    assert np.datetime64(r.bar_close.rstrip("Z"), "ns") <= np.datetime64(
        r.received_at.rstrip("Z"),
        "ns",  # type: ignore[union-attr]
    )


def test_reloj_adelantado_mas_de_una_barra_se_marca(tmp_path: Path) -> None:
    # Reloj del proceso en el 19 y el venue sirviendo hasta el 17: hueco de datos.
    runner, _, _ = armar(tmp_path, "2026-10-19T00:00:30")
    r, _ = runner.take_snapshot(Due(t("2026-10-19T00:00:30"), "daily"))
    assert not r.valid
    assert r.reasons[0].startswith("barra_vieja")


def test_reloj_es_el_inyectado(tmp_path: Path) -> None:
    runner, reloj, _ = armar(tmp_path, "2026-10-17T17:30:00")
    assert isinstance(reloj, SimulatedClock)
    r, _ = runner.take_snapshot(Due(t("2026-10-17T17:30:00"), "periodic"))
    assert r.taken_at == "2026-10-17T17:30:00.000Z"
