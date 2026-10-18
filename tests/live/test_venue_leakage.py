"""El venue no llena una orden contra un libro anterior a su decision."""

from __future__ import annotations

from pathlib import Path

import pytest

from live.venue import ShadowVenueError

from .test_venue import LIBRO, armar, libro, orden

pytestmark = pytest.mark.leakage


def test_libro_anterior_a_la_orden_es_un_error(tmp_path: Path) -> None:
    viejo = libro([[100.0, 1.0]], [[101.0, 1.0]], cuando="00:00:10")
    venue, _, _ = armar(tmp_path, viejo)
    venue.submit_order(orden(0.5))  # reloj en 00:00:30
    with pytest.raises(ShadowVenueError, match="anterior a la orden"):
        venue.settle()
    assert venue.get_fills() == []
    assert venue.get_positions() == {}


def test_el_fill_nunca_es_anterior_a_la_decision(tmp_path: Path) -> None:
    venue, _, _ = armar(tmp_path, LIBRO)
    venue.submit_order(orden(1.0))
    [f] = venue.settle()
    assert f.timestamp_fill >= f.timestamp_decision
