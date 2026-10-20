"""Anti-leakage del grabador: nunca pide ni registra algo que aun no existe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.live.test_news import MIN, montar, registros, t

pytestmark = pytest.mark.leakage


def test_nunca_pide_un_rotulo_mas_alla_de_lead_minutes_en_el_futuro(
    tmp_path: Path,
) -> None:
    rotulos = ("20261020003000", "20261020004500", "20261020010000", "20261020011500")
    rec, _, srv = montar(
        tmp_path, "2026-10-20T00:40:00", {r: "2026-10-20T00:00:00" for r in rotulos}
    )
    rec.run(lambda: len(registros(tmp_path)) >= 5)  # cabecera + 4 lotes
    assert srv.pedidos
    for cuando, rotulo in srv.pedidos:
        rotulo_t = np.datetime64(
            f"{rotulo[:4]}-{rotulo[4:6]}-{rotulo[6:8]}T{rotulo[8:10]}:{rotulo[10:12]}:{rotulo[12:]}",
            "ns",
        )
        assert rotulo_t - cuando <= 15 * MIN


def test_received_at_es_la_hora_del_reloj_al_recibir_no_el_rotulo(
    tmp_path: Path,
) -> None:
    # Un lote rotulado 01:15 que aparece a las 01:03:41 NO se registra como recibido
    # a las 01:15: pub_time lo decide la alineacion, aqui solo queda el hecho medido.
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:00:00", {"20261020011500": "2026-10-20T01:03:41"}
    )
    r = rec.collect(t("2026-10-20T01:15:00"))
    assert r["received_at"] == "2026-10-20T01:04:00.000Z"
    assert r["received_at"] < r["label"]


def test_ultimo_404_siempre_es_anterior_a_la_recepcion(tmp_path: Path) -> None:
    rec, _, _ = montar(
        tmp_path, "2026-10-20T01:00:00", {"20261020011500": "2026-10-20T01:07:30"}
    )
    r = rec.collect(t("2026-10-20T01:15:00"))
    assert isinstance(r["last_404_at"], str)
    assert isinstance(r["received_at"], str)
    assert r["last_404_at"] < r["received_at"]
