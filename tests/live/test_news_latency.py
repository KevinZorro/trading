"""p99 del retraso de los lotes en vivo (ADR 0012, 2.2), con oraculos a mano."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from live.news_latency import LatencyError, delays_minutes, p99_minutes


def lote(
    label: str, recibido: str, *, ultimo_404: str | None, status: str = "ok"
) -> str:
    return json.dumps(
        {
            "type": "batch",
            "label": label,
            "status": status,
            "last_404_at": ultimo_404,
            "received_at": recibido,
        }
    )


def test_solo_cuentan_los_lotes_ok_con_last_404(tmp_path: Path) -> None:
    ruta = tmp_path / "news-2026-10-20.jsonl"
    ruta.write_text(
        "\n".join(
            [
                json.dumps({"type": "header"}),
                lote("2026-10-20T01:00:00Z", "2026-10-20T01:05:00Z", ultimo_404="x"),
                lote("2026-10-20T01:15:00Z", "2026-10-20T02:00:00Z", ultimo_404=None),
                lote("2026-10-20T01:30:00Z", "2026-10-20T01:40:00Z", ultimo_404="x"),
                lote(
                    "2026-10-20T01:45:00Z",
                    "2026-10-20T02:00:00Z",
                    ultimo_404="x",
                    status="missing",
                ),
            ]
        )
    )
    assert delays_minutes([ruta]) == [5.0, 10.0]


def test_p99_sobre_cien_lotes_a_mano(tmp_path: Path) -> None:
    # 99 lotes con 1 minuto y uno con 46: el percentil 99 interpola a 1 + 0.01*45.
    filas = [
        lote(
            "2026-10-20T00:00:00Z",
            "2026-10-20T00:01:00Z" if i < 99 else "2026-10-20T00:46:00Z",
            ultimo_404="x",
        )
        for i in range(100)
    ]
    (tmp_path / "news-2026-10-20.jsonl").write_text("\n".join(filas))
    p99, n = p99_minutes(tmp_path)
    assert n == 100
    assert p99 == pytest.approx(1.45)


def test_sin_datos_falla_con_su_motivo(tmp_path: Path) -> None:
    with pytest.raises(LatencyError, match="no se puede medir"):
        p99_minutes(tmp_path)
