"""ShadowRunner contra las respuestas grabadas de Binance (sin red).

El libro grabado es de 2026-10-17T17:17:09Z: mejor bid 81261.01, mejor ask
81261.02, con 8.34 BTC en el primer ask, asi que 10 USDT se llenan en el primer
nivel. Las klines: la ultima fila (apertura 2026-10-17 00:00) estaba en curso;
la ultima **cerrada** abrio el 16 y cierra el 17 a las 00:00:00
(1792108800000 + 86 400 000 ms = 1792195200000 = 2026-10-17T00:00:00Z).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from live.config import ShadowConfig
from live.schedule import Due
from live.shadow import ADR_PATH, ShadowRunner, build_runner
from live.store import StoreError
from sim.clock import SimulatedClock

from .conftest import Servidor, grabacion


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


def armar(
    tmp_path: Path, inicio: str, servidor: Servidor | None = None
) -> tuple[ShadowRunner, SimulatedClock, list[float]]:
    reloj = SimulatedClock(t(inicio))
    pausas: list[float] = []

    def dormir(s: float) -> None:
        pausas.append(s)
        reloj.advance_to(reloj.now() + np.timedelta64(round(s * 1e9), "ns"))

    runner = build_runner(
        ShadowConfig(data_dir=tmp_path),
        reloj,
        sleep=dormir,
        fetch=servidor or Servidor(),
    )
    return runner, reloj, pausas


def registros(tmp_path: Path, dia: str) -> list[dict[str, object]]:
    ruta = tmp_path / f"snapshots-{dia}.jsonl"
    return [json.loads(x) for x in ruta.read_text().splitlines()]


def test_snapshot_valido_con_oraculo_a_mano(tmp_path: Path) -> None:
    runner, _, _ = armar(tmp_path, "2026-10-17T17:30:00")
    r, _ = runner.take_snapshot(Due(t("2026-10-17T17:30:00"), "periodic"))
    assert r.valid
    assert r.reasons == []
    assert r.bar_close == "2026-10-17T00:00:00.000Z"
    assert r.mid == pytest.approx(81261.015)
    assert len(r.cells) == 6  # 3 tamanos x 2 lados
    compra10 = r.cells[0]
    assert (compra10["size"], compra10["side"]) == (10.0, "buy")
    # un tick: medio spread = 0.005 / mid; 10 USDT caben en el primer nivel
    assert compra10["real"]["half_spread_rel"] == pytest.approx(0.005 / 81261.015)
    assert compra10["real"]["depth_impact_rel"] == 0.0
    assert compra10["ratios"]["ratio_spread"] == pytest.approx(
        compra10["real"]["half_spread_rel"] / compra10["model"]["half_spread_rel"]
    )


def test_kline_cerrada_aun_no_publicada_es_barra_vieja(tmp_path: Path) -> None:
    # A las 00:00:30 del 18 el venue todavia sirve hasta la barra que cierra el 17.
    klines = copy.deepcopy(grabacion("klines_1d.json"))[:-1]  # sin la del 17
    runner, _, _ = armar(tmp_path, "2026-10-18T00:00:30", Servidor(klines_1d=klines))
    r, _ = runner.take_snapshot(Due(t("2026-10-18T00:00:30"), "daily"))
    assert not r.valid
    assert r.bar_close == "2026-10-17T00:00:00.000Z"
    assert len(r.reasons) == 1
    assert r.reasons[0].startswith("barra_vieja")
    assert "2026-10-18T00:00:00.000Z" in r.reasons[0]
    # el libro es sano: el costo real se guarda, sin modelo ni cocientes
    assert len(r.cells) == 6
    assert all(c["model"] is None and c["ratios"] is None for c in r.cells)


def test_falla_de_red_se_guarda_y_el_proceso_sigue(tmp_path: Path) -> None:
    servidor = Servidor()
    original = servidor.__call__

    def fallar(url: str) -> bytes:
        if "depth" in url:
            raise OSError("timeout")
        return original(url)

    runner, _, _ = armar(tmp_path, "2026-10-17T17:30:00", servidor)
    runner._market._fetch = fallar  # type: ignore[attr-defined]
    r = runner.step(Due(t("2026-10-17T17:45:00"), "periodic"))
    assert not r.valid
    assert r.cells == []
    assert r.reasons[0].startswith("depth:")
    assert not (
        tmp_path / "book-2026-10-17.jsonl.gz"
    ).exists()  # sin libro, sin archivo
    beat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert beat["consecutive_invalid"] == 1
    assert beat["last_valid_scheduled"] is None


def test_run_sigue_la_cadencia_y_actualiza_el_heartbeat(tmp_path: Path) -> None:
    runner, _, pausas = armar(tmp_path, "2026-10-17T23:59:00")
    cuenta = {"n": 0}

    def parar() -> bool:
        cuenta["n"] += 1
        return cuenta["n"] > 6  # tres vueltas completas

    runner.run(parar)
    assert pausas == [60.0, 30.0, 870.0]  # 00:00:00, 00:00:30, 00:15:00
    filas = registros(tmp_path, "2026-10-18")
    assert filas[0]["type"] == "header"
    assert [(f["kind"], f["scheduled"]) for f in filas[1:]] == [
        ("periodic", "2026-10-18T00:00:00.000Z"),
        ("daily", "2026-10-18T00:00:30.000Z"),
        ("periodic", "2026-10-18T00:15:00.000Z"),
    ]
    beat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert beat["last_scheduled"] == "2026-10-18T00:15:00.000Z"
    assert beat["at"] == "2026-10-18T00:15:00.000Z"


def test_cabecera_lleva_el_sha_del_adr(tmp_path: Path) -> None:
    runner, _, _ = armar(tmp_path, "2026-10-17T17:30:00")
    runner.step(Due(t("2026-10-17T17:30:00"), "periodic"))
    cabecera = registros(tmp_path, "2026-10-17")[0]
    assert cabecera["adr_sha256"] == hashlib.sha256(ADR_PATH.read_bytes()).hexdigest()
    assert "telegram_token" not in json.dumps(cabecera)


def test_sin_adr_no_arranca(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="no se puede leer"):
        build_runner(
            ShadowConfig(data_dir=tmp_path),
            SimulatedClock(t("2026-10-17T17:30:00")),
            sleep=lambda s: None,
            fetch=Servidor(),
            adr_path=tmp_path / "no_existe.md",
        )
