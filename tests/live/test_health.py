"""Salud de la sombra: oraculos calculados a mano, sin red ni reloj real."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from live.config import ShadowConfig
from live.health import check, daily_missing, evaluate, read_heartbeat

CFG = ShadowConfig(data_dir=Path("."))  # umbral 45 min, diario 00:00:30


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


def hb(at: str, invalidos: int = 0) -> dict[str, object]:
    return {"at": at, "last_scheduled": None, "consecutive_invalid": invalidos}


def test_heartbeat_en_el_limite_es_sano_y_un_minuto_despues_no() -> None:
    # 12:00 -> 12:45 son exactamente 45 min: no supera el maximo.
    assert evaluate(hb("2026-10-18T12:00:00.000Z"), t("2026-10-18T12:45:00"), CFG).ok
    malo = evaluate(hb("2026-10-18T12:00:00.000Z"), t("2026-10-18T12:46:00"), CFG)
    assert not malo.ok
    assert malo.reasons == ("heartbeat atrasado mas de 45 min",)


def test_tres_invalidos_seguidos_alertan_dos_no() -> None:
    ahora = t("2026-10-18T12:01:00")
    assert evaluate(hb("2026-10-18T12:00:00.000Z", 2), ahora, CFG).ok
    assert not evaluate(hb("2026-10-18T12:00:00.000Z", 3), ahora, CFG).ok


def test_heartbeat_ausente_o_sin_campo_at() -> None:
    ahora = t("2026-10-18T12:00:00")
    assert "falta" in evaluate(None, ahora, CFG).reasons[0]
    assert "sin campo" in evaluate({"x": 1}, ahora, CFG).reasons[0]
    assert "sin campo" in evaluate({"at": "no-es-fecha"}, ahora, CFG).reasons[0]


def test_read_heartbeat_tolera_ausente_y_corrupto(tmp_path: Path) -> None:
    assert read_heartbeat(tmp_path) is None
    (tmp_path / "heartbeat.json").write_text("{roto")
    assert read_heartbeat(tmp_path) is None
    (tmp_path / "heartbeat.json").write_text("[1]")
    assert read_heartbeat(tmp_path) is None


def diario(scheduled: str, valid: bool = True) -> dict[str, object]:
    return {"type": "snapshot", "kind": "daily", "valid": valid, "scheduled": scheduled}


def test_diario_no_se_exige_dentro_de_la_gracia() -> None:
    # 00:10:29 < 00:00:30 + 10 min.
    assert daily_missing([], t("2026-10-18T00:10:29"), CFG) is None
    assert daily_missing([], t("2026-10-18T00:10:30"), CFG) is not None


def test_diario_valido_del_dia_lo_cubre_y_el_invalido_o_ajeno_no() -> None:
    ahora = t("2026-10-18T09:00:00")
    ok = diario("2026-10-18T00:00:30.000Z")
    assert daily_missing([ok], ahora, CFG) is None
    assert daily_missing([diario("2026-10-18T00:00:30.000Z", False)], ahora, CFG)
    assert daily_missing([diario("2026-10-17T00:00:30.000Z")], ahora, CFG)
    periodico = {**ok, "kind": "periodic"}
    assert daily_missing([periodico], ahora, CFG)


def test_check_lee_el_archivo_del_dia_y_salta_la_linea_rota(tmp_path: Path) -> None:
    cfg = ShadowConfig(data_dir=tmp_path)
    ahora = t("2026-10-18T09:00:00")
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(hb("2026-10-18T08:55:00.000Z")) + "\n"
    )
    assert not check(tmp_path, ahora, cfg).ok  # sin archivo de snapshots
    lineas = [
        json.dumps({"type": "header"}),
        json.dumps(diario("2026-10-18T00:00:30.000Z")),
        '{"type": "snap',  # ultima linea rota por una caida
    ]
    (tmp_path / "snapshots-2026-10-18.jsonl").write_text("\n".join(lineas))
    assert check(tmp_path, ahora, cfg).ok
