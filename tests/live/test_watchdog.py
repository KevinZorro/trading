"""Watchdog y StallGuard con reloj simulado: la dedup y el plazo, a mano."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from live.config import ShadowConfig
from live.stall import StallGuard
from live.watchdog import CICLO_S, Watchdog
from sim.clock import SimulatedClock


def t(texto: str) -> np.datetime64:
    return np.datetime64(texto, "ns")


def minutos(n: int) -> np.timedelta64:
    return np.timedelta64(n, "m")


class Captura:
    def __init__(self, ok: bool = True) -> None:
        self.textos: list[str] = []
        self.ok = ok

    def send(self, texto: str) -> bool:
        if self.ok:
            self.textos.append(texto)
        return self.ok


def latido(dir_: Path, at: str) -> None:
    (dir_ / "heartbeat.json").write_text(
        json.dumps({"at": at, "consecutive_invalid": 0}) + "\n"
    )


def armar(
    tmp_path: Path, inicio: str, ok: bool = True
) -> tuple[Watchdog, SimulatedClock, Captura]:
    reloj = SimulatedClock(t(inicio))
    cap = Captura(ok)
    cfg = ShadowConfig(data_dir=tmp_path)
    return Watchdog(cfg, reloj, cap, sleep=lambda s: None), reloj, cap


def test_avisa_una_vez_recuerda_a_las_6h_y_avisa_la_recuperacion(
    tmp_path: Path,
) -> None:
    # 00:00:30 + 10 min ya pasaron el dia anterior, asi que solo cuenta el heartbeat.
    w, reloj, cap = armar(tmp_path, "2026-10-18T00:05:00")
    (tmp_path / "snapshots-2026-10-18.jsonl").write_text(
        json.dumps(
            {
                "type": "snapshot",
                "kind": "daily",
                "valid": True,
                "scheduled": "2026-10-18T00:00:30.000Z",
            }
        )
        + "\n"
    )
    latido(tmp_path, "2026-10-18T00:00:00.000Z")
    w.step()
    assert cap.textos == []  # 5 min de edad: sano
    reloj.advance_to(t("2026-10-18T00:50:00"))  # 50 min > 45
    w.step()
    assert len(cap.textos) == 1
    assert "ALERTA" in cap.textos[0]
    reloj.advance_to(t("2026-10-18T00:55:00"))
    w.step()
    assert len(cap.textos) == 1  # mismo motivo: sin spam
    reloj.advance_to(t("2026-10-18T06:50:00"))  # justo 6 h despues del primer aviso
    w.step()
    assert len(cap.textos) == 2  # recordatorio
    latido(tmp_path, "2026-10-18T06:55:00.000Z")
    reloj.advance_to(t("2026-10-18T06:56:00"))
    w.step()
    assert "recuperada" in cap.textos[-1]


def test_si_el_aviso_no_sale_se_reintenta_en_el_siguiente_ciclo(tmp_path: Path) -> None:
    w, reloj, cap = armar(tmp_path, "2026-10-18T05:00:00", ok=False)  # sin heartbeat
    w.step()
    assert cap.textos == []
    cap.ok = True
    reloj.advance_to(reloj.now() + np.timedelta64(int(CICLO_S), "s"))
    w.step()
    assert len(cap.textos) == 1


def test_stall_guard_dispara_solo_pasado_el_plazo_y_touch_lo_reinicia() -> None:
    reloj = SimulatedClock(t("2026-10-18T12:00:00"))
    muertes: list[int] = []
    g = StallGuard(reloj, minutos(35), lambda: muertes.append(1))
    reloj.advance_to(t("2026-10-18T12:35:00"))
    assert not g.check()  # justo en el plazo, no lo supera
    g.touch()
    reloj.advance_to(t("2026-10-18T13:10:00"))
    assert not g.check()  # 35 min desde el touch
    reloj.advance_to(t("2026-10-18T13:11:00"))
    assert g.check()
    assert muertes == [1]
