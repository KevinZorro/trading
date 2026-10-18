"""Alertas: el POST se inyecta, el token no se filtra."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from live.alerts import (
    DeadmanPing,
    LogNotifier,
    TelegramNotifier,
    build_deadman,
    build_notifier,
)
from live.config import ShadowConfig


def test_telegram_envia_chat_id_y_texto_a_la_url_del_bot() -> None:
    enviados: list[tuple[str, bytes]] = []
    n = TelegramNotifier(
        "T0K", "42", lambda url, cuerpo: enviados.append((url, cuerpo))
    )
    assert n.send("hola")
    ((url, cuerpo),) = enviados
    assert url == "https://api.telegram.org/botT0K/sendMessage"
    assert json.loads(cuerpo) == {"chat_id": "42", "text": "hola"}


def test_fallo_de_red_devuelve_false_y_no_filtra_el_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def roto(url: str, cuerpo: bytes) -> None:
        raise OSError(f"fallo contra {url}")

    with caplog.at_level(logging.ERROR):
        assert not TelegramNotifier("T0K", "42", roto).send("x")
    assert "T0K" not in caplog.text
    assert caplog.records[0].__dict__["error"] == "OSError"


def test_build_notifier_elige_canal_segun_credenciales(tmp_path: Path) -> None:
    assert isinstance(build_notifier(ShadowConfig(data_dir=tmp_path)), LogNotifier)
    con = ShadowConfig(data_dir=tmp_path, telegram_token="t", telegram_chat_id="1")
    assert isinstance(build_notifier(con), TelegramNotifier)


def test_log_notifier_da_el_aviso_por_enviado(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert LogNotifier().send("aviso")
    assert caplog.records[0].__dict__["text"] == "aviso"


def test_deadman_hace_get_a_la_url_y_no_la_filtra_si_falla(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pedidos: list[str] = []
    assert DeadmanPing("https://hc-ping.com/UUID", pedidos.append).ping()
    assert pedidos == ["https://hc-ping.com/UUID"]

    def roto(url: str) -> None:
        raise OSError(f"fallo contra {url}")

    with caplog.at_level(logging.ERROR):
        assert not DeadmanPing("https://hc-ping.com/UUID", roto).ping()
    assert "UUID" not in caplog.text
    assert caplog.records[0].__dict__["error"] == "OSError"


def test_build_deadman_solo_si_hay_url(tmp_path: Path) -> None:
    assert build_deadman(ShadowConfig(data_dir=tmp_path)) is None
    con = ShadowConfig(data_dir=tmp_path, deadman_url="https://hc-ping.com/u")
    assert isinstance(build_deadman(con), DeadmanPing)
