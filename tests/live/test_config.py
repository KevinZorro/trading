"""Configuracion por entorno: valores del ADR 0007 por defecto, sin secretos."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from live.config import MARKET_DATA_URL, ConfigError, ShadowConfig
from live.logs import configure_json_logging

TOKEN = "123456:SECRETO-DE-PRUEBA"


def test_por_defecto_son_los_valores_del_adr() -> None:
    config = ShadowConfig.from_env({"SHADOW_DATA_DIR": "/datos"})
    assert config.data_dir == Path("/datos")
    assert config.order_sizes_usdt == (10.0, 100.0, 1000.0)
    assert config.snapshot_minutes == 15
    assert config.daily_snapshot_utc == "00:00:30"
    assert config.market_data_url == MARKET_DATA_URL
    assert not config.alerts_enabled


def test_sin_directorio_de_datos_no_arranca() -> None:
    with pytest.raises(ConfigError, match="falta SHADOW_DATA_DIR"):
        ShadowConfig.from_env({})


def test_lee_y_valida_el_entorno() -> None:
    config = ShadowConfig.from_env(
        {
            "SHADOW_DATA_DIR": "/d",
            "SHADOW_SYMBOL": "ethusdt",
            "SHADOW_ORDER_SIZES_USDT": "10, 50",
            "SHADOW_SNAPSHOT_MINUTES": "30",
            "SHADOW_HEARTBEAT_MAX_AGE_MINUTES": "90",
        }
    )
    assert config.symbol == "ETHUSDT"
    assert config.order_sizes_usdt == (10.0, 50.0)
    assert config.snapshot_minutes == 30


@pytest.mark.parametrize(
    ("entorno", "motivo"),
    [
        ({"SHADOW_SNAPSHOT_MINUTES": "7"}, "debe dividir a 60"),
        ({"SHADOW_SNAPSHOT_MINUTES": "abc"}, "valor numerico invalido"),
        ({"SHADOW_ORDER_SIZES_USDT": "10,-5"}, "tamanios positivos"),
        ({"SHADOW_HEARTBEAT_MAX_AGE_MINUTES": "15"}, "debe superar"),
        ({"SHADOW_TELEGRAM_TOKEN": TOKEN}, "van juntos"),
        ({"SHADOW_MARKET_DATA_URL": "http://data-api.binance.vision"}, "https"),
    ],
)
def test_rechaza_configuraciones_invalidas(
    entorno: dict[str, str], motivo: str
) -> None:
    with pytest.raises(ConfigError, match=motivo):
        ShadowConfig.from_env({"SHADOW_DATA_DIR": "/d", **entorno})


def test_el_token_no_se_filtra_ni_en_repr_ni_en_describe() -> None:
    config = ShadowConfig.from_env(
        {
            "SHADOW_DATA_DIR": "/d",
            "SHADOW_TELEGRAM_TOKEN": TOKEN,
            "SHADOW_TELEGRAM_CHAT_ID": "42",
        }
    )
    assert config.alerts_enabled
    assert TOKEN not in repr(config)
    assert TOKEN not in json.dumps(config.describe())


def test_los_logs_son_json_por_linea_y_redactan_secretos() -> None:
    salida = io.StringIO()
    logger = configure_json_logging(salida)
    logger.info("snapshot", extra={"size_usdt": 100.0, "telegram_token": TOKEN})
    try:
        raise RuntimeError("fallo de prueba")
    except RuntimeError:
        logger.exception("error")
    lineas = [json.loads(x) for x in salida.getvalue().splitlines()]
    assert lineas[0]["msg"] == "snapshot"
    assert lineas[0]["size_usdt"] == 100.0
    assert lineas[0]["telegram_token"] == "[REDACTADO]"
    assert TOKEN not in salida.getvalue()
    assert lineas[0]["level"] == "INFO"
    assert "fallo de prueba" in lineas[1]["exc"]
    logging.getLogger("live").handlers.clear()
