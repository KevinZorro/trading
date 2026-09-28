"""Configuracion del modo sombra, por variables de entorno. Sin secretos en el repo.

El entorno se **inyecta** (``from_env(environ)``) en vez de leerse de
``os.environ`` adentro: los tests no tocan el entorno del proceso y la
configuracion efectiva es una funcion pura de lo que se le pasa.

Los secretos (token del bot de Telegram) no aparecen en ``repr`` ni en
``describe``: la configuracion se serializa junto a cada corrida y un secreto
serializado es un secreto filtrado.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

PREFIJO = "SHADOW_"
# Host de datos publicos de mercado de mainnet: solo lectura, sin API key.
MARKET_DATA_URL = "https://data-api.binance.vision"


class ConfigError(ValueError):
    """La configuracion del entorno es invalida o esta incompleta."""


@dataclass(frozen=True)
class ShadowConfig:
    """Parametros del modo sombra. Los valores por defecto son los del ADR 0007."""

    data_dir: Path
    symbol: str = "BTCUSDT"
    order_sizes_usdt: tuple[float, ...] = (10.0, 100.0, 1000.0)
    snapshot_minutes: int = 15
    daily_snapshot_utc: str = "00:00:30"  # comparable al open de t+1 del backtest
    depth_limit: int = 1000
    heartbeat_max_age_minutes: int = 45  # tres snapshots sin escribir: alerta
    market_data_url: str = MARKET_DATA_URL
    telegram_chat_id: str = ""
    telegram_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.order_sizes_usdt or min(self.order_sizes_usdt) <= 0:
            raise ConfigError("order_sizes_usdt debe tener tamanios positivos")
        if not 1 <= self.snapshot_minutes <= 60 or 60 % self.snapshot_minutes:
            raise ConfigError(
                f"snapshot_minutes={self.snapshot_minutes} debe dividir a 60: los "
                "snapshots se alinean a la hora"
            )
        if self.heartbeat_max_age_minutes <= self.snapshot_minutes:
            raise ConfigError(
                "heartbeat_max_age_minutes debe superar a snapshot_minutes: si no, "
                "la alerta salta entre dos snapshots sanos"
            )
        if not self.market_data_url.startswith("https://"):
            raise ConfigError("market_data_url debe ser https")
        if bool(self.telegram_token) != bool(self.telegram_chat_id):
            raise ConfigError(
                "SHADOW_TELEGRAM_TOKEN y SHADOW_TELEGRAM_CHAT_ID van juntos: con uno "
                "solo la alerta no puede salir"
            )

    @property
    def alerts_enabled(self) -> bool:
        return bool(self.telegram_token)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> ShadowConfig:
        """Lee ``SHADOW_*``. Solo ``SHADOW_DATA_DIR`` es obligatoria."""

        def leer(nombre: str) -> str | None:
            valor = environ.get(PREFIJO + nombre)
            return valor.strip() if valor is not None and valor.strip() else None

        directorio = leer("DATA_DIR")
        if directorio is None:
            raise ConfigError("falta SHADOW_DATA_DIR: la sombra persiste a disco")
        opciones: dict[str, object] = {"data_dir": Path(directorio)}
        try:
            if (valor := leer("SYMBOL")) is not None:
                opciones["symbol"] = valor.upper()
            if (valor := leer("MARKET_DATA_URL")) is not None:
                opciones["market_data_url"] = valor
            if (valor := leer("ORDER_SIZES_USDT")) is not None:
                opciones["order_sizes_usdt"] = tuple(
                    float(x) for x in valor.split(",") if x.strip()
                )
            for nombre, clave in (
                ("SNAPSHOT_MINUTES", "snapshot_minutes"),
                ("DEPTH_LIMIT", "depth_limit"),
                ("HEARTBEAT_MAX_AGE_MINUTES", "heartbeat_max_age_minutes"),
            ):
                if (valor := leer(nombre)) is not None:
                    opciones[clave] = int(valor)
        except ValueError as exc:
            raise ConfigError(f"valor numerico invalido en el entorno: {exc}") from exc
        opciones["telegram_token"] = leer("TELEGRAM_TOKEN") or ""
        opciones["telegram_chat_id"] = leer("TELEGRAM_CHAT_ID") or ""
        return cls(**opciones)  # type: ignore[arg-type]

    def describe(self) -> dict[str, object]:
        """Serializable junto a cada corrida. **Sin el token.**"""
        return {
            "data_dir": str(self.data_dir),
            "symbol": self.symbol,
            "order_sizes_usdt": list(self.order_sizes_usdt),
            "snapshot_minutes": self.snapshot_minutes,
            "daily_snapshot_utc": self.daily_snapshot_utc,
            "depth_limit": self.depth_limit,
            "heartbeat_max_age_minutes": self.heartbeat_max_age_minutes,
            "market_data_url": self.market_data_url,
            "alerts_enabled": self.alerts_enabled,
        }
