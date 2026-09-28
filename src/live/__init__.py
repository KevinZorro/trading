"""Infraestructura en vivo de la Etapa 3.5. Ver ``docs/adr/0007``.

Dos validaciones con propositos distintos que no se mezclan: la sombra sobre
mainnet mide COSTOS en solo lectura; testnet valida MECANICA y nunca costos.

Mainnet es de solo lectura **por construccion**: el unico camino que envia
ordenes pasa por :func:`live.guard.assert_testnet_order_url`.
"""

from live.config import ConfigError, ShadowConfig
from live.guard import MainnetOrderError, assert_testnet_order_url
from live.logs import JsonFormatter, configure_json_logging

__all__ = [
    "ConfigError",
    "JsonFormatter",
    "MainnetOrderError",
    "ShadowConfig",
    "assert_testnet_order_url",
    "configure_json_logging",
]
