"""Logs estructurados: una linea JSON por evento, a stdout.

Un proceso que corre 24/7 en un VPS se diagnostica leyendo logs, y los logs se
leen con herramientas. JSON por linea se filtra con ``jq`` y lo entiende
cualquier colector. La hora sale de ``record.created``, que pone ``logging``:
este modulo no consulta el reloj del sistema.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import IO, Any

# Atributos estandar de LogRecord: todo lo demas llego por ``extra=`` y va al JSON.
_ESTANDAR = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
# Nombres de campo que nunca se escriben en claro, vengan de donde vengan.
_SECRETOS = ("token", "secret", "password", "api_key", "apikey", "signature")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        salida: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for clave, valor in vars(record).items():
            if clave in _ESTANDAR:
                continue
            salida[clave] = (
                "[REDACTADO]" if any(s in clave.lower() for s in _SECRETOS) else valor
            )
        if record.exc_info:
            salida["exc"] = self.formatException(record.exc_info)
        return json.dumps(salida, ensure_ascii=False, default=str)


def configure_json_logging(
    stream: IO[str], level: int = logging.INFO
) -> logging.Logger:
    """Configura el logger ``live`` para escribir JSON por linea en ``stream``."""
    logger = logging.getLogger("live")
    logger.handlers.clear()
    manejador = logging.StreamHandler(stream)
    manejador.setFormatter(JsonFormatter())
    logger.addHandler(manejador)
    logger.setLevel(level)
    logger.propagate = False
    return logger
