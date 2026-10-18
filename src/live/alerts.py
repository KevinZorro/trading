"""Canal de alertas de la sombra (ADR 0007, PR 6).

``Notifier.send`` devuelve si el aviso salio: el watchdog reintenta en el
siguiente ciclo si no. El POST HTTP se inyecta, asi que los tests no tocan red.

El token vive en la URL de Telegram, asi que **nunca** se loguea la URL ni el
mensaje de la excepcion: solo su tipo.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable
from typing import Protocol

from live.config import ShadowConfig

log = logging.getLogger(__name__)
TELEGRAM_HOST = "https://api.telegram.org"
TIMEOUT_S = 10.0

Post = Callable[[str, bytes], None]


class Notifier(Protocol):
    def send(self, texto: str) -> bool: ...


def _post_http(url: str, cuerpo: bytes) -> None:  # pragma: no cover - red real
    pedido = urllib.request.Request(
        url, data=cuerpo, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(pedido, timeout=TIMEOUT_S):
        pass


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, post: Post = _post_http) -> None:
        self._url = f"{TELEGRAM_HOST}/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._post = post

    def send(self, texto: str) -> bool:
        cuerpo = json.dumps({"chat_id": self._chat_id, "text": texto}).encode()
        try:
            self._post(self._url, cuerpo)
        except Exception as exc:  # cualquier fallo de red es "no salio"
            log.error("alerta no enviada", extra={"error": type(exc).__name__})
            return False
        return True


class LogNotifier:
    """Respaldo sin credenciales: la alerta queda en el log y se da por enviada."""

    def send(self, texto: str) -> bool:
        log.warning("alerta (sin canal configurado)", extra={"text": texto})
        return True


def build_notifier(config: ShadowConfig, post: Post = _post_http) -> Notifier:
    if config.alerts_enabled:
        return TelegramNotifier(config.telegram_token, config.telegram_chat_id, post)
    return LogNotifier()
