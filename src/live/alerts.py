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
Get = Callable[[str], None]


class Notifier(Protocol):
    def send(self, texto: str) -> bool: ...


def _post_http(url: str, cuerpo: bytes) -> None:  # pragma: no cover - red real
    pedido = urllib.request.Request(
        url, data=cuerpo, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(pedido, timeout=TIMEOUT_S):
        pass


def _get_http(url: str) -> None:  # pragma: no cover - red real
    with urllib.request.urlopen(url, timeout=TIMEOUT_S):
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


class DeadmanPing:
    """Ping periodico a un servicio externo (healthchecks.io) cuando todo esta sano.

    Cubre lo que Telegram no puede: si el VPS entero cae, el watchdog cae con el y
    los pings dejan de llegar; el servicio externo avisa de la ausencia. La URL
    lleva un UUID que funciona como secreto: no se loguea.
    """

    def __init__(self, url: str, get: Get = _get_http) -> None:
        self._url = url
        self._get = get

    def ping(self) -> bool:
        try:
            self._get(self._url)
        except Exception as exc:  # cualquier fallo de red es "no salio"
            log.error("ping externo no enviado", extra={"error": type(exc).__name__})
            return False
        return True


def build_deadman(config: ShadowConfig, get: Get = _get_http) -> DeadmanPing | None:
    return DeadmanPing(config.deadman_url, get) if config.deadman_enabled else None
