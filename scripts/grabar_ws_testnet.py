"""Graba los mensajes reales del WebSocket de testnet de Binance (PR 8a).

Uso (desde la raiz del repo, con red). Las claves son de **testnet** y se pasan por
el entorno, nunca por argumento::

    $env:TESTNET_API_KEY = "..."; $env:TESTNET_API_SECRET = "..."
    $env:PYTHONPATH = "src"; uv run --group ws python scripts/grabar_ws_testnet.py

Sirve para fijar, con datos reales y antes de escribir el observador de fills
(PR 8b), lo que hoy es solo lo que dice una documentacion de terceros:

* si ``userDataStream.subscribe.signature`` es la forma vigente de suscribirse en
  spot (y no el ``listenKey``), y como se ven la respuesta y los eventos;
* como llegan ``executionReport``, ``outboundAccountPosition`` y ``balanceUpdate``;
* **que pasa con un fill ocurrido mientras el socket estaba caido**: si al
  reconectar el venue lo repite o no. De eso depende el contrato de recuperacion;
* como se ven un error de firma y la baja de la suscripcion.

Tres fases, en testnet (dinero ficticio, 0.001 BTC por orden):

A. suscribe, compra y vende. Eventos de un fill normal.
B. cierra el socket, compra **sin** socket, reconecta, se suscribe y escucha; luego
   vende. Muestra si el fill perdido se repite.
C. firma invalida en un socket aparte y baja de la suscripcion.

Lo que **nunca** se escribe: la API key, la firma ni los pedidos enviados (solo se
guardan las respuestas y los eventos). El campo ``uid`` se borra y antes de escribir
se verifica que ni la clave ni el secreto aparezcan y que nada tenga forma de clave
o firma (``live.testnet_http.find_secret_shapes``). Si la suscripcion no es aceptada,
el script se detiene **sin escribir** y muestra el codigo de error de Binance, que es
justo el hallazgo que importa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from live.guard import assert_testnet_ws_url
from live.testnet_http import (
    BinanceApiError,
    BinanceHttpError,
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    find_secret_shapes,
    sign,
)
from sim.clock import Clock, SystemClock

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "tests" / "live" / "respuestas" / "testnet_ws"
WS_URL = "wss://ws-api.testnet.binance.vision/ws-api/v3"
CAMPOS_A_BORRAR = frozenset({"uid"})
SEGUNDOS_FILL = 10  # silencio tolerado esperando el fill propio
SEGUNDOS_RECONEXION = 5  # silencio con el que se da por cerrada la escucha
MAX_FRAMES = 200


class Socket(Protocol):
    def send(self, texto: str) -> None: ...

    def recv(self, timeout: float) -> str | None:
        """Un mensaje, o ``None`` si no llego ninguno en ``timeout`` segundos."""
        ...

    def close(self) -> None: ...


def conectar(url: str) -> Socket:  # pragma: no cover - red
    from websockets.sync.client import connect  # grupo opcional ``ws``

    class _Real:
        def __init__(self) -> None:
            self._ws = connect(url, open_timeout=10)

        def send(self, texto: str) -> None:
            self._ws.send(texto)

        def recv(self, timeout: float) -> str | None:
            try:
                dato = self._ws.recv(timeout=timeout)
            except TimeoutError:
                return None
            return dato if isinstance(dato, str) else dato.decode()

        def close(self) -> None:
            self._ws.close()

    return _Real()


def pedido_firmado(
    metodo: str, api_key: str, secret: str, timestamp_ms: int, id_: str
) -> str:
    """Pedido del WebSocket API firmado: HMAC sobre los params ordenados por clave."""
    params: dict[str, str | int] = {"apiKey": api_key, "timestamp": timestamp_ms}
    cadena = "&".join(f"{k}={params[k]}" for k in sorted(params))
    params["signature"] = sign(secret, cadena)
    return json.dumps({"id": id_, "method": metodo, "params": params})


class Fase:
    """Frames de una fase en orden de llegada, con la hora local de recepcion."""

    def __init__(self, nombre: str, reloj: Clock) -> None:
        self.nombre = nombre
        self._reloj = reloj
        self.lineas: list[dict[str, Any]] = []

    def anotar(self, texto: str) -> dict[str, Any]:
        try:
            frame = json.loads(texto)
        except ValueError:
            frame = {"raw": texto}
        recibido = np.datetime_as_string(self._reloj.now(), unit="ms") + "Z"
        self.lineas.append({"received_at_utc": recibido, "frame": _limpiar(frame)})
        return frame if isinstance(frame, dict) else {}


def _limpiar(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items() if k not in CAMPOS_A_BORRAR}
    if isinstance(valor, list):
        return [_limpiar(v) for v in valor]
    return valor


def es_fill_de(frame: dict[str, Any], client_order_id: str) -> bool:
    evento = frame.get("event", frame)
    return bool(
        isinstance(evento, dict)
        and evento.get("e") == "executionReport"
        and evento.get("c") == client_order_id
        and evento.get("X") == "FILLED"
    )


def escuchar(
    sock: Socket,
    fase: Fase,
    *,
    silencio_s: int,
    hasta: Callable[[dict[str, Any]], bool] | None = None,
) -> bool:
    """Anota frames hasta ``hasta`` o hasta ``silencio_s`` segundos sin mensajes."""
    vacios = 0
    for _ in range(MAX_FRAMES):
        texto = sock.recv(1.0)
        if texto is None:
            vacios += 1
            if vacios >= silencio_s:
                return False
            continue
        vacios = 0
        frame = fase.anotar(texto)
        if hasta is not None and hasta(frame):
            return True
    return False


def suscribir(
    sock: Socket,
    fase: Fase,
    creds: BinanceTestnetCredentials,
    ahora_ms: Callable[[], int],
    id_: str,
) -> None:
    sock.send(
        pedido_firmado(
            "userDataStream.subscribe.signature",
            creds.api_key,
            creds.api_secret,
            ahora_ms(),
            id_,
        )
    )
    for _ in range(MAX_FRAMES):
        texto = sock.recv(5.0)
        if texto is None:
            break
        respuesta = fase.anotar(texto)
        if respuesta.get("id") == id_:
            if respuesta.get("status") != 200:
                raise SystemExit(
                    "Binance no acepto la suscripcion "
                    f"(status {respuesta.get('status')}, "
                    f"error {respuesta.get('error')}). "
                    "No se grabo nada: este es el dato que decide el diseno del PR 8b, "
                    "pegalo en el hilo."
                )
            return
    raise SystemExit("la suscripcion no tuvo respuesta en 5 s. No se grabo nada.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--qty", default="0.001", help="cantidad de cada orden")
    args = parser.parse_args(argv)

    creds = BinanceTestnetCredentials.from_env(os.environ)
    reloj = SystemClock()
    cliente = BinanceTestnetClient(creds, reloj)
    offset = cliente.sync_time()
    sonda = cliente.send("GET", "/api/v3/account")
    if sonda.status != 200:
        raise SystemExit(
            f"las claves no sirven (HTTP {sonda.status}: {sonda.body[:120]!r}). "
            "Revisa TESTNET_API_KEY y TESTNET_API_SECRET: deben ser de "
            "https://testnet.binance.vision, sin comillas ni espacios. "
            "No se grabo nada."
        )
    url = assert_testnet_ws_url(WS_URL)
    corrida = str(int(reloj.now().astype("datetime64[s]").astype(np.int64)))

    def ahora_ms() -> int:
        return int(reloj.now().astype("datetime64[ms]").astype(np.int64)) + offset

    def orden(cid: str, lado: str) -> None:
        cliente.call(
            "POST",
            "/api/v3/order",
            {
                "symbol": args.symbol,
                "side": lado,
                "type": "MARKET",
                "quantity": args.qty,
                "newClientOrderId": cid,
                "newOrderRespType": "FULL",
            },
        )

    def operar(sock: Socket, fase: Fase, cid: str, lado: str) -> None:
        orden(cid, lado)
        escuchar(
            sock, fase, silencio_s=SEGUNDOS_FILL, hasta=lambda f: es_fill_de(f, cid)
        )

    a = Fase("fase_a_fill_normal", reloj)
    b = Fase("fase_b_fill_con_socket_caido", reloj)
    c = Fase("fase_c_firma_invalida_y_baja", reloj)

    s1 = conectar(url)
    suscribir(s1, a, creds, ahora_ms, f"sub-a-{corrida}")
    operar(s1, a, f"rec8-a-buy-{corrida}", "BUY")
    operar(s1, a, f"rec8-a-sell-{corrida}", "SELL")

    s1.close()  # cierre sin baja de suscripcion: una caida
    orden(f"rec8-b-gap-{corrida}", "BUY")  # el fill ocurre sin nadie escuchando
    s2 = conectar(url)
    suscribir(s2, b, creds, ahora_ms, f"sub-b-{corrida}")
    escuchar(s2, b, silencio_s=SEGUNDOS_RECONEXION)
    operar(s2, b, f"rec8-b-sell-{corrida}", "SELL")

    s3 = conectar(url)
    s3.send(
        pedido_firmado(
            "userDataStream.subscribe.signature",
            creds.api_key,
            "0" * 64,
            ahora_ms(),
            f"sub-c-{corrida}",
        )
    )
    texto = s3.recv(5.0)
    if texto is not None:
        c.anotar(texto)
    s3.close()
    s2.send(
        json.dumps({"id": f"unsub-{corrida}", "method": "userDataStream.unsubscribe"})
    )
    escuchar(s2, c, silencio_s=2, hasta=lambda f: f.get("id") == f"unsub-{corrida}")
    s2.close()

    contenidos: dict[str, bytes] = {}
    archivos: dict[str, Any] = {}
    for fase in (a, b, c):
        cuerpo = "".join(json.dumps(x) + "\n" for x in fase.lineas).encode()
        contenidos[f"{fase.nombre}.jsonl"] = cuerpo
        archivos[f"{fase.nombre}.jsonl"] = {
            "frames": len(fase.lineas),
            "sha256": hashlib.sha256(cuerpo).hexdigest(),
        }
    manifest = {
        "script": "scripts/grabar_ws_testnet.py",
        "ws_url": WS_URL,
        "symbol": args.symbol,
        "files": archivos,
    }
    texto_manifest = json.dumps(manifest, indent=2) + "\n"
    conjunto = texto_manifest + b"".join(contenidos.values()).decode()
    if (
        creds.api_key in conjunto
        or creds.api_secret in conjunto
        or find_secret_shapes(conjunto)
    ):
        raise SystemExit("hay algo con forma de clave o firma: no se escribe nada")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, cuerpo in contenidos.items():
        (DESTINO / nombre).write_bytes(cuerpo)
    (DESTINO / "manifest.json").write_text(texto_manifest, encoding="utf-8")
    print(
        f"{sum(len(f.lineas) for f in (a, b, c))} frames en {DESTINO.relative_to(RAIZ)}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BinanceApiError, BinanceHttpError) as exc:
        sys.exit(f"fallo la grabacion: {exc}")
