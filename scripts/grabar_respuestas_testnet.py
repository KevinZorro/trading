"""Graba respuestas reales de testnet de Binance para testear el venue (PR 7b).

Uso (desde la raiz del repo, con red). Las claves son de **testnet**
(https://testnet.binance.vision) y se pasan por el entorno, nunca por argumento::

    export TESTNET_API_KEY=...  TESTNET_API_SECRET=...
    PYTHONPATH=src uv run python scripts/grabar_respuestas_testnet.py

Compra y vende una cantidad minima en testnet (dinero ficticio) y deja en
``tests/live/respuestas/testnet/`` el cuerpo de cada respuesta, incluidos los
errores, mas un ``manifest.json`` con metodo, path, parametros, estado HTTP,
hora de recepcion y SHA256.

Lo que **nunca** se escribe a disco: la API key, la firma, el query firmado ni
las cabeceras ``X-MBX-APIKEY``. El manifest guarda los parametros sin ``signature``,
``timestamp`` ni ``recvWindow``; el campo ``uid`` de la cuenta se borra; y antes de
escribir se verifica que ni la clave ni el secreto aparezcan y que nada tenga forma
de clave o firma (``live.testnet_http.find_secret_shapes``). Si algo falla, no se
escribe nada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np

from live.testnet_http import (
    TESTNET_BASE_URL,
    BinanceApiError,
    BinanceHttpError,
    BinanceTestnetClient,
    BinanceTestnetCredentials,
    HttpRequest,
    find_secret_shapes,
    urllib_transport,
)
from sim.clock import SystemClock

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "tests" / "live" / "respuestas" / "testnet"
CAMPOS_A_BORRAR = frozenset({"uid"})


def _limpiar(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items() if k not in CAMPOS_A_BORRAR}
    if isinstance(valor, list):
        return [_limpiar(v) for v in valor]
    return valor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--qty", default="0.001", help="cantidad de la compra/venta")
    args = parser.parse_args(argv)

    creds = BinanceTestnetCredentials.from_env(os.environ)
    reloj = SystemClock()
    cliente = BinanceTestnetClient(creds, reloj)
    cliente.sync_time()
    corrida = str(int(reloj.now().astype("datetime64[s]").astype(np.int64)))
    sym, qty = args.symbol, args.qty
    ids = {k: f"rec7a-{k}-{corrida}" for k in ("buy", "sell")}

    def orden(cid: str, lado: str) -> dict[str, str | int]:
        return {
            "symbol": sym,
            "side": lado,
            "type": "MARKET",
            "quantity": qty,
            "newClientOrderId": cid,
            "newOrderRespType": "FULL",
        }

    consulta = {"symbol": sym, "origClientOrderId": ids["buy"]}
    pasos: list[tuple[str, str, str, dict[str, str | int]]] = [
        ("time", "GET", "/api/v3/time", {}),
        ("exchange_info", "GET", "/api/v3/exchangeInfo", {"symbol": sym}),
        ("account_antes", "GET", "/api/v3/account", {}),
        ("order_buy", "POST", "/api/v3/order", orden(ids["buy"], "BUY")),
        ("order_query", "GET", "/api/v3/order", consulta),
        # El reenvio del mismo id ya lleno: lo que decide como se hace idempotente 7b.
        ("order_buy_duplicado", "POST", "/api/v3/order", orden(ids["buy"], "BUY")),
        ("open_orders", "GET", "/api/v3/openOrders", {"symbol": sym}),
        ("my_trades", "GET", "/api/v3/myTrades", {"symbol": sym, "limit": 10}),
        ("order_sell", "POST", "/api/v3/order", orden(ids["sell"], "SELL")),
        ("account_despues", "GET", "/api/v3/account", {}),
        (
            "err_consulta_inexistente",
            "GET",
            "/api/v3/order",
            {"symbol": sym, "origClientOrderId": f"no-existe-{corrida}"},
        ),
        (
            "err_cancelar_inexistente",
            "DELETE",
            "/api/v3/order",
            {"symbol": sym, "origClientOrderId": f"no-existe-{corrida}"},
        ),
        ("err_id_invalido", "POST", "/api/v3/order", orden("id con espacios!", "BUY")),
    ]
    grabado: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for nombre, metodo, path, params in pasos:
        respuesta = cliente.send(metodo, path, params)
        recibido = reloj.now()
        try:
            cuerpo = json.dumps(_limpiar(json.loads(respuesta.body))).encode()
        except ValueError as exc:
            raise SystemExit(f"{nombre}: respuesta que no es JSON") from exc
        grabado[nombre] = (
            cuerpo,
            {
                "method": metodo,
                "path": path,
                "params": params,
                "http_status": respuesta.status,
                "received_at_utc": np.datetime_as_string(recibido, unit="ms") + "Z",
                "sha256": hashlib.sha256(cuerpo).hexdigest(),
            },
        )

    # Firma invalida a proposito, armada a mano: el cliente no deja firmar mal.
    mala = urllib.parse.urlencode({"timestamp": 1, "signature": "0" * 64})
    url = f"{TESTNET_BASE_URL}/api/v3/account?{mala}"
    respuesta = urllib_transport(
        HttpRequest("GET", url, {"X-MBX-APIKEY": creds.api_key})
    )
    cuerpo = json.dumps(json.loads(respuesta.body)).encode()
    grabado["err_firma_invalida"] = (
        cuerpo,
        {
            "method": "GET",
            "path": "/api/v3/account",
            "params": {},
            "http_status": respuesta.status,
            "received_at_utc": np.datetime_as_string(reloj.now(), unit="ms") + "Z",
            "sha256": hashlib.sha256(cuerpo).hexdigest(),
        },
    )

    manifest = {
        "script": "scripts/grabar_respuestas_testnet.py",
        "base_url": TESTNET_BASE_URL,
        "symbol": sym,
        "files": {f"{n}.json": meta for n, (_, meta) in grabado.items()},
    }
    texto = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    conjunto = texto + b"".join(c for c, _ in grabado.values()).decode()
    prohibidos = [creds.api_key, creds.api_secret]
    if any(p in conjunto for p in prohibidos) or find_secret_shapes(conjunto):
        raise SystemExit("hay algo con forma de clave o firma: no se escribe nada")

    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, (cuerpo, _) in grabado.items():
        (DESTINO / f"{nombre}.json").write_bytes(cuerpo)
    (DESTINO / "manifest.json").write_text(texto, encoding="utf-8")
    print(f"{len(grabado)} respuestas en {DESTINO.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BinanceApiError, BinanceHttpError) as exc:
        sys.exit(f"fallo la grabacion: {exc}")
