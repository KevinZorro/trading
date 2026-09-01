"""Baselines heuristicos.

Todo experimento del proyecto los incluye. Si un agente de RL no supera a
buy-and-hold neto de costos, el resultado es "no supera a buy-and-hold".

Implementan el mismo protocolo :class:`~sim.engine.Strategy` que usaran los
agentes de RL, de modo que el motor no distingue entre unos y otros.
"""

from __future__ import annotations

import numpy as np

from sim.orders import MarketOrder
from sim.view import AccountSnapshot, MarketView


class BuyAndHold:
    """Compra en la primera barra posible y no vuelve a operar.

    Detalle que no es un descuido: **no puede quedar invertido al 100%**. La
    orden se dimensiona con ``close[t]`` pero se ejecuta al ``open[t+1]``, que
    puede ser mas alto; sin margen de seguridad la orden se rechazaria por cash
    insuficiente. Estar exactamente all-in exigiria conocer el precio de
    ejecucion antes de enviar la orden, es decir, lookahead. El sobrante en cash
    es el precio honesto de no tenerlo.
    """

    name = "buy_and_hold"

    def __init__(self, *, safety: float = 0.98, retry: bool = True) -> None:
        if not 0.0 < safety <= 1.0:
            raise ValueError("safety debe estar en (0, 1]")
        self.safety = safety
        self.retry = retry
        self._done = False

    def reset(self, seed: int | None = None) -> None:
        self._done = False

    def on_bar(
        self, view: MarketView, account: AccountSnapshot
    ) -> MarketOrder | None:
        if self._done:
            return None
        if account.position > 0:
            # La compra se ejecuto: no se vuelve a operar.
            self._done = True
            return None
        if not self.retry and view.t > 0:
            self._done = True
            return None
        qty = account.max_affordable_qty(view.close(), safety=self.safety)
        if qty <= 0:
            return None
        # No se pre-redondea a cero: si la cantidad no alcanza el minimo del
        # venue, la orden se envia igual y el rechazo queda en el log. Un
        # baseline que se autocensura hace indistinguible "no quiso operar" de
        # "no pudo operar", que es justo lo que mide el barrido de capital.
        return MarketOrder(qty=qty, tag="buy_and_hold")


class RandomAgent:
    """Agente aleatorio con semilla propia.

    Es el baseline que separa "el agente aprendio algo" de "el activo subio".
    Opera long-only para ser comparable con el resto del estudio.
    """

    name = "random"

    def __init__(
        self,
        *,
        seed: int = 0,
        trade_prob: float = 0.1,
        size_fraction: float = 0.25,
        safety: float = 0.98,
    ) -> None:
        if not 0.0 <= trade_prob <= 1.0:
            raise ValueError("trade_prob debe estar en [0, 1]")
        if not 0.0 < size_fraction <= 1.0:
            raise ValueError("size_fraction debe estar en (0, 1]")
        self.seed = seed
        self.trade_prob = trade_prob
        self.size_fraction = size_fraction
        self.safety = safety
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def on_bar(
        self, view: MarketView, account: AccountSnapshot
    ) -> MarketOrder | None:
        if self._rng.random() >= self.trade_prob:
            return None
        instrument = view.instrument
        if self._rng.random() < 0.5:
            qty = (
                account.max_affordable_qty(view.close(), safety=self.safety)
                * self.size_fraction
            )
            return MarketOrder(qty=qty, tag="random_buy") if qty > 0 else None
        qty = instrument.round_qty(account.position * self.size_fraction)
        return MarketOrder(qty=-qty, tag="random_sell") if qty > 0 else None


class MovingAverageCross:
    """Cruce de medias moviles simples, long-only.

    Entra cuando la media rapida cruza por encima de la lenta y sale en el cruce
    contrario. Solo consume ``view.history``, asi que no puede ver mas alla
    de ``t`` ni aunque quisiera.
    """

    name = "ma_cross"

    def __init__(
        self, *, fast: int = 20, slow: int = 50, safety: float = 0.98
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("las ventanas deben ser positivas")
        if fast >= slow:
            raise ValueError("fast debe ser menor que slow")
        self.fast = fast
        self.slow = slow
        self.safety = safety

    def reset(self, seed: int | None = None) -> None:
        return None

    def _above(self, view: MarketView, lookback: int) -> bool:
        closes = view.history("close", self.slow + lookback)
        window = closes[: len(closes) - lookback] if lookback else closes
        return float(window[-self.fast :].mean()) > float(window[-self.slow :].mean())

    def on_bar(
        self, view: MarketView, account: AccountSnapshot
    ) -> MarketOrder | None:
        # Hace falta una barra extra para conocer el estado anterior y detectar
        # el cruce en vez del nivel.
        if len(view) < self.slow + 1:
            return None
        now = self._above(view, 0)
        before = self._above(view, 1)
        if now and not before and account.position == 0:
            qty = account.max_affordable_qty(view.close(), safety=self.safety)
            return MarketOrder(qty=qty, tag="ma_entry") if qty > 0 else None
        if before and not now and account.position > 0:
            qty = view.instrument.round_qty(account.position)
            return MarketOrder(qty=-qty, tag="ma_exit") if qty > 0 else None
        return None
