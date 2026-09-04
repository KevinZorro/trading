"""Capa de riesgo. Vive fuera del agente y se aplica en simulacion y en vivo.

Por que fuera del agente: un agente de RL ante un estado fuera de distribucion
puede decidir cualquier cosa, y no hay entrenamiento que garantice lo contrario.
El limite de riesgo es lo que separa una mala racha de una perdida total, y no
puede depender de que la politica aprendida se comporte.

Se aplica en un unico punto -entre la decision de la estrategia y el envio al
venue- implementando :class:`~sim.gate.OrderGate`. El mismo objeto se inserta en
el mismo lugar del runner en vivo. La dependencia va ``risk`` -> ``sim`` y nunca
al reves: el simulador no sabe que esto existe.

Tres decisiones de diseno que cambian el comportamiento:

1. **Rechaza, nunca redimensiona.** Es la misma regla que ya cumple
   ``check_tradable``. Una capa que recorta una orden al limite hace
   indistinguible "el agente pidio esto" de "el limite lo recorto hasta aca", y
   ese es exactamente el borrado de informacion que el proyecto prohibe en los
   baselines que se autocensuran. El rechazo queda en el log con su motivo.

2. **El kill switch permite cerrar, no comprar.** Un kill switch que bloquea
   todo deja la posicion abierta justo cuando algo salio mal, que suele ser peor
   que el riesgo que lo disparo. Solo pasan las ordenes que reducen exposicion.

3. **El dia lo define el ``Clock`` inyectado**, no el reloj de la maquina. Sin
   eso, correr el mismo backtest en husos horarios distintos daria limites
   diarios distintos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sim.clock import Clock
from sim.gate import GateDecision
from sim.orders import Fill, MarketOrder
from sim.view import AccountSnapshot


class RiskRejectReason(StrEnum):
    """Motivos por los que la capa de riesgo no deja salir una orden."""

    MAX_POSITION_NOTIONAL = "MAX_POSITION_NOTIONAL"
    MAX_POSITION_PCT_EQUITY = "MAX_POSITION_PCT_EQUITY"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_DAILY_TURNOVER = "MAX_DAILY_TURNOVER"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass(frozen=True)
class RiskLimits:
    """Limites configurables. ``None`` desactiva el limite.

    Todos los limites son sobre la posicion **resultante** de la orden, no sobre
    la orden en si: lo que importa es a que exposicion se llega, no cuanto se
    mueve para llegar.
    """

    max_position_notional: float | None = None
    max_position_pct_equity: float | None = None
    max_daily_loss_pct: float | None = None
    max_daily_turnover_ratio: float | None = None
    # Si es True, superar max_daily_loss_pct no solo rechaza esa orden: dispara
    # el kill switch y deja de operar hasta que alguien lo reponga a mano.
    trip_on_daily_loss: bool = True

    def __post_init__(self) -> None:
        for nombre in (
            "max_position_notional",
            "max_position_pct_equity",
            "max_daily_loss_pct",
            "max_daily_turnover_ratio",
        ):
            valor = getattr(self, nombre)
            if valor is not None and valor < 0:
                raise ValueError(f"{nombre} no puede ser negativo")

    def describe(self) -> dict[str, object]:
        return {
            "max_position_notional": self.max_position_notional,
            "max_position_pct_equity": self.max_position_pct_equity,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_daily_turnover_ratio": self.max_daily_turnover_ratio,
            "trip_on_daily_loss": self.trip_on_daily_loss,
        }


@dataclass(frozen=True)
class RiskEvent:
    """Algo que la capa de riesgo hizo. Se guarda para poder auditarlo."""

    timestamp: np.datetime64
    kind: str
    reason: str
    detail: str


def _dia(timestamp: np.datetime64) -> np.datetime64:
    """Fecha calendaria del timestamp, en UTC.

    ASSUMPTION: el dia de riesgo es el dia calendario UTC, no la sesion del
    mercado. Para un instrumento que opera de 9:30 a 16:00 hora de Nueva York
    coinciden; para cripto, que opera 24/7, el corte a medianoche UTC es una
    convencion y no un hecho del mercado. Cuando la Etapa 6 agregue instrumentos
    con sesiones distintas, esto tiene que pasar a preguntarle al ``Calendar``.
    """
    fecha: np.datetime64 = timestamp.astype("datetime64[D]")
    return fecha


class RiskLayer:
    """Aplica limites de riesgo a cada orden antes de que llegue al venue."""

    def __init__(self, limits: RiskLimits, clock: Clock) -> None:
        self._limits = limits
        self._clock = clock
        self._events: list[RiskEvent] = []
        self._kill_switch = False
        self._dia_actual: np.datetime64 | None = None
        self._equity_inicio_dia: float | None = None
        self._turnover_dia = 0.0
        self._equity_ultimo = 0.0

    # -- estado ---------------------------------------------------------

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    @property
    def events(self) -> list[RiskEvent]:
        return list(self._events)

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    def trip_kill_switch(self, detail: str) -> None:
        """Corta la operacion. Solo pasaran ordenes que reduzcan exposicion."""
        if self._kill_switch:
            return
        self._kill_switch = True
        self._events.append(
            RiskEvent(
                timestamp=self._clock.now(),
                kind="kill_switch_tripped",
                reason=RiskRejectReason.KILL_SWITCH.value,
                detail=detail,
            )
        )

    def reset_kill_switch(self, detail: str = "reposicion manual") -> None:
        """Repone la operacion. Deliberadamente manual: si se repusiera solo,
        no seria un kill switch sino una pausa."""
        if not self._kill_switch:
            return
        self._kill_switch = False
        self._events.append(
            RiskEvent(
                timestamp=self._clock.now(),
                kind="kill_switch_reset",
                reason="",
                detail=detail,
            )
        )

    # -- OrderGate ------------------------------------------------------

    def observe_bar(self, account: AccountSnapshot) -> None:
        """Actualiza el estado diario. Se llama una vez por barra."""
        hoy = _dia(self._clock.now())
        if self._dia_actual is None:
            # Primera barra de la corrida: no hay dia anterior al que anclar.
            self._dia_actual = hoy
            self._equity_inicio_dia = account.equity
            self._turnover_dia = 0.0
        elif hoy != self._dia_actual:
            # El ancla del dia nuevo es el equity con que **cerro el anterior**,
            # no el primero de hoy. Con barras diarias -donde una barra es un
            # dia entero- anclar al equity de la propia barra compararia el
            # equity contra si mismo y el limite no morderia nunca. Con barras
            # intradiarias las dos definiciones casi coinciden; con diarias, la
            # diferencia es entre un limite que funciona y uno decorativo.
            self._dia_actual = hoy
            self._equity_inicio_dia = self._equity_ultimo
            self._turnover_dia = 0.0
        self._equity_ultimo = account.equity

        perdida = self._perdida_del_dia(account.equity)
        limite = self._limits.max_daily_loss_pct
        if (
            limite is not None
            and perdida > limite
            and self._limits.trip_on_daily_loss
            and not self._kill_switch
        ):
            self.trip_kill_switch(
                f"perdida del dia {perdida:.4%} supera el limite {limite:.4%}"
            )

    def observe_fill(self, fill: Fill) -> None:
        """Acumula el turnover del dia con lo efectivamente llenado."""
        if fill.is_executed:
            self._turnover_dia += fill.notional

    def check(self, order: MarketOrder, account: AccountSnapshot) -> GateDecision:
        """Aprueba o rechaza. **Nunca devuelve una orden modificada.**"""
        reduce = self._reduce_exposicion(order, account)

        if self._kill_switch and not reduce:
            return self._rechazo(
                RiskRejectReason.KILL_SWITCH,
                "el kill switch esta activo; solo pasan ordenes que reducen "
                "exposicion",
            )

        # Una orden que reduce exposicion no puede violar un limite de tamano ni
        # de perdida: acerca la cuenta al estado seguro, no la aleja.
        if reduce:
            return GateDecision(approved=True)

        posicion_resultante = account.position + order.qty
        notional_resultante = abs(posicion_resultante) * account.mark_price

        limite_nocional = self._limits.max_position_notional
        if limite_nocional is not None and notional_resultante > limite_nocional:
            return self._rechazo(
                RiskRejectReason.MAX_POSITION_NOTIONAL,
                f"posicion resultante {notional_resultante:.2f} supera el "
                f"limite {limite_nocional:.2f}",
            )

        limite_pct = self._limits.max_position_pct_equity
        if limite_pct is not None and account.equity > 0:
            pct = notional_resultante / account.equity
            if pct > limite_pct:
                return self._rechazo(
                    RiskRejectReason.MAX_POSITION_PCT_EQUITY,
                    f"posicion resultante {pct:.4%} del equity supera el "
                    f"limite {limite_pct:.4%}",
                )

        limite_perdida = self._limits.max_daily_loss_pct
        if limite_perdida is not None:
            perdida = self._perdida_del_dia(account.equity)
            if perdida > limite_perdida:
                return self._rechazo(
                    RiskRejectReason.MAX_DAILY_LOSS,
                    f"perdida del dia {perdida:.4%} supera el limite "
                    f"{limite_perdida:.4%}",
                )

        limite_turnover = self._limits.max_daily_turnover_ratio
        if limite_turnover is not None and account.equity > 0:
            # El nocional de esta orden se estima al precio de marca: el de
            # ejecucion todavia no existe (es de la barra siguiente) y usarlo
            # seria lookahead.
            proyectado = self._turnover_dia + abs(order.qty) * account.mark_price
            ratio = proyectado / account.equity
            if ratio > limite_turnover:
                return self._rechazo(
                    RiskRejectReason.MAX_DAILY_TURNOVER,
                    f"turnover del dia {ratio:.4f}x el equity supera el limite "
                    f"{limite_turnover:.4f}x",
                )

        return GateDecision(approved=True)

    # -- internos -------------------------------------------------------

    def _perdida_del_dia(self, equity: float) -> float:
        """Fraccion positiva perdida desde el equity de apertura del dia."""
        inicio = self._equity_inicio_dia
        if inicio is None or inicio <= 0:
            return 0.0
        return max(0.0, (inicio - equity) / inicio)

    @staticmethod
    def _reduce_exposicion(order: MarketOrder, account: AccountSnapshot) -> bool:
        """Con ``allow_short=False``, reducir es vender teniendo posicion.

        No se exige que la venta sea menor o igual a la posicion: el venue ya
        rechaza vender mas de lo que hay, y duplicar esa validacion aqui haria
        que el mismo error se reportara con dos motivos distintos segun quien lo
        atrape primero.
        """
        return order.qty < 0 and account.position > 0

    def _rechazo(self, reason: RiskRejectReason, detail: str) -> GateDecision:
        self._events.append(
            RiskEvent(
                timestamp=self._clock.now(),
                kind="order_rejected",
                reason=reason.value,
                detail=detail,
            )
        )
        return GateDecision(approved=False, reason=reason.value, detail=detail)
