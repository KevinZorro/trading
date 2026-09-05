"""Construccion de la observacion. Aqui vive la garantia anti-leakage del entorno.

El generador ``Simulator.drive`` impide adelantar el cursor sin decidir, pero no
puede impedir que el codigo del entorno tenga la ``BarSeries`` en la mano: el
entorno la necesita para construir el simulador. La garantia real es esta clase.

``build`` recibe una :class:`~sim.engine.Decision` y **nada mas**. Todo lo que la
``Decision`` contiene es del presente o del pasado: la vista esta limitada a
``[0..t]``, la cuenta es la del cierre de ``t``, y el fill y el veto son de la
barra que acaba de abrir. No hay forma de leer ``t+1`` desde aca, no por
disciplina sino porque el objeto no lo tiene.

El test que lo demuestra construye dos series identicas hasta ``t`` y distintas
despues, y verifica que las observaciones hasta ``t`` son bit-identicas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schema import BarSeries, FloatArray
from features.scaler import FeatureScaler
from features.technical import atr, bollinger, macd, normalized_volume, rsi
from sim.engine import Decision
from sim.orders import OrderStatus
from sim.view import AccountSnapshot, MarketView


@dataclass(frozen=True)
class ObservationSpec:
    """Ventanas de los indicadores. Determina el calentamiento del episodio."""

    lookback: int = 10
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_k: float = 2.0
    volume_period: int = 20

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback debe ser positivo")

    @property
    def warmup(self) -> int:
        """Barras necesarias antes de que toda la observacion sea valida.

        Es el maximo de lo que pide cada indicador, no la suma. Evaluar uno con
        menos historia de la que necesita devuelve un numero que parece valido y
        no lo es, asi que el episodio no empieza hasta cubrirlos a todos.
        """
        return max(
            self.lookback + 1,
            self.rsi_period + 1,
            self.macd_slow,
            self.atr_period + 1,
            self.bollinger_period,
            self.volume_period,
        )

    def describe(self) -> dict[str, object]:
        return {
            "lookback": self.lookback,
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "atr_period": self.atr_period,
            "bollinger_period": self.bollinger_period,
            "bollinger_k": self.bollinger_k,
            "volume_period": self.volume_period,
            "warmup": self.warmup,
        }


# Columnas que se normalizan con estadisticas de train. Las de cuenta y las de
# resultado quedan fuera: ya viven en rangos acotados y naturales, y aplicarles
# un z-score las sacaria de ellos sin ganar nada.
class ObservationBuilder:
    """Ensambla el vector de observacion. No calcula indicadores: los llama."""

    def __init__(self, spec: ObservationSpec | None = None) -> None:
        self.spec = spec or ObservationSpec()
        self._names, self._scale_mask = self._layout(self.spec)

    @staticmethod
    def _layout(spec: ObservationSpec) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        nombres: list[str] = []
        escalar: list[bool] = []

        # -- mercado: se normaliza con estadisticas de train ---------------
        for i in range(spec.lookback):
            nombres.append(f"log_return_{i}")
            escalar.append(True)
        for nombre in (
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_rel",
            "bollinger_pct_b",
            "bollinger_width",
            "volume_norm",
        ):
            nombres.append(nombre)
            escalar.append(True)

        # -- cuenta y resultado: ya acotadas, pasan sin escalar ------------
        for nombre in (
            "position_weight",
            "cash_weight",
            "friction_gap",
            "last_order_blocked",
            "last_order_rejected",
            "last_fill_ratio",
        ):
            nombres.append(nombre)
            escalar.append(False)

        return tuple(nombres), tuple(escalar)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def scale_mask(self) -> tuple[bool, ...]:
        return self._scale_mask

    def __len__(self) -> int:
        return len(self._names)

    def build(self, decision: Decision) -> FloatArray:
        """Vector crudo, sin normalizar. El escalador se aplica despues.

        Recibe solo la ``Decision``: no tiene acceso a la serie ni a nada
        posterior a ``t``.
        """
        spec = self.spec
        view = decision.view
        account = decision.account

        valores: list[float] = []
        valores.extend(float(r) for r in view.log_returns(spec.lookback))

        cierres = view.history("close", spec.macd_slow)
        cierres_rsi = view.history("close", spec.rsi_period + 1)
        cierres_bb = view.history("close", spec.bollinger_period)
        altos = view.history("high", spec.atr_period + 1)
        bajos = view.history("low", spec.atr_period + 1)
        cierres_atr = view.history("close", spec.atr_period + 1)
        volumenes = view.history("volume", spec.volume_period)

        valores.append(rsi(cierres_rsi, spec.rsi_period))
        linea, senal, histograma = macd(
            cierres, spec.macd_fast, spec.macd_slow, spec.macd_signal
        )
        precio = view.close()
        # MACD y ATR estan en unidades de precio y no son comparables entre
        # instrumentos ni entre niveles de precio del mismo instrumento. Se
        # dividen por el cierre para volverlos adimensionales; el precio crudo
        # no entra en la observacion porque no es estacionario.
        valores.extend([linea / precio, senal / precio, histograma / precio])
        valores.append(atr(altos, bajos, cierres_atr, spec.atr_period) / precio)
        pct_b, ancho = bollinger(cierres_bb, spec.bollinger_period, spec.bollinger_k)
        valores.extend([pct_b, ancho])
        valores.append(normalized_volume(volumenes, spec.volume_period))

        # -- cuenta ---------------------------------------------------------
        equity = account.equity
        if equity > 0:
            # La posicion es la REALMENTE llenada: sale del ledger del
            # simulador, no de lo que el agente pretendio. Tras un fill parcial
            # esas dos divergen y el agente tiene que ver la real.
            valores.append(account.position_value / equity)
            valores.append(account.cash / equity)
            valores.append((equity - decision.equity_liquidation) / equity)
        else:
            valores.extend([0.0, 0.0, 0.0])

        # -- resultado de la decision anterior -------------------------------
        # Sin esto, "no quise operar" y "no pude" se ven identicos desde la
        # posicion, y el agente no puede aprender la diferencia.
        vetada = decision.last_gate_rejection is not None
        fill = decision.last_fill
        rechazada = fill is not None and fill.status is OrderStatus.REJECTED
        valores.append(1.0 if vetada else 0.0)
        valores.append(1.0 if rechazada else 0.0)
        if fill is not None and fill.qty_requested != 0.0:
            valores.append(fill.qty_filled / fill.qty_requested)
        else:
            valores.append(0.0)

        vector = np.asarray(valores, dtype=np.float64)
        if len(vector) != len(self._names):
            raise ValueError(
                f"la observacion tiene {len(vector)} valores y el layout declara "
                f"{len(self._names)}"
            )
        return vector


def fit_scaler_on_train(
    train: BarSeries,
    builder: ObservationBuilder | None = None,
) -> FeatureScaler:
    """Ajusta el escalador recorriendo **solo** la porcion de train.

    Camina ``MarketView(train, t)`` desde el fin del calentamiento, asi que las
    estadisticas no pueden contener ni una barra de test: la serie de test no
    esta en el objeto. Es el mecanismo, no la promesa.

    Las columnas de cuenta y de resultado no dependen de la serie -son cero
    mientras no hay posicion ni ordenes previas- y ademas quedan fuera de la
    mascara de escalado, asi que ajustarlas con una cuenta vacia no las afecta.
    """
    constructor = builder or ObservationBuilder()
    inicio = constructor.spec.warmup
    if len(train) <= inicio:
        raise ValueError(
            f"la porcion de train tiene {len(train)} barras y el calentamiento "
            f"necesita {inicio}: no queda nada con que ajustar"
        )

    vacia = AccountSnapshot(
        t=0, cash=0.0, position=0.0, mark_price=1.0, equity=0.0, pending_qty=0.0
    )
    filas = []
    for t in range(inicio, len(train)):
        view = MarketView(train, t)
        decision = Decision(
            view=view,
            account=AccountSnapshot(
                t=t,
                cash=vacia.cash,
                position=vacia.position,
                mark_price=view.close(),
                equity=vacia.equity,
                pending_qty=0.0,
            ),
            equity_liquidation=0.0,
        )
        filas.append(constructor.build(decision))

    return FeatureScaler.fit(
        np.asarray(filas, dtype=np.float64),
        constructor.names,
        constructor.scale_mask,
    )
