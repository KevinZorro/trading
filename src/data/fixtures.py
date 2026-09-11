"""Fixtures sinteticos con senal conocida y optimo calculable.

**Esto es infraestructura de validacion, no datos de investigacion.** Ningun
resultado del estudio se reporta sobre estas series. Existen para responder una
sola pregunta -"¿el pipeline aprende una senal cuando la senal esta ahi?"- y
para responderla hace falta un techo conocido: sin optimo no hay forma de
distinguir "el agente aprendio poco" de "habia poco que aprender".

La etiqueta viaja con los datos, no solo en este docstring: ``series.source``
empieza con ``fixture:``, y ``source`` termina dentro de ``SimResult.config``
via ``BarSeries.meta()``. Cualquier resultado generado sobre un fixture es
identificable en el tracking de experimentos sin depender de que alguien se
acuerde de anotarlo.

Escalera de cinco niveles, cada uno aislando un fallo distinto:

===== ================================== ==========================================
Nivel Proceso                            Que falla aisla
===== ================================== ==========================================
0     alternancia determinista           el pipeline: obs -> accion -> orden -> reward
1     AR(1) con ruido, SNR parametrizable extraccion de senal degradada
2     AR(1) + costos calibrados          ¿la penalizacion llega al reward?
3     AR(1) con beta que invierte signo  adaptacion contra memorizacion
4     Heston puro, sin senal             sobreajuste de ruido (control negativo)
===== ================================== ==========================================

Tres decisiones de diseno que condicionan todo lo demas:

**La senal esta embebida en el precio, no en una columna aparte.** La
observacion del entorno (``envs.observation.ObservationBuilder``) es cerrada:
19 features derivadas del precio y de la cuenta. Una columna exogena de senal
seria invisible para el agente. El estado predictivo es por lo tanto el ultimo
log-retorno, que el agente ve como la feature ``log_return_{lookback-1}``.

**Gap cero.** ``open[t+1] == close[t]`` exactamente, porque el camino
intra-barra es continuo entre barras. Asi el precio de decision y el de
referencia de la ejecucion coinciden, mantener la posicion durante ``t+1``
captura exactamente ``r_{t+1}``, y el techo es exacto en vez de depender de la
realizacion del gap. **Limitacion:** estos fixtures no ejercitan la ruta de gap
del motor; esa la cubren los tests de la Etapa 1.

**Sin redondeo a tick y con un instrumento sin minimos.** El trabajo de este
objeto es ser una regla de medir: si el redondeo a tick moviera los retornos
realizados respecto del proceso disenado, el techo dejaria de ser exacto. Las
restricciones de venue (lote, nocional minimo, participacion) se miden en el
barrido de capital de la Etapa 6, no aqui.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.errors import DataError
from data.instruments import CommissionSchema, InstrumentSpec
from data.schema import BarSeries, FloatArray
from data.synthetic import RISK_REGIMES, HestonParams, generate_gbm_sv

StateArray = npt.NDArray[np.int8]

# Barras por defecto de los niveles 1-4. Con n=8000 el error estandar de beta
# estimado por OLS es sqrt((1-beta^2)/n) ~ 0.011, asi que las tolerancias de
# los tests estadisticos se derivan del tamano de muestra en vez de tantearse.
DEFAULT_N_BARS = 8_000

# Grilla del programa dinamico del nivel 2. 201 puntos sobre +-6 desvios cubren
# la distribucion estacionaria con un paso de ~0.06 desvios: mas fino que la
# banda de no-operar que los costos generan, que es lo que hay que resolver.
DP_GRID_POINTS = 201
DP_GRID_SPAN = 6.0
DP_MAX_SWEEPS = 20_000
DP_TOL = 1e-14

_SQRT2 = math.sqrt(2.0)

# Mismo criterio de tolerancia relativa que `eval.metrics.DISPERSION_NULA_REL`
# y `features.scaler.DESVIO_MINIMO`: por debajo de esto, la brecha entre el
# techo y estar siempre invertido es indistinguible de cero.
DISPERSION_NULA_REL = 1e-12

# Margen para declarar que el techo quedo por debajo de estar siempre invertido.
# La regla informada es optima en esperanza, no en cada camino realizado, asi que
# una diferencia chica es ruido; una grande es un techo mal especificado.
CEILING_TOL = 0.05


class FixtureError(DataError):
    """El fixture no se puede construir o el techo no se puede calcular."""


# ---------------------------------------------------------------------------
# Especificacion del proceso
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """AR(1) sobre log-retornos: ``r_{t+1} = mu + beta*r_t + sigma*eps``.

    Se parametriza por los **momentos estacionarios** (``drift``,
    ``sigma_target``) y no por ``mu`` y ``sigma`` directamente::

        mu(beta)    = drift * (1 - beta)
        sigma(beta) = sigma_target * sqrt(1 - beta^2)

    Dos razones, y ninguna es cosmetica:

    1. El R^2 del pronostico a un paso es exactamente ``beta^2``. El barrido de
       SNR del nivel 1 es literalmente un barrido de ``beta``, sin estimar nada.
    2. La media y la varianza marginales quedan **fijas** al mover ``beta``. Sin
       esto, el barrido de SNR confundiria SNR con regimen de volatilidad, y el
       cambio de regimen del nivel 3 cambiaria el drift ademas de la
       predictibilidad: dos efectos superpuestos e inseparables.

    ``beta_after_flip`` y ``flip_at`` son el nivel 3. ``sigma_target=0`` es el
    nivel 0: proceso degenerado y determinista, que admite ``|beta| = 1``.
    """

    drift: float
    beta: float
    sigma_target: float
    beta_after_flip: float | None = None
    flip_at: int | None = None
    r0: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_target < 0:
            raise FixtureError("sigma_target no puede ser negativo")
        limite = 1.0 if self.sigma_target == 0.0 else 1.0 - 1e-12
        parametros = (("beta", self.beta), ("beta_after_flip", self.beta_after_flip))
        for nombre, valor in parametros:
            if valor is None:
                continue
            if abs(valor) > limite:
                raise FixtureError(
                    f"{nombre}={valor!r} fuera del rango estacionario. Con "
                    "sigma_target>0 el proceso necesita |beta|<1 o la varianza "
                    "marginal no existe y el techo no esta definido."
                )
        if (self.beta_after_flip is None) != (self.flip_at is None):
            raise FixtureError(
                "beta_after_flip y flip_at van juntos: uno sin el otro deja el "
                "cambio de regimen a medio declarar"
            )
        if self.flip_at is not None and self.flip_at < 1:
            raise FixtureError("flip_at debe ser >= 1")

    @property
    def is_deterministic(self) -> bool:
        return self.sigma_target == 0.0

    def mu_for(self, beta: float) -> float:
        """Intercepto que deja la media estacionaria en ``drift``."""
        return self.drift * (1.0 - beta)

    def sigma_for(self, beta: float) -> float:
        """Desvio del shock que deja el desvio marginal en ``sigma_target``."""
        return self.sigma_target * math.sqrt(max(1.0 - beta * beta, 0.0))

    def betas(self, n: int) -> FloatArray:
        """``beta`` vigente en cada indice, con el cambio de regimen aplicado."""
        salida = np.full(n, self.beta, dtype=np.float64)
        if self.flip_at is not None and self.beta_after_flip is not None:
            salida[self.flip_at :] = self.beta_after_flip
        return salida

    def conditional_mean(self, signal: FloatArray, n: int | None = None) -> FloatArray:
        """``E[r_{t+1} | F_t] = mu + beta*r_t``, conocido en ``t``.

        Usa ``beta_{t+1}``, **no** ``beta_t``: el generador produce ``r_t`` con
        el beta indexado en ``t`` (``r[t] = mu(b_t) + b_t*r[t-1]``), asi que el
        que gobierna la transicion de ``t`` a ``t+1`` es el de ``t+1``. Dentro de
        un regimen los dos coinciden y la diferencia no se nota; en la barra
        anterior al cambio de regimen, usar ``beta_t`` aplica la regla del
        regimen viejo a una transicion que ya pertenece al nuevo.

        La ultima barra no tiene ``t+1``: su decision expira sin ejecutarse, asi
        que se reusa el ultimo beta y el valor no afecta a ningun resultado.
        """
        total = len(signal) if n is None else n
        betas = self.betas(total)
        siguientes = np.concatenate([betas[1:], betas[-1:]])
        return np.asarray(
            [
                self.mu_for(b) + b * float(s)
                for b, s in zip(siguientes, signal, strict=True)
            ],
            dtype=np.float64,
        )

    def describe(self) -> dict[str, object]:
        return {
            "drift": self.drift,
            "beta": self.beta,
            "sigma_target": self.sigma_target,
            "beta_after_flip": self.beta_after_flip,
            "flip_at": self.flip_at,
            "r0": self.r0,
            "r_squared": self.beta * self.beta,
            "mu": self.mu_for(self.beta),
            "sigma": self.sigma_for(self.beta),
        }


@dataclass(frozen=True)
class FixtureCosts:
    """Costos del fixture. **Solo datos**, igual que ``CommissionSchema``.

    No construye los modelos ejecutables de ``sim.costs``: ``data`` no depende
    de ``sim``. Quien corre el fixture arma la configuracion asi::

        SimConfig(
            initial_cash=...,
            spread=FixedBpsSpread(bps=fixture.costs.spread_bps),
            commission=SchemaCommission(schema=fixture.costs.commission),
        )

    El instrumento del fixture ya lleva el mismo ``CommissionSchema``, asi que
    olvidarse de pasar ``commission`` no cambia el resultado.

    **El slippage queda fuera de la calibracion a proposito.** Al capital de
    referencia del fixture la participacion sobre el volumen de la barra es del
    orden de 1e-5 y el impacto es ruido frente al spread. El regimen donde el
    slippage manda es el barrido de capital de la Etapa 6, y ahi se mide contra
    volumenes reales, no contra los de un fixture.
    """

    spread_bps: float = 0.0
    commission: CommissionSchema = field(
        default_factory=lambda: CommissionSchema(kind="percent", value=0.0)
    )

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise FixtureError("spread_bps no puede ser negativo")
        if self.commission.kind != "percent":
            raise FixtureError(
                "el fixture calibra con comision porcentual: es la unica que da "
                "un costo de ida y vuelta independiente del precio y del "
                "tamano, que es lo que la calibracion necesita invertir"
            )
        if self.commission.minimum != 0.0:
            raise FixtureError(
                "una comision minima hace que el costo relativo dependa del "
                "tamano de la orden y rompe la calibracion analitica"
            )

    @property
    def one_way(self) -> float:
        """Costo de girar el nocional una vez, como fraccion del nocional."""
        return self.spread_bps / 2e4 + self.commission.value

    @property
    def round_trip(self) -> float:
        """Ida y vuelta. Es el numero contra el que se compara el edge."""
        return 2.0 * self.one_way

    @property
    def half_spread_rel(self) -> float:
        return self.spread_bps / 2e4

    def describe(self) -> dict[str, object]:
        return {
            "spread_bps": self.spread_bps,
            "commission_kind": self.commission.kind,
            "commission_value": self.commission.value,
            "one_way": self.one_way,
            "round_trip": self.round_trip,
        }


ZERO_COSTS = FixtureCosts()


def fixture_instrument(
    symbol: str = "FIXT", *, commission: CommissionSchema | None = None
) -> InstrumentSpec:
    """Instrumento sin friccion de venue: fraccional, tick fino, sin minimos.

    Deliberado. Un rechazo por ``MIN_NOTIONAL`` o un redondeo de lote harian que
    el agente no alcanzara el techo por un motivo que no tiene nada que ver con
    aprender la senal, y el diagnostico de la Parte B se volveria ambiguo. Las
    restricciones de venue se miden donde corresponde: en el barrido de capital.
    """
    return InstrumentSpec(
        symbol=symbol,
        venue="FIXTURE",
        tick_size=1e-8,
        lot_size=1e-8,
        allow_fractional=True,
        qty_precision=8,
        min_order_qty=1e-8,
        min_notional=0.0,
        commission_schema=commission or CommissionSchema(kind="percent", value=0.0),
        asset_class="synthetic",
    )


# ---------------------------------------------------------------------------
# Generador
# ---------------------------------------------------------------------------


def _ar1_returns(
    n: int, spec: SignalSpec, seed: int | None, burn_in: int
) -> FloatArray:
    """Log-retornos del proceso. ``r[0]`` es el estado inicial, no un retorno.

    ``close[0] = s0`` por definicion, asi que la barra 0 no tiene retorno
    observable. ``r[0]`` guarda igual el estado que genero ``r[1]``, porque el
    techo lo necesita para evaluar la decision de la barra 0. Es la unica
    entrada de ``signal`` que **no** se puede recuperar de los precios de la
    serie; el episodio arranca despues del calentamiento de los indicadores
    (26 barras), asi que nunca es un punto de decision.
    """
    betas = spec.betas(n)
    r = np.zeros(n, dtype=np.float64)

    if spec.is_deterministic:
        # Sin ruido no hay semilla que valga: el fixture es identico siempre.
        previo = spec.r0
        r[0] = previo
        for t in range(1, n):
            b = float(betas[t])
            previo = spec.mu_for(b) + b * previo
            r[t] = previo
        return r

    if seed is None:
        raise FixtureError("un fixture con ruido necesita semilla explicita")

    rng = np.random.default_rng([seed, 0])
    eps = rng.standard_normal(burn_in + n)
    b0 = float(betas[0])
    # Arranque en la media estacionaria mas quemado: sin esto el transitorio
    # inicial tiene una media distinta de la del resto y el techo del primer
    # tramo no seria comparable con el del resto de la serie.
    previo = spec.drift
    for i in range(burn_in):
        previo = spec.mu_for(b0) + b0 * previo + spec.sigma_for(b0) * float(eps[i])
    r[0] = previo
    for t in range(1, n):
        b = float(betas[t])
        ruido = spec.sigma_for(b) * float(eps[burn_in + t])
        previo = spec.mu_for(b) + b * previo + ruido
        r[t] = previo
    return r


def _intrabar_paths(
    log_open: FloatArray,
    log_close: FloatArray,
    *,
    sub_steps: int,
    scale: float,
    rng: np.random.Generator | None,
) -> FloatArray:
    """Camino de log-precio dentro de cada barra, anclado en ambos extremos.

    Sin camino intra-barra, ``high = max(open, close)`` y ``low = min(...)``:
    ATR queda igual al rango de la barra y ``CorwinSchultzSpread`` devuelve
    cero, con lo cual el nivel 2 correria **con el spread apagado en silencio**.

    Con ``rng=None`` el camino es un arco deterministico (una onda completa, que
    se anula en ambos extremos): el nivel 0 no tiene ni una fuente de
    aleatoriedad, ni siquiera en el ``high``.
    """
    n = len(log_open)
    u = np.linspace(0.0, 1.0, sub_steps + 1)
    base = log_open[:, None] + (log_close - log_open)[:, None] * u[None, :]
    if rng is None:
        return np.asarray(base + scale * np.sin(2.0 * np.pi * u)[None, :])
    incrementos = rng.standard_normal((n, sub_steps)) * (scale / math.sqrt(sub_steps))
    camino = np.concatenate([np.zeros((n, 1)), np.cumsum(incrementos, axis=1)], axis=1)
    # Puente: se resta la interpolacion lineal del extremo, asi ambos extremos
    # quedan clavados y open/close siguen siendo exactamente los del proceso.
    puente = camino - u[None, :] * camino[:, -1:]
    return np.asarray(base + puente)


def generate_signal_bars(
    n_bars: int,
    *,
    spec: SignalSpec,
    seed: int | None = None,
    instrument: InstrumentSpec | None = None,
    s0: float = 100.0,
    start: str = "2020-01-01T21:00:00Z",
    freq: str = "1D",
    sub_steps: int = 8,
    intrabar_vol_frac: float = 0.6,
    base_volume: float = 1e7,
    volume_dispersion: float = 0.25,
    burn_in: int = 256,
    source: str = "fixture:signal",
) -> tuple[BarSeries, FloatArray]:
    """Barras OHLCV cuyo retorno de cierre a cierre sigue exactamente ``spec``.

    Devuelve ``(serie, signal)``, donde ``signal[t]`` es el log-retorno de la
    barra ``t`` -el estado visible en ``t``- y ``signal[t] ==
    log(close[t]/close[t-1])`` para todo ``t >= 1``.

    ``open[t] == close[t-1]`` exactamente: gap cero, ver el docstring del
    modulo. El volumen es independiente del retorno a proposito, para que la
    senal tenga **un solo** canal y "el agente aprendio del ultimo retorno" sea
    una afirmacion verdadera y no una de dos posibilidades.
    """
    if n_bars < 3:
        raise FixtureError("se necesitan al menos 3 barras")
    if sub_steps < 2:
        raise FixtureError("sub_steps debe ser >= 2 para que high y low existan")
    if intrabar_vol_frac < 0:
        raise FixtureError("intrabar_vol_frac no puede ser negativo")
    if s0 <= 0:
        raise FixtureError("s0 debe ser positivo")

    instrument = instrument or fixture_instrument()
    senal = _ar1_returns(n_bars, spec, seed, burn_in)

    pasos = senal.copy()
    pasos[0] = 0.0  # close[0] = s0: la barra 0 no arrastra retorno.
    log_close = math.log(s0) + np.cumsum(pasos)
    log_open = np.concatenate([[math.log(s0)], log_close[:-1]])

    escala_barra = spec.sigma_target if spec.sigma_target > 0 else abs(spec.r0)
    # Un stream por componente. Con un solo generador, cuantos numeros consume
    # el camino intra-barra depende de n_bars, y entonces el volumen de la
    # barra 10 cambiaria segun cuantas barras se pidieron: el pasado del
    # fixture dependeria del futuro. Hay un test de leakage que lo fija.
    rng: np.random.Generator | None
    if spec.is_deterministic:
        rng = None
    elif seed is None:  # pragma: no cover - _ar1_returns ya lo rechazo
        raise FixtureError("un fixture con ruido necesita semilla explicita")
    else:
        rng = np.random.default_rng([seed, 1])
    caminos = _intrabar_paths(
        log_open,
        log_close,
        sub_steps=sub_steps,
        scale=intrabar_vol_frac * escala_barra,
        rng=rng,
    )
    precios = np.exp(caminos)
    open_ = precios[:, 0].copy()
    close = precios[:, -1].copy()
    high = precios.max(axis=1)
    low = precios.min(axis=1)

    if rng is None or seed is None:
        volume = np.full(n_bars, base_volume, dtype=np.float64)
    else:
        rng_volumen = np.random.default_rng([seed, 2])
        volume = base_volume * np.exp(
            volume_dispersion * rng_volumen.standard_normal(n_bars)
            - 0.5 * volume_dispersion**2
        )

    timestamps = pd.date_range(start=pd.Timestamp(start), periods=n_bars, freq=freq)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")

    serie = BarSeries(
        instrument=instrument,
        freq=freq,
        timestamp=timestamps.tz_localize(None).to_numpy("datetime64[ns]"),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        split_factor=np.ones(n_bars),
        cash_dividend=np.zeros(n_bars),
        source=source,
    )
    return serie, senal


# ---------------------------------------------------------------------------
# Contabilidad del techo
#
# Criterio, unico y explicito: **maximizar el equity terminal**, equivalente a
# maximizar el crecimiento logaritmico medio por barra. Es lo que miden el
# retorno total y el CAGR, que son las metricas primarias del estudio.
#
# El reward del entorno (`NetReturnReward`, suma no descontada de retornos
# simples) tiene un optimo marginalmente distinto: su umbral esta en
# `E[R] > rate` y el de aca en `E[log(1+R)] > log(1+rate)`, o sea a sigma^2/2 de
# distancia. Con sigma = 1.2% por barra eso es 7e-5 contra un edge tipico de
# 2e-3: un 3% del edge. No se disimula, se declara.
# ---------------------------------------------------------------------------


def _commission_cost(schema: CommissionSchema, qty: float, price: float) -> float:
    """Replica de ``sim.costs.SchemaCommission.compute``.

    ``data`` no puede importar ``sim`` -la dependencia va en el otro sentido-,
    asi que la formula esta duplicada. La duplicacion esta guardada por un test
    que compara las dos implementaciones sobre una grilla de cantidades y
    precios; si alguna se mueve, el test lo dice.
    """
    notional = abs(qty) * price
    if schema.kind == "fixed":
        fee = schema.value
    elif schema.kind == "percent":
        fee = notional * schema.value
    else:
        fee = abs(qty) * schema.value
    fee = max(fee, schema.minimum)
    if schema.max_pct_notional is not None:
        fee = min(fee, notional * schema.max_pct_notional)
    return fee


def evaluate_states(
    close: FloatArray,
    states: StateArray,
    *,
    initial_cash: float = 1.0,
    safety: float = 1.0,
    rate_per_bar: float = 0.0,
    costs: FixtureCosts = ZERO_COSTS,
    first_decision: int = 0,
) -> FloatArray:
    """Curva de equity de una politica bang-bang, replicando el motor.

    ``states[t]`` es el peso decidido al cierre de ``t`` (0 o 1), que se ejecuta
    al open de ``t+1``. Sigue el mismo orden que ``Simulator._open_bar``:
    primero interes sobre el cash del periodo anterior, despues la ejecucion de
    lo decidido en ``t-1``, y recien ahi la marca a close.

    Que replique al motor no es una promesa: hay un test que corre esta funcion
    y una estrategia equivalente dentro de ``Simulator`` sobre la misma serie y
    exige que las dos curvas de equity coincidan.
    """
    n = len(close)
    if len(states) != n:
        raise FixtureError(f"states tiene {len(states)} entradas y close {n}")
    if not 0.0 < safety <= 1.0:
        raise FixtureError("safety debe estar en (0, 1]")
    if initial_cash <= 0:
        raise FixtureError("initial_cash debe ser positivo")

    equity = np.empty(n, dtype=np.float64)
    cash = initial_cash
    qty = 0.0
    equity[0] = initial_cash
    half = costs.half_spread_rel

    for t in range(1, n):
        cash *= 1.0 + rate_per_bar
        peso = float(states[t - 1]) if t - 1 >= first_decision else 0.0
        precio_ref = float(close[t - 1])  # gap cero: open[t] == close[t-1]
        objetivo = peso * safety * equity[t - 1] / precio_ref
        delta = objetivo - qty
        if delta != 0.0:
            signo = 1.0 if delta > 0 else -1.0
            precio_fill = precio_ref * (1.0 + signo * half)
            comision = _commission_cost(costs.commission, delta, precio_fill)
            cash -= delta * precio_fill + comision
            qty += delta
        equity[t] = cash + qty * float(close[t])
    return equity


def _gauss_hermite(nodes: int = 48) -> tuple[FloatArray, FloatArray]:
    """Nodos y pesos para ``E[f(X)]`` con ``X`` normal. Pesos suman 1."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return np.asarray(x, dtype=np.float64), np.asarray(
        w / math.sqrt(2.0 * math.pi), dtype=np.float64
    )


def _grilla_normal(
    points: int = 20_001, span: float = 10.0
) -> tuple[FloatArray, FloatArray]:
    """Nodos estandarizados y pesos de una normal, por regla del trapecio.

    Para integrandos con quiebre -``max(valor_de_mantener, valor_del_cash)``- la
    cuadratura gaussiana pierde su convergencia espectral y el error se vuelve
    de primer orden alrededor del quiebre. Una grilla fina converge O(h^2) sin
    depender de la suavidad, que es lo que hace falta aca.
    """
    x = np.linspace(-span, span, points)
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    w = pdf * (x[1] - x[0])
    return x, np.asarray(w / w.sum(), dtype=np.float64)


def _hold_log_value(
    m: FloatArray, sigma: FloatArray, *, safety: float, rate: float
) -> FloatArray:
    """``E[log(1 + safety*R' + (1-safety)*rate) | F_t]`` para cada ``t``.

    Con ``safety=1`` la expresion colapsa a ``m`` exactamente, porque
    ``log(1 + (e^{r'} - 1)) = r'``. Ese caso es el optimo analitico cerrado; el
    resto es cuadratura, y hay un test que verifica que converge a ``m`` cuando
    ``safety -> 1``.
    """
    if safety == 1.0 and rate == 0.0:
        return np.asarray(m, dtype=np.float64)
    x, w = _gauss_hermite()
    r = m[:, None] + sigma[:, None] * x[None, :]
    bruto = 1.0 + safety * np.expm1(r) + (1.0 - safety) * rate
    if np.any(bruto <= 0.0):
        raise FixtureError(
            "el factor de crecimiento se vuelve no positivo en la cuadratura: "
            "el proceso tiene retornos tan negativos que la posicion se arruina, "
            "y ahi el crecimiento logaritmico no esta definido"
        )
    return np.asarray(np.log(bruto) @ w, dtype=np.float64)


def myopic_states(
    conditional_mean: FloatArray,
    sigma: FloatArray,
    *,
    safety: float = 1.0,
    rate_per_bar: float = 0.0,
    first_decision: int = 0,
) -> StateArray:
    """Regla optima sin costos: invertir sii mantener crece mas que estar en cash.

    Sin friccion las decisiones se desacoplan y el optimo es miope y bang-bang.
    Con ``safety=1`` y ``cash_rate=0`` la regla es exactamente::

        w_t = 1  sii  mu + beta*r_t > 0

    Una linea, sin numerica, y es la que el test verifica con oraculo a mano.
    """
    valor_mantener = _hold_log_value(
        conditional_mean, sigma, safety=safety, rate=rate_per_bar
    )
    valor_cash = math.log1p(rate_per_bar)
    estados = (valor_mantener > valor_cash).astype(np.int8)
    estados[:first_decision] = 0
    return np.asarray(estados, dtype=np.int8)


def clairvoyant_states(
    close: FloatArray,
    *,
    safety: float = 1.0,
    rate_per_bar: float = 0.0,
    costs: FixtureCosts = ZERO_COSTS,
    first_decision: int = 0,
) -> StateArray:
    """Optimo ex-post sobre el camino realizado. **Cota dura, no criterio.**

    Programa dinamico exacto de dos estados sobre la serie: en cada barra se
    conoce el retorno realizado y se elige mantener o estar en cash, pagando el
    costo de una vuelta cada vez que el estado cambia. Sin costos se reduce a
    "mantener en las barras de retorno positivo".

    Ningun agente puede alcanzarlo: exige ver el ruido. Se expone para acotar
    por arriba, nunca como barra de aprobacion. Ignora la friccion de
    rebalanceo dentro de un tramo mantenido, lo cual solo lo hace mas alto: una
    cota superior que se queda corta en costos sigue siendo cota superior.
    """
    n = len(close)
    r = np.zeros(n, dtype=np.float64)
    r[1:] = close[1:] / close[:-1] - 1.0
    crecer = np.log(1.0 + safety * r + (1.0 - safety) * rate_per_bar)
    plano = math.log1p(rate_per_bar)
    cambio = math.log1p(-safety * costs.one_way)

    # valor[s] = mejor log-equity acumulado desde el final estando en el estado s
    # al cierre de la barra actual.
    valor = np.zeros(2, dtype=np.float64)
    decision = np.zeros((n, 2), dtype=np.int8)
    for t in range(n - 1, first_decision - 1, -1):
        nuevo = np.empty(2, dtype=np.float64)
        for s in (0, 1):
            opciones = []
            for a in (0, 1):
                paso = 0.0 if t + 1 >= n else float(crecer[t + 1] if a else plano)
                total = paso + (cambio if a != s else 0.0) + float(valor[a])
                opciones.append(total)
            mejor = 1 if opciones[1] > opciones[0] else 0
            decision[t, s] = mejor
            nuevo[s] = opciones[mejor]
        valor = nuevo

    estados = np.zeros(n, dtype=np.int8)
    s_actual = 0
    for t in range(first_decision, n):
        s_actual = int(decision[t, s_actual])
        estados[t] = s_actual
    return estados


@dataclass(frozen=True)
class HysteresisPolicy:
    """Politica optima con costos: dos umbrales sobre ``r_t``, uno por estado.

    Con costos el optimo deja de ser miope. Cambiar de estado cuesta, asi que la
    decision depende de donde se esta: se entra con un edge mas exigente del que
    hace falta para quedarse. Esa brecha -``no_trade_band``- es exactamente lo
    que "operar selectivamente" significa, y **es cero sii los costos son
    cero**. Hay un test que lo fija en los dos sentidos.

    ``enter_below``: estando plano, se entra si ``r_t < enter_below``.
    ``exit_above``: estando invertido, se sale si ``r_t > exit_above``.
    (Con ``beta > 0`` los sentidos se invierten; ``decreasing`` lo indica.)
    """

    enter_below: float
    exit_above: float
    decreasing: bool
    growth_per_bar: float
    grid: FloatArray
    policy: npt.NDArray[np.int8]

    @property
    def no_trade_band(self) -> float:
        return abs(self.exit_above - self.enter_below)

    def states_for(self, signal: FloatArray, *, first_decision: int = 0) -> StateArray:
        """Aplica la politica sobre un camino realizado."""
        paso = float(self.grid[1] - self.grid[0])
        indices = np.clip(
            np.rint((signal - self.grid[0]) / paso).astype(np.int64),
            0,
            len(self.grid) - 1,
        )
        estados = np.zeros(len(signal), dtype=np.int8)
        s = 0
        for t in range(first_decision, len(signal)):
            s = int(self.policy[int(indices[t]), s])
            estados[t] = s
        return estados

    def describe(self) -> dict[str, object]:
        return {
            "enter_below": self.enter_below,
            "exit_above": self.exit_above,
            "no_trade_band": self.no_trade_band,
            "decreasing": self.decreasing,
            "growth_per_bar": self.growth_per_bar,
        }


def _normal_cdf_row(x: FloatArray) -> FloatArray:
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(v) / _SQRT2)) for v in x], dtype=np.float64
    )


def _transition_matrix(
    grid: FloatArray, mu: float, beta: float, sigma: float
) -> FloatArray:
    """Kernel del AR(1) discretizado sobre ``grid`` (metodo de Tauchen)."""
    bordes = np.concatenate([[-np.inf], (grid[:-1] + grid[1:]) / 2.0, [np.inf]])
    filas = []
    for r in grid:
        z = (bordes - (mu + beta * float(r))) / sigma
        cdf = _normal_cdf_row(np.asarray(z, dtype=np.float64))
        filas.append(np.diff(cdf))
    matriz = np.asarray(filas, dtype=np.float64)
    return np.asarray(matriz / matriz.sum(axis=1, keepdims=True))


def solve_optimal_policy(
    spec: SignalSpec,
    costs: FixtureCosts,
    *,
    safety: float = 1.0,
    rate_per_bar: float = 0.0,
    grid_points: int = DP_GRID_POINTS,
    span: float = DP_GRID_SPAN,
) -> HysteresisPolicy:
    """Optimo ex-ante con costos: iteracion de valor relativa sobre ``(r, s)``.

    El planteo es analitico -una ecuacion de Bellman de recompensa media sobre
    el estado ``(r_t, posicion)``- y la resolucion es numerica sobre una grilla.
    No hay forma cerrada: el costo de cambiar de estado acopla las decisiones y
    la politica optima deja de ser miope. Se declara asi y no como "formula".

    Dos validaciones lo atan a algo verificable: con costo cero converge a
    ``myopic_states``, y su resultado sobre el camino queda siempre por debajo
    del clarividente.
    """
    if spec.flip_at is not None:
        raise FixtureError(
            "el programa dinamico supone un proceso estacionario; con cambio de "
            "regimen no hay una politica optima unica que valga para toda la serie"
        )
    if spec.is_deterministic:
        raise FixtureError("el proceso determinista no necesita programa dinamico")

    beta = spec.beta
    mu = spec.mu_for(beta)
    sigma = spec.sigma_for(beta)
    grid = np.linspace(
        spec.drift - span * spec.sigma_target,
        spec.drift + span * spec.sigma_target,
        grid_points,
    )
    P = _transition_matrix(grid, mu, beta, sigma)

    crecer = np.log(1.0 + safety * np.expm1(grid) + (1.0 - safety) * rate_per_bar)
    plano = np.full(grid_points, math.log1p(rate_per_bar))
    barra = np.stack([plano, P @ crecer], axis=1)
    # Modelo de primer orden del costo de cambiar de estado. El cargo exacto
    # -que aplica `evaluate_states`- es `(P1/P0)*(1-c) - c` por vuelta
    # completa; este es el mismo numero a primer orden en `c` y difiere en
    # O(c^2), unos 4e-6 con los costos del nivel 2. La politica que sale de
    # aca se evalua despues con el cargo exacto, asi que `informed` sigue
    # siendo alcanzable; lo unico que la aproximacion afecta es donde cae el
    # umbral, en el cuarto decimal.
    cambio = math.log1p(-safety * costs.one_way)

    h = np.zeros((grid_points, 2), dtype=np.float64)
    politica = np.zeros((grid_points, 2), dtype=np.int8)
    g = 0.0
    for _ in range(DP_MAX_SWEEPS):
        continuacion = np.stack([P @ h[:, 0], P @ h[:, 1]], axis=1)  # (grid, accion)
        q = barra + continuacion  # valor de tomar la accion a desde r_i
        nuevo = np.empty_like(h)
        for s in (0, 1):
            # El costo se paga solo cuando la accion difiere del estado actual.
            penalizacion = np.array(
                [0.0 if s == 0 else cambio, cambio if s == 0 else 0.0]
            )
            con_costo = q + penalizacion
            politica[:, s] = (con_costo[:, 1] > con_costo[:, 0]).astype(np.int8)
            nuevo[:, s] = con_costo.max(axis=1)
        delta = nuevo - h
        g = float(delta.mean())
        if float(delta.max() - delta.min()) < DP_TOL:
            h = nuevo - g
            break
        h = nuevo - g
    else:  # pragma: no cover - la cadena es ergodica y converge mucho antes
        raise FixtureError("la iteracion de valor no convergio")

    decreciente = beta < 0.0
    entrar = _umbral(grid, politica[:, 0], decreciente)
    salir = _umbral(grid, politica[:, 1], decreciente)
    return HysteresisPolicy(
        enter_below=entrar,
        exit_above=salir,
        decreasing=decreciente,
        growth_per_bar=g,
        grid=grid,
        policy=politica,
    )


def _umbral(
    grid: FloatArray, columna: npt.NDArray[np.int8], decreciente: bool
) -> float:
    """Frontera de una politica monotona. Falla si hay mas de un cruce.

    Un cruce doble significa que la grilla es demasiado gruesa o que el kernel
    esta mal armado. Devolver "el primer cruce" y seguir escondería el problema.
    """
    cambios = np.flatnonzero(np.diff(columna.astype(np.int64)) != 0)
    if len(cambios) == 0:
        return float(grid[-1] if decreciente else grid[0])
    if len(cambios) > 1:
        raise FixtureError(
            f"la politica optima no es monotona en r ({len(cambios)} cruces): "
            "la grilla del programa dinamico es demasiado gruesa"
        )
    i = int(cambios[0])
    return float((grid[i] + grid[i + 1]) / 2.0)


# ---------------------------------------------------------------------------
# El fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ceilings:
    """Techos y referencias de un fixture, como curvas de equity.

    Son curvas y no numeros para que ``eval.metrics`` se aplique tal cual sobre
    ellas: el techo tiene Sharpe, drawdown y CAGR igual que cualquier corrida, y
    comparar solo retornos totales contra un techo esconde que el techo puede
    alcanzarse por caminos con riesgo muy distinto.

    - ``informed``: **la barra a superar.** Conoce el proceso, no el ruido.
    - ``clairvoyant``: cota dura ex-post. Inalcanzable por construccion.
    - ``always_long``: accion 1.0 en cada barra. Es el peso constante que el
      agente produce si aprende "estar siempre invertido"; **no** es
      ``agents.baselines.BuyAndHold``, que compra una vez y no rebalancea.
    - ``always_flat``: no operar nunca, devengando ``cash_rate``.
    - ``memorizer``: solo en el nivel 3. La regla optima de la primera mitad
      aplicada a toda la serie.
    """

    informed: FloatArray
    clairvoyant: FloatArray
    always_long: FloatArray
    always_flat: FloatArray
    informed_states: StateArray
    clairvoyant_states: StateArray
    informed_below_reference: bool
    expected_growth_per_bar: float | None
    rule: str
    memorizer: FloatArray | None = None
    policy: HysteresisPolicy | None = None

    @property
    def informed_total_return(self) -> float:
        return float(self.informed[-1] / self.informed[0] - 1.0)

    @property
    def time_invested(self) -> float:
        """Fraccion de barras en las que el optimo esta invertido."""
        return float(np.mean(self.informed_states))

    @property
    def turnover_count(self) -> int:
        """Cambios de estado del optimo. En el nivel 4 tiene que ser 0."""
        return int(np.count_nonzero(np.diff(self.informed_states.astype(np.int64))))

    def capture(self, equity: FloatArray) -> float | None:
        """Fraccion del camino entre ``always_long`` e ``informed`` capturada.

        Es la metrica con la que la Parte B tiene que reportar el barrido de
        SNR. El retorno crudo no sirve: al bajar el SNR el techo baja tambien, y
        una degradacion trivial del techo se leeria como degradacion del agente.

        Devuelve ``None`` en dos casos, y los dos importan:

        1. **El techo coincide con estar siempre invertido** (el nivel 4): el
           denominador es cero y la fraccion no esta definida. Un cero se leeria
           como "no capturo nada", que es distinto de "no habia nada que
           capturar", y esa diferencia **es** el resultado del control negativo.
           Mismo criterio que ``eval.trades`` con ``win_rate``.
        2. **El techo queda por debajo de estar siempre invertido**
           (``informed_below_reference``): el denominador es negativo y el
           cociente premia alejarse del techo. Estar siempre invertido es una
           politica factible, asi que un optimo sistematicamente por debajo no es
           un resultado, es un techo mal especificado. Devolver un numero ahi
           seria reportar como desempeno lo que es un bug.
        """
        if self.informed_below_reference:
            return None

        def crecimiento(curva: FloatArray) -> float:
            return math.log(float(curva[-1]) / float(curva[0]))

        techo = crecimiento(self.informed) - crecimiento(self.always_long)
        if abs(techo) <= DISPERSION_NULA_REL:
            return None
        return (crecimiento(equity) - crecimiento(self.always_long)) / techo

    def describe(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "informed_total_return": self.informed_total_return,
            "clairvoyant_total_return": float(
                self.clairvoyant[-1] / self.clairvoyant[0] - 1.0
            ),
            "always_long_total_return": float(
                self.always_long[-1] / self.always_long[0] - 1.0
            ),
            "expected_growth_per_bar": self.expected_growth_per_bar,
            "time_invested": self.time_invested,
            "turnover_count": self.turnover_count,
            "informed_below_reference": self.informed_below_reference,
            "policy": self.policy.describe() if self.policy else None,
        }


@dataclass(frozen=True)
class Fixture:
    """Serie con senal conocida, su proceso y su optimo. Validacion, no estudio."""

    level: int
    name: str
    series: BarSeries
    spec: SignalSpec
    costs: FixtureCosts
    signal: FloatArray
    seed: int | None
    validation_only: bool = True

    def __post_init__(self) -> None:
        if len(self.signal) != len(self.series):
            raise FixtureError(
                f"signal tiene {len(self.signal)} entradas y la serie "
                f"{len(self.series)} barras"
            )
        senal = np.ascontiguousarray(self.signal, dtype=np.float64)
        senal.flags.writeable = False
        object.__setattr__(self, "signal", senal)
        if not self.series.source.startswith("fixture:"):
            raise FixtureError(
                "el source de un fixture tiene que empezar con 'fixture:'. La "
                "etiqueta viaja dentro de SimResult.config y es lo que hace "
                "identificable un resultado corrido sobre datos de validacion."
            )

    def __len__(self) -> int:
        return len(self.series)

    @property
    def expected_next_log_return(self) -> FloatArray:
        """``E[r_{t+1} | F_t]``, funcion de datos hasta ``t`` y nada mas."""
        salida = self.spec.conditional_mean(self.signal)
        salida.flags.writeable = False
        return salida

    @property
    def sigma_by_bar(self) -> FloatArray:
        """Desvio condicional vigente en cada barra (cambia en el flip)."""
        return np.asarray(
            [self.spec.sigma_for(float(b)) for b in self.spec.betas(len(self))],
            dtype=np.float64,
        )

    # -- techos ---------------------------------------------------------

    def ceilings(
        self,
        *,
        safety: float = 1.0,
        cash_rate: float = 0.0,
        bars_per_year: float = 252.0,
        initial_cash: float = 1.0,
        first_decision: int = 0,
    ) -> Ceilings:
        """Calcula el techo para una configuracion de ejecucion concreta.

        ``safety`` **no es un detalle**: el ``TargetWeightSizer`` del entorno usa
        0.98, asi que una accion de 1.0 invierte el 98% del equity. La Parte B
        tiene que pedir el techo con el mismo ``safety`` con el que corre, o va a
        leer un deficit del 2% como un fallo de aprendizaje.

        Con ``safety=1.0`` el techo es **teorico y el venue lo rechazaria**: la
        orden pide todo el cash y no deja con que pagar spread y comision, asi
        que el motor la rechaza por ``INSUFFICIENT_CASH``. Hay un test que lo
        demuestra. Se expone igual porque es la referencia sin friccion de
        ejecucion, pero no es una barra que ningun agente pueda alcanzar.
        """
        rate = periodic_rate_per_bar(cash_rate, bars_per_year)
        close = np.asarray(self.series.close, dtype=np.float64)
        comun = {
            "initial_cash": initial_cash,
            "safety": safety,
            "rate_per_bar": rate,
            "costs": self.costs,
            "first_decision": first_decision,
        }

        politica: HysteresisPolicy | None = None
        if self.costs.one_way > 0.0:
            politica = solve_optimal_policy(
                self.spec, self.costs, safety=safety, rate_per_bar=rate
            )
            estados = politica.states_for(self.signal, first_decision=first_decision)
            regla = (
                f"programa dinamico con histeresis: entrar si r_t "
                f"{'<' if politica.decreasing else '>'} {politica.enter_below:.6f}, "
                f"salir si r_t {'>' if politica.decreasing else '<'} "
                f"{politica.exit_above:.6f}"
            )
            crecimiento: float | None = politica.growth_per_bar
        else:
            estados = myopic_states(
                self.expected_next_log_return,
                self.sigma_by_bar,
                safety=safety,
                rate_per_bar=rate,
                first_decision=first_decision,
            )
            regla = "mantener sii E[log(1+R_{t+1})|F_t] > log(1+rate)"
            if safety == 1.0:
                regla = "mantener sii mu + beta*r_t > log(1+rate)"
            crecimiento = self._expected_growth(safety=safety, rate=rate)

        memorizador = None
        if self.spec.flip_at is not None:
            sin_flip = replace(self.spec, beta_after_flip=None, flip_at=None)
            estados_mem = myopic_states(
                sin_flip.conditional_mean(self.signal),
                np.full(len(self), sin_flip.sigma_for(sin_flip.beta)),
                safety=safety,
                rate_per_bar=rate,
                first_decision=first_decision,
            )
            memorizador = evaluate_states(close, estados_mem, **comun)  # type: ignore[arg-type]

        unos = np.ones(len(self), dtype=np.int8)
        unos[:first_decision] = 0
        curva_informado = evaluate_states(close, estados, **comun)  # type: ignore[arg-type]
        curva_larga = evaluate_states(close, unos, **comun)  # type: ignore[arg-type]
        # `always_long` es una politica factible: el optimo informado no puede
        # quedar sistematicamente por debajo. Cuando pasa, el techo esta mal
        # especificado -por ejemplo, aplicando el beta del regimen equivocado- y
        # hay que decirlo en vez de dejar que `capture` produzca un cociente con
        # denominador negativo. La tolerancia deja pasar el ruido de camino: la
        # regla es optima en esperanza, no en cada realizacion.
        debajo = bool(
            math.log(float(curva_informado[-1]) / float(curva_informado[0]))
            < math.log(float(curva_larga[-1]) / float(curva_larga[0])) - CEILING_TOL
        )
        videntes = clairvoyant_states(
            close,
            safety=safety,
            rate_per_bar=rate,
            costs=self.costs,
            first_decision=first_decision,
        )
        return Ceilings(
            informed=curva_informado,
            clairvoyant=evaluate_states(close, videntes, **comun),  # type: ignore[arg-type]
            always_long=curva_larga,
            always_flat=evaluate_states(
                close,
                np.zeros(len(self), dtype=np.int8),
                **comun,  # type: ignore[arg-type]
            ),
            informed_states=estados,
            clairvoyant_states=videntes,
            informed_below_reference=debajo,
            expected_growth_per_bar=crecimiento,
            rule=regla,
            memorizer=memorizador,
            policy=politica,
        )

    def _expected_growth(self, *, safety: float, rate: float) -> float | None:
        """Crecimiento logaritmico esperado por barra del optimo sin costos.

        Poblacional, no muestral: integra sobre la distribucion estacionaria de
        ``r_t``, no sobre el camino generado. Con ``safety=1`` y sin cash rate
        hay forma cerrada -``E[max(m, 0)]`` de una normal- y el test la compara
        contra esta cuadratura.
        """
        if self.spec.is_deterministic:
            return None
        # Grilla fina y no Gauss-Hermite: el integrando lleva un `max`, o sea un
        # quiebre en el umbral de la politica, y la cuadratura gaussiana supone
        # un integrando suave. Medido: con 48 nodos el error contra la forma
        # cerrada era del 0.5%, suficiente para que un techo mal calculado
        # pasara por bueno.
        x, w = _grilla_normal()
        total = 0.0
        for beta, peso in self._regime_weights():
            mu = self.spec.mu_for(beta)
            sigma = self.spec.sigma_for(beta)
            r = self.spec.drift + self.spec.sigma_target * x
            m = mu + beta * r
            valor = _hold_log_value(m, np.full(len(m), sigma), safety=safety, rate=rate)
            mejor = np.maximum(valor, math.log1p(rate))
            total += peso * float(mejor @ w)
        return total

    def _regime_weights(self) -> list[tuple[float, float]]:
        n = len(self)
        if self.spec.flip_at is None or self.spec.beta_after_flip is None:
            return [(self.spec.beta, 1.0)]
        corte = min(self.spec.flip_at, n)
        return [
            (self.spec.beta, corte / n),
            (self.spec.beta_after_flip, (n - corte) / n),
        ]

    # -- capacidad ------------------------------------------------------

    def capacity_check(
        self,
        initial_cash: float,
        *,
        safety: float = 0.98,
        max_participation: float = 0.10,
    ) -> dict[str, float | bool]:
        """¿El venue deja alcanzar el techo con este capital?

        Un techo que la capacidad de la barra no permite alcanzar es un techo
        mal calculado: el agente se quedaria corto por llenados parciales y el
        diagnostico de la Parte B culparia al aprendizaje.
        """
        techos = self.ceilings(safety=safety, initial_cash=initial_cash)
        close = np.asarray(self.series.close, dtype=np.float64)
        volumen = np.asarray(self.series.volume, dtype=np.float64)
        equity = techos.informed
        estados = techos.informed_states.astype(np.float64)
        qty_objetivo = estados * safety * equity / close
        delta = np.abs(np.diff(np.concatenate([[0.0], qty_objetivo])))
        participacion = delta[1:] / volumen[1:]
        operadas = delta[1:][delta[1:] > 0]
        nocional_min = (
            float((operadas * close[1:][delta[1:] > 0]).min()) if len(operadas) else 0.0
        )
        pico = float(participacion.max()) if len(participacion) else 0.0
        return {
            "peak_participation": pico,
            "capacity_limit": max_participation,
            "min_trade_notional": nocional_min,
            "min_notional_required": self.series.instrument.min_notional,
            "binds": bool(
                pico > max_participation
                or nocional_min < self.series.instrument.min_notional
            ),
        }

    # -- particion y serializacion ---------------------------------------

    def split(
        self, *, train: float = 0.6, validation: float = 0.2, test: float = 0.2
    ) -> tuple[Fixture, Fixture, Fixture]:
        """Particion contigua train/validation/test.

        Contigua y en ese orden porque el walk-forward de la Parte B lo exige y
        porque una particion aleatoria sobre una serie con autocorrelacion
        filtra informacion entre los tramos.
        """
        total = train + validation + test
        if abs(total - 1.0) > 1e-9:
            raise FixtureError(f"las fracciones suman {total}, deben sumar 1")
        n = len(self)
        i = int(round(n * train))
        j = i + int(round(n * validation))
        cortes = ((0, i), (i, j), (j, n))
        nombres = ("train", "validation", "test")
        partes = []
        for (a, b), sufijo in zip(cortes, nombres, strict=True):
            if b - a < 3:
                raise FixtureError(f"el tramo {sufijo} queda con {b - a} barras")
            partes.append(
                replace(
                    self,
                    name=f"{self.name}:{sufijo}",
                    series=self.series.slice(a, b),
                    signal=self.signal[a:b],
                    spec=self._spec_para_tramo(a, b),
                )
            )
        return partes[0], partes[1], partes[2]

    def _spec_para_tramo(self, start: int, stop: int) -> SignalSpec:
        """Traslada el cambio de regimen a las coordenadas del tramo.

        Sin esto, un tramo posterior al flip conserva ``flip_at`` del padre -un
        indice que su propia serie ni siquiera alcanza- y ``betas()`` devuelve el
        beta **anterior** al cambio para datos que ya son del regimen nuevo. La
        regla optima queda exactamente invertida y el techo pasa a ser peor que
        no operar.

        Medido cuando aparecio: sobre el tramo de validacion del nivel 3, el
        techo "optimo" perdia el 89% del capital mientras estar siempre invertido
        perdia el 1.3%. Lo detecto correr el protocolo, no un test.
        """
        if self.spec.flip_at is None or self.spec.beta_after_flip is None:
            return self.spec
        corte = self.spec.flip_at - start
        if corte <= 0:
            # El tramo entero es posterior al cambio: ya no hay flip que aplicar.
            return replace(
                self.spec,
                beta=self.spec.beta_after_flip,
                beta_after_flip=None,
                flip_at=None,
            )
        if corte >= stop - start:
            # El tramo entero es anterior al cambio.
            return replace(self.spec, beta_after_flip=None, flip_at=None)
        return replace(self.spec, flip_at=corte)

    def describe(self) -> dict[str, Any]:
        """Todo lo necesario para reproducir el fixture, junto al resultado."""
        return {
            "level": self.level,
            "name": self.name,
            "seed": self.seed,
            "validation_only": self.validation_only,
            "spec": self.spec.describe(),
            "costs": self.costs.describe(),
            "series": self.series.meta(),
        }


def periodic_rate_per_bar(annual_rate: float, bars_per_year: float) -> float:
    """Misma formula que ``SimConfig.rate_per_bar`` y ``eval.metrics.periodic_rate``.

    Duplicada porque ``data`` no depende de ``sim`` ni de ``eval``; hay un test
    que compara las tres.
    """
    if annual_rate == 0.0:
        return 0.0
    if bars_per_year <= 0:
        raise FixtureError("bars_per_year debe ser positivo")
    return float((1.0 + annual_rate) ** (1.0 / bars_per_year) - 1.0)


# ---------------------------------------------------------------------------
# Los cinco niveles
# ---------------------------------------------------------------------------

# Amplitud del nivel 0. Elegida por capacidad, no por estetica: el optimo crece
# `amplitude` cada dos barras, y sobre 2000 barras eso son ~7.4x. Subirla haria
# que la orden optima terminara chocando contra `max_participation * volumen` y
# el techo dejaria de ser alcanzable.
LEVEL_0_AMPLITUDE = 0.002
LEVEL_0_BARS = 2_000

# Regimen comun de los niveles 1-3: 1.2% de desvio por barra (~19% anualizado) y
# un drift de 0.03% por barra (~7.8% anual). Numeros de una accion grande, no de
# un fixture imposible.
DEFAULT_SIGMA = 0.012
DEFAULT_DRIFT = 0.0003
DEFAULT_BETA = -0.3

# Barrido de SNR del nivel 1. El R^2 del pronostico es beta^2, asi que estos
# cuatro valores son R^2 de 1%, 4%, 9% y 25%.
SNR_SWEEP_BETAS: tuple[float, ...] = (-0.1, -0.2, -0.3, -0.5)

# Comision del nivel 2: 5 bps por lado, porcentual. Del orden de un broker
# retail de cripto sin descuentos.
LEVEL_2_COMMISSION = 0.0005
LEVEL_2_TARGET_QUANTILE = 0.30


def calibrate_costs(
    spec: SignalSpec,
    *,
    target_quantile: float = LEVEL_2_TARGET_QUANTILE,
    commission_value: float = LEVEL_2_COMMISSION,
) -> FixtureCosts:
    """Resuelve el spread que pone el costo de ida y vuelta en un cuantil dado.

    El edge condicional ``m_t = mu + beta*r_t`` es normal con media ``drift`` y
    desvio ``|beta| * sigma_target`` -la media sale exacta porque
    ``mu = drift*(1-beta)``-, asi que el costo que deja el edge por encima en
    una fraccion ``q`` de las barras se despeja en cerrado::

        round_trip = drift + |beta| * sigma_target * Phi^{-1}(1 - q)

    Con ``q = 0.30``, en el 70% de las barras el edge de una barra **no** cubre
    la ida y vuelta. El comportamiento correcto deja de ser "seguir la senal" y
    pasa a ser "seguirla solo cuando paga", que es justo lo que el nivel 2 mide.
    """
    if not 0.0 < target_quantile < 1.0:
        raise FixtureError("target_quantile debe estar en (0, 1)")
    if commission_value < 0:
        raise FixtureError("commission_value no puede ser negativo")
    z = NormalDist().inv_cdf(1.0 - target_quantile)
    round_trip = spec.drift + abs(spec.beta) * spec.sigma_target * z
    resto = round_trip - 2.0 * commission_value
    if resto <= 0:
        raise FixtureError(
            f"la comision sola ({2 * commission_value:.6f} de ida y vuelta) ya "
            f"supera el costo objetivo ({round_trip:.6f}): no queda spread que "
            "calibrar. Bajar commission_value o subir target_quantile."
        )
    return FixtureCosts(
        spread_bps=resto * 1e4,
        commission=CommissionSchema(kind="percent", value=commission_value),
    )


def level_0_deterministic(
    n_bars: int = LEVEL_0_BARS, *, amplitude: float = LEVEL_0_AMPLITUDE
) -> Fixture:
    """Nivel 0: alternancia exacta. Sin ruido, sin costos, sin semilla.

    ``r_{t+1} = -r_t`` con ``r_0 = amplitude``: el ultimo retorno determina el
    siguiente sin ambiguedad, y el optimo es "invertir sii el ultimo retorno fue
    negativo". El camino intra-barra tambien es determinista, asi que el fixture
    entero es identico en cada generacion y no admite semilla.

    Si el agente no se acerca al techo aca, el problema no esta en los
    hiperparametros: esta en el pipeline.
    """
    if amplitude <= 0:
        raise FixtureError("amplitude debe ser positiva")
    spec = SignalSpec(drift=0.0, beta=-1.0, sigma_target=0.0, r0=amplitude)
    serie, senal = generate_signal_bars(
        n_bars,
        spec=spec,
        instrument=fixture_instrument("FIXT0"),
        source=f"fixture:level_0:amplitude={amplitude}",
    )
    return Fixture(
        level=0,
        name="level_0_deterministic",
        series=serie,
        spec=spec,
        costs=ZERO_COSTS,
        signal=senal,
        seed=None,
    )


def level_1_noisy(
    n_bars: int = DEFAULT_N_BARS,
    *,
    seed: int,
    beta: float = DEFAULT_BETA,
    sigma_target: float = DEFAULT_SIGMA,
    drift: float = DEFAULT_DRIFT,
) -> Fixture:
    """Nivel 1: la misma senal con ruido. ``R^2 = beta^2``, y nada mas cambia.

    La varianza marginal queda fija al mover ``beta`` (ver ``SignalSpec``), asi
    que el barrido de SNR mueve **solo** la predictibilidad. Sin eso, un agente
    que rinde peor con menos SNR podria estar rindiendo peor por operar en un
    regimen de volatilidad distinto, y las dos explicaciones serian inseparables.
    """
    spec = SignalSpec(drift=drift, beta=beta, sigma_target=sigma_target)
    serie, senal = generate_signal_bars(
        n_bars,
        spec=spec,
        seed=seed,
        instrument=fixture_instrument("FIXT1"),
        source=f"fixture:level_1:beta={beta}:seed={seed}",
    )
    return Fixture(
        level=1,
        name=f"level_1_noisy:beta={beta}",
        series=serie,
        spec=spec,
        costs=ZERO_COSTS,
        signal=senal,
        seed=seed,
    )


def level_2_costly(
    n_bars: int = DEFAULT_N_BARS,
    *,
    seed: int,
    beta: float = DEFAULT_BETA,
    sigma_target: float = DEFAULT_SIGMA,
    drift: float = DEFAULT_DRIFT,
    target_quantile: float = LEVEL_2_TARGET_QUANTILE,
    commission_value: float = LEVEL_2_COMMISSION,
) -> Fixture:
    """Nivel 2: la senal del nivel 1 con costos calibrados contra su propio edge.

    El costo de ida y vuelta se despeja para que el edge de una barra lo supere
    en el ``target_quantile`` de las barras. Operar en cada barra deja de ser
    rentable; el optimo desarrolla una banda de no-operar (ver
    ``HysteresisPolicy``). Si el agente opera igual con y sin costos, la
    penalizacion no esta llegando al reward.
    """
    spec = SignalSpec(drift=drift, beta=beta, sigma_target=sigma_target)
    costos = calibrate_costs(
        spec, target_quantile=target_quantile, commission_value=commission_value
    )
    serie, senal = generate_signal_bars(
        n_bars,
        spec=spec,
        seed=seed,
        instrument=fixture_instrument("FIXT2", commission=costos.commission),
        source=f"fixture:level_2:beta={beta}:seed={seed}",
    )
    return Fixture(
        level=2,
        name=f"level_2_costly:beta={beta}",
        series=serie,
        spec=spec,
        costs=costos,
        signal=senal,
        seed=seed,
    )


def level_3_regime_flip(
    n_bars: int = DEFAULT_N_BARS,
    *,
    seed: int,
    beta: float = DEFAULT_BETA,
    beta_after_flip: float | None = None,
    sigma_target: float = DEFAULT_SIGMA,
    drift: float = DEFAULT_DRIFT,
    flip_at: int | None = None,
) -> Fixture:
    """Nivel 3: la relacion se invierte a mitad de serie.

    ``beta`` pasa de negativo (reversion) a positivo (momentum) con la **misma**
    media y el **mismo** desvio marginales: lo unico que cambia es el signo de
    la predictibilidad. Un cambio que ademas moviera el drift produciria dos
    efectos superpuestos e inseparables.

    El indice del flip se expone pero no es observable: la observacion del
    entorno no tiene feature de tiempo, asi que adaptarse solo es posible
    infiriendo el regimen de los retornos recientes. Ese es el argumento a favor
    de la politica recurrente, y ``Ceilings.memorizer`` lo vuelve medible.
    """
    corte = flip_at if flip_at is not None else n_bars // 2
    spec = SignalSpec(
        drift=drift,
        beta=beta,
        sigma_target=sigma_target,
        beta_after_flip=-beta if beta_after_flip is None else beta_after_flip,
        flip_at=corte,
    )
    serie, senal = generate_signal_bars(
        n_bars,
        spec=spec,
        seed=seed,
        instrument=fixture_instrument("FIXT3"),
        source=f"fixture:level_3:beta={beta}:flip={corte}:seed={seed}",
    )
    return Fixture(
        level=3,
        name=f"level_3_regime_flip:beta={beta}",
        series=serie,
        spec=spec,
        costs=ZERO_COSTS,
        signal=senal,
        seed=seed,
    )


def log_drift_per_bar(mu: float, theta: float, bars_per_year: float) -> float:
    """Drift **logaritmico** por barra de un proceso de Heston: ``(mu - theta/2)/bpy``.

    El generador integra ``d log S = (mu - v_t/2) dt + ...``, asi que el drift de
    los log-retornos **no** es ``mu*dt``. La diferencia no es cosmetica: con
    ``mu=0.08`` y ``theta=0.09`` el aritmetico es 3.17e-4 por barra y el
    logaritmico 1.39e-4, o sea menos de la mitad, y el estadistico ``t`` que
    decide si el drift es detectable sale 1.16 contra 0.51.

    Se usa el valor de largo plazo ``theta`` en lugar del ``v_t`` instantaneo: el
    drift condicional varia con la volatilidad, y este es su valor poblacional.
    """
    return (mu - theta / 2.0) / bars_per_year


def drift_t_population(
    mu: float, theta: float, n_bars: int, bars_per_year: float = 252.0
) -> float:
    """``t`` poblacional del drift sobre ``n_bars``: cuan detectable es en esa muestra.

    Sale de dividir el drift logaritmico por barra entre el error estandar de su
    media::

        t = (mu - theta/2) * sqrt(n) / sqrt(theta * bars_per_year)

    Es la formula con la que se calibra el nivel 4b, y la que hay que mirar
    **antes** de exigirle a un agente que aprenda un drift.
    """
    if n_bars < 2:
        raise FixtureError("hacen falta al menos 2 barras")
    if theta <= 0 or bars_per_year <= 0:
        raise FixtureError("theta y bars_per_year deben ser positivos")
    return (mu - theta / 2.0) * math.sqrt(n_bars) / math.sqrt(theta * bars_per_year)


def level_4_control(
    n_bars: int = DEFAULT_N_BARS,
    *,
    seed: int,
    regime: str = "medium",
    bars_per_year: float = 252.0,
) -> Fixture:
    """Nivel 4: Heston puro. Control negativo.

    Envuelve ``generate_gbm_sv`` sin tocarlo, para que el control sea el
    generador ya validado en la Etapa 1. Su afirmacion analitica es la del
    CLAUDE.md: ``E[r_{t+1} | F_t] = mu*dt`` constante e independiente de la
    historia, asi que el optimo informado es **estar siempre invertido** y el
    techo coincide exactamente con ``always_long``. El ``SignalSpec`` degenerado
    (``beta = 0``) no es un parche: es el modelo correcto del proceso, y por eso
    la misma maquinaria de techos se aplica sin excepciones.

    Si un agente "gana" aca, esta sobreajustando ruido.
    """
    if regime not in RISK_REGIMES:
        raise FixtureError(f"regimen desconocido {regime!r}")
    params = RISK_REGIMES[regime]
    serie = generate_gbm_sv(
        n_bars,
        seed=seed,
        instrument=fixture_instrument("FIXT4"),
        params=params,
        bars_per_year=bars_per_year,
        overnight_gap_frac=0.0,
        round_to_tick=False,
    )
    serie = replace(serie, source=f"fixture:level_4:regime={regime}:seed={seed}")
    close = np.asarray(serie.close, dtype=np.float64)
    senal = np.zeros(n_bars, dtype=np.float64)
    senal[1:] = np.log(close[1:] / close[:-1])
    spec = SignalSpec(
        drift=log_drift_per_bar(params.mu, params.theta, bars_per_year),
        beta=0.0,
        sigma_target=math.sqrt(params.theta / bars_per_year),
    )
    return Fixture(
        level=4,
        name=f"level_4_control:{regime}",
        series=serie,
        spec=spec,
        costs=ZERO_COSTS,
        signal=senal,
        seed=seed,
    )


# Drift anual del nivel 4b. Calibrado -no elegido- para que el drift sea
# detectable en la ventana de entrenamiento: con theta=0.09 y 4800 barras de
# train, `drift_t_population(0.42, 0.09, 4800)` da t = 5.46, y el t muestral se
# distribuye alrededor de ese valor con desvio ~1, asi que P(t < 3) ~ 0.7%.
#
# **No es un drift realista**: 42% anual con 30% de volatilidad es un Sharpe
# aritmetico de ~1.4 sostenido durante veinte anos. El fixture no pretende
# parecerse a un mercado; pretende contestar una pregunta que el nivel 4a no
# puede contestar, que es si el agente reconoce un drift **cuando existe**.
LEVEL_4B_MU = 0.42


def level_4b_detectable_drift(
    n_bars: int = DEFAULT_N_BARS,
    *,
    seed: int,
    mu: float = LEVEL_4B_MU,
    bars_per_year: float = 252.0,
) -> Fixture:
    """Nivel 4b: Heston con drift **detectable**. Sigue sin senal direccional.

    Identico al nivel 4a salvo por ``mu``. Que cambie una sola cosa es el punto:
    si el agente se comporta distinto entre 4a y 4b, la unica explicacion
    disponible es el drift.

    El nivel 4a pregunta "¿el agente inventa senal donde no la hay?" y no puede
    contestar "¿reconoce drift cuando lo hay?", porque en su fixture el drift no
    es detectable en la muestra (t ~ 0.5). Exigirle ahi que converja a estar
    invertido le pide aprender algo que la serie no contiene. Este nivel separa
    esa segunda pregunta y la hace contestable.
    """
    base = RISK_REGIMES["medium"]
    params = HestonParams(
        mu=mu,
        v0=base.v0,
        theta=base.theta,
        kappa=base.kappa,
        xi=base.xi,
        rho=base.rho,
    )
    serie = generate_gbm_sv(
        n_bars,
        seed=seed,
        instrument=fixture_instrument("FIXT4B"),
        params=params,
        bars_per_year=bars_per_year,
        overnight_gap_frac=0.0,
        round_to_tick=False,
    )
    serie = replace(serie, source=f"fixture:level_4b:mu={mu}:seed={seed}")
    close = np.asarray(serie.close, dtype=np.float64)
    senal = np.zeros(n_bars, dtype=np.float64)
    senal[1:] = np.log(close[1:] / close[:-1])
    spec = SignalSpec(
        drift=log_drift_per_bar(mu, params.theta, bars_per_year),
        beta=0.0,
        sigma_target=math.sqrt(params.theta / bars_per_year),
    )
    return Fixture(
        level=4,
        name=f"level_4b_detectable_drift:mu={mu}",
        series=serie,
        spec=spec,
        costs=ZERO_COSTS,
        signal=senal,
        seed=seed,
    )


#: Registro de los cinco niveles. La Parte B recorre esto en orden.
LEVELS: dict[str, Any] = {
    "level_0": level_0_deterministic,
    "level_1": level_1_noisy,
    "level_2": level_2_costly,
    "level_3": level_3_regime_flip,
    "level_4": level_4_control,
    "level_4b": level_4b_detectable_drift,
}
