"""Metricas sobre una serie de equity.

Funciones puras sobre arrays, no metodos de ``SimResult``: las mismas metricas
tienen que poder aplicarse a una serie que no salio del simulador (un benchmark
externo, una curva de buy-and-hold construida a mano, la salida de un broker en
la Etapa 7). Acoplarlas al resultado del motor obligaria a fabricar un
``SimResult`` falso cada vez que hay que medir algo que no simulamos nosotros.

Convenciones del modulo:

- Una serie de ``n`` observaciones tiene ``n - 1`` periodos. Un ano de barras
  diarias son 253 observaciones, no 252.
- Los retornos son **simples** (aritmeticos), no logaritmicos. El Sharpe se
  define sobre retornos aritmeticos; calcularlo sobre logaritmicos lo subestima
  de forma sistematica y creciente con la volatilidad.
- El drawdown se reporta como fraccion **positiva**: 0.25 es una caida del 25%.
- Ninguna funcion anualiza en silencio un periodo corto. Anualizan siempre, y
  ``years_elapsed`` esta expuesto para que quien reporte pueda marcar cuando la
  extrapolacion no es creible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data.schema import FloatArray


class MetricError(ValueError):
    """La serie no permite calcular la metrica pedida."""


# Por debajo de esta dispersion *relativa* a la magnitud de los retornos, el
# desvio estandar es ruido de coma flotante y no una medida de riesgo.
#
# El caso no es teorico: una serie que crece exactamente 10% por barra
# ([100, 110, 121, 133.1]) no produce retornos identicos en float64 -110/100 y
# 133.1/121 difieren en el ultimo bit- y sin este umbral el Sharpe sale del
# orden de 1e15 en vez de infinito. En un barrido de semillas y regimenes ese
# numero gana cualquier ranking sin significar nada, y una comparacion entre el
# agente A y el B se decidiria por error de redondeo.
DISPERSION_NULA_REL = 1e-12


# ---------------------------------------------------------------------------
# Utilidades de base
# ---------------------------------------------------------------------------


def periodic_rate(annual_rate: float, bars_per_year: float) -> float:
    """Tasa por barra equivalente a una anual compuesta.

    Es la misma formula que ``SimConfig.rate_per_bar``, que es la que el motor
    usa para devengar interes sobre el cash ocioso. Si las dos divergieran, el
    Sharpe descontaria una tasa libre de riesgo distinta de la que el equity
    efectivamente devengo, y la estrategia que esta fuera del mercado saldria
    premiada o castigada por un error de contabilidad. Hay un test que las
    compara.
    """
    if bars_per_year <= 0:
        raise MetricError("bars_per_year debe ser positivo")
    if annual_rate == 0.0:
        return 0.0
    return float((1.0 + annual_rate) ** (1.0 / bars_per_year) - 1.0)


def _validar(equity: FloatArray) -> FloatArray:
    serie = np.asarray(equity, dtype=np.float64)
    if serie.ndim != 1:
        raise MetricError(f"la serie de equity debe ser 1-D, tiene ndim={serie.ndim}")
    if len(serie) < 2:
        raise MetricError("se necesitan al menos 2 observaciones de equity")
    if not np.all(np.isfinite(serie)):
        raise MetricError("la serie de equity tiene valores no finitos")
    if serie[0] <= 0:
        raise MetricError(
            f"el equity inicial debe ser positivo, es {serie[0]!r}: sin una base "
            "positiva no hay retorno que calcular"
        )
    return serie


def first_ruin_index(equity: FloatArray) -> int | None:
    """Primera barra con equity <= 0, o ``None`` si nunca ocurre.

    La serie marcada a close no puede llegar aqui con ``allow_short=False`` y
    sin apalancamiento. La de liquidacion si: si el cash es cero y el producto
    de cerrar la posicion no cubre la comision de salida, el contrafactual de
    liquidar es negativo. Ese es justamente el caso que hay que poder reportar.
    """
    serie = np.asarray(equity, dtype=np.float64)
    indices = np.flatnonzero(serie <= 0.0)
    return int(indices[0]) if len(indices) else None


def truncate_at_ruin(equity: FloatArray) -> tuple[FloatArray, int | None]:
    """Corta la serie en la primera barra no positiva, incluyendola.

    Despues de tocar cero no hay retorno definido: ``equity[t] / equity[t-1]``
    divide por cero y contamina todo lo que venga despues con ``inf`` o ``nan``.
    Se corta ahi y se devuelve el indice para que el reporte lo muestre en vez
    de esconderlo detras de un ``nan``.
    """
    ruina = first_ruin_index(equity)
    if ruina is None:
        return np.asarray(equity, dtype=np.float64), None
    return np.asarray(equity, dtype=np.float64)[: ruina + 1], ruina


def years_elapsed(n_bars: int, bars_per_year: float) -> float:
    """Anos que cubren ``n_bars`` observaciones. Hay ``n_bars - 1`` periodos."""
    if bars_per_year <= 0:
        raise MetricError("bars_per_year debe ser positivo")
    if n_bars < 2:
        raise MetricError("se necesitan al menos 2 observaciones")
    return (n_bars - 1) / bars_per_year


def simple_returns(equity: FloatArray) -> FloatArray:
    """Retornos simples barra a barra. Longitud ``n - 1``."""
    serie = _validar(equity)
    out: FloatArray = serie[1:] / serie[:-1] - 1.0
    return out


def excess_returns(
    equity: FloatArray, bars_per_year: float, risk_free: float
) -> FloatArray:
    """Retornos por encima de la tasa libre de riesgo, barra a barra."""
    return simple_returns(equity) - periodic_rate(risk_free, bars_per_year)


# ---------------------------------------------------------------------------
# Retorno
# ---------------------------------------------------------------------------


def total_return(equity: FloatArray) -> float:
    """Retorno total del periodo. Puede ser menor que -1 si la serie se hunde."""
    serie = _validar(equity)
    return float(serie[-1] / serie[0] - 1.0)


def cagr(equity: FloatArray, bars_per_year: float) -> float:
    """Tasa de crecimiento anual compuesta.

    Con equity final no positivo devuelve -1.0: se perdio todo. La formula de
    potencia fraccionaria no esta definida sobre una base negativa y no tiene
    sentido inventar un numero peor que "todo".
    """
    serie = _validar(equity)
    years = years_elapsed(len(serie), bars_per_year)
    if serie[-1] <= 0:
        return -1.0
    return float((serie[-1] / serie[0]) ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Riesgo
# ---------------------------------------------------------------------------


def annual_volatility(equity: FloatArray, bars_per_year: float) -> float:
    """Desviacion estandar anualizada de los retornos simples.

    ASSUMPTION: la anualizacion por raiz de ``bars_per_year`` supone retornos
    independientes. Los retornos de una estrategia con posicion persistente
    estan autocorrelacionados y esta cifra los subestima. Se reporta igual
    porque es la convencion, pero no es una medida de riesgo defendible por si
    sola.
    """
    retornos = simple_returns(equity)
    if len(retornos) < 2:
        return 0.0
    return float(np.std(retornos, ddof=1) * math.sqrt(bars_per_year))


def sharpe(equity: FloatArray, bars_per_year: float, risk_free: float = 0.0) -> float:
    """Sharpe anualizado sobre retornos simples en exceso de la tasa sin riesgo.

    Casos degenerados, explicitos en vez de ``nan``:

    - Volatilidad cero y exceso medio cero (serie constante): 0.0.
    - Volatilidad cero y exceso medio distinto de cero: infinito con signo. Es
      lo que es: un retorno sin variabilidad. Deberia leerse como bug de datos,
      no como una estrategia excelente.

    "Volatilidad cero" se evalua contra ``DISPERSION_NULA_REL``, no contra cero
    exacto. Ver el comentario de esa constante: una serie de retornos constantes
    no lo es exactamente en coma flotante.
    """
    exceso = excess_returns(equity, bars_per_year, risk_free)
    if len(exceso) < 2:
        raise MetricError("se necesitan al menos 3 observaciones para el Sharpe")
    media = float(np.mean(exceso))
    desvio = float(np.std(exceso, ddof=1))
    escala = max(abs(media), float(np.max(np.abs(exceso))))
    if escala == 0.0:
        return 0.0
    if desvio <= DISPERSION_NULA_REL * escala:
        if media == 0.0:
            return 0.0
        return math.inf if media > 0 else -math.inf
    return float(media / desvio * math.sqrt(bars_per_year))


def sortino(equity: FloatArray, bars_per_year: float, risk_free: float = 0.0) -> float:
    """Sortino anualizado. Solo la desviacion a la baja entra al denominador.

    ASSUMPTION: la desviacion a la baja es la raiz del segundo momento parcial
    inferior sobre **todas** las observaciones, no solo sobre las negativas.
    Dividir entre la cantidad de observaciones negativas es un error frecuente
    que infla el ratio de las estrategias que rara vez pierden, que son
    justamente las que hay que mirar con mas desconfianza.
    """
    exceso = excess_returns(equity, bars_per_year, risk_free)
    if len(exceso) < 2:
        raise MetricError("se necesitan al menos 3 observaciones para el Sortino")
    media = float(np.mean(exceso))
    a_la_baja = np.minimum(exceso, 0.0)
    desvio_baja = float(math.sqrt(float(np.mean(a_la_baja**2))))
    escala = max(abs(media), float(np.max(np.abs(exceso))))
    if escala == 0.0:
        return 0.0
    if desvio_baja <= DISPERSION_NULA_REL * escala:
        return 0.0 if media == 0.0 else math.inf
    return float(media / desvio_baja * math.sqrt(bars_per_year))


@dataclass(frozen=True)
class Drawdown:
    """Peor caida desde un maximo previo.

    ``depth`` es positiva: 0.25 es una caida del 25% desde el pico. Puede
    superar 1.0 si el equity se vuelve negativo.
    """

    depth: float
    peak_index: int
    trough_index: int
    recovery_index: int | None
    duration_bars: int
    underwater_bars: int


def max_drawdown(equity: FloatArray) -> Drawdown:
    """Maximo drawdown y su geometria.

    ``duration_bars`` es del pico al valle. ``underwater_bars`` va del pico a la
    recuperacion, o hasta el final de la serie si nunca recupera. Las dos cosas
    importan: una caida del 20% que recupera en tres barras y una que sigue
    abierta al final del periodo son riesgos distintos.
    """
    serie = _validar(equity)
    picos = np.maximum.accumulate(serie)
    caidas = 1.0 - serie / picos
    valle = int(np.argmax(caidas))
    profundidad = float(caidas[valle])

    if profundidad <= 0.0:
        # Serie que nunca cae por debajo de un maximo previo.
        return Drawdown(
            depth=0.0,
            peak_index=0,
            trough_index=0,
            recovery_index=0,
            duration_bars=0,
            underwater_bars=0,
        )

    pico = int(np.argmax(serie[: valle + 1]))
    nivel_pico = float(serie[pico])
    posteriores = np.flatnonzero(serie[valle:] >= nivel_pico)
    recuperacion = int(valle + posteriores[0]) if len(posteriores) else None
    bajo_agua = (
        recuperacion - pico if recuperacion is not None else len(serie) - 1 - pico
    )
    return Drawdown(
        depth=profundidad,
        peak_index=pico,
        trough_index=valle,
        recovery_index=recuperacion,
        duration_bars=valle - pico,
        underwater_bars=bajo_agua,
    )


def calmar(equity: FloatArray, bars_per_year: float) -> float:
    """CAGR dividido por el maximo drawdown.

    Sin drawdown el cociente no esta acotado: se devuelve infinito con signo, o
    0.0 si tampoco hubo retorno.
    """
    crecimiento = cagr(equity, bars_per_year)
    caida = max_drawdown(equity).depth
    if caida == 0.0:
        if crecimiento == 0.0:
            return 0.0
        return math.inf if crecimiento > 0 else -math.inf
    return crecimiento / caida
