"""Eventos corporativos y calculo de retornos.

Distincion central del modulo:

- **Ajuste hacia adelante** (``total_return_index``): parte de la primera barra
  y aplica cada evento cuando ocurre. En cada ``t`` usa solo eventos con fecha
  ex <= t, asi que es point-in-time seguro y puede usarse como feature.
- **Ajuste hacia atras** (``backward_adjusted_close``): reescribe el pasado con
  los eventos futuros. Es lo que devuelven Yahoo y compania. Sirve para graficar
  y para nada mas; exige un flag explicito para poder auditarlo.

Convenciones:

- ``split_factor[t]``: acciones nuevas por accion antigua con efecto en la barra
  ``t`` (2.0 en un split 2:1). Los precios OHLC de ``t`` ya estan post-split,
  porque son los precios que efectivamente se negociaron.
- ``cash_dividend[t]``: dividendo por accion cuya fecha ex es la barra ``t``.
"""

from __future__ import annotations

import numpy as np

from data.errors import LookaheadError
from data.schema import BarSeries


def total_return_index(series: BarSeries, base: float = 1.0) -> np.ndarray:
    """Indice de retorno total, point-in-time seguro.

    ``TRI[0] = base`` y, para ``t >= 1``::

        TRI[t] = TRI[t-1] * (close[t] + div[t]) * split[t] / close[t-1]

    Un tenedor de una accion antes de ``t`` posee ``split[t]`` acciones en ``t``
    y cobra ``div[t]`` por accion antigua. Sin eventos se reduce al cociente de
    cierres, como debe ser.

    ASSUMPTION: el dividendo se reinvierte al cierre de la fecha ex y no se
    modelan retenciones fiscales. Documentado como limitacion del estudio.
    """
    close = np.asarray(series.close, dtype=np.float64)
    div = np.asarray(series.cash_dividend, dtype=np.float64)
    split = np.asarray(series.split_factor, dtype=np.float64)

    growth = np.ones(len(close), dtype=np.float64)
    growth[1:] = (close[1:] + div[1:]) * split[1:] / close[:-1]
    return base * np.cumprod(growth)


def total_return_log_returns(series: BarSeries) -> np.ndarray:
    """Retornos logaritmicos de retorno total. ``r[0] = 0`` por convencion.

    Los precios crudos no son estacionarios y no se usan como feature; esta es
    la serie que alimenta a los agentes.
    """
    tri = total_return_index(series)
    out = np.zeros(len(tri), dtype=np.float64)
    out[1:] = np.log(tri[1:] / tri[:-1])
    return out


def backward_adjusted_close(
    series: BarSeries, *, allow_lookahead: bool = False
) -> np.ndarray:
    """Cierres ajustados hacia atras. **Solo para graficar.**

    Cada precio anterior a un evento se reescribe usando informacion publicada
    despues de esa barra. Alimentar esto a un modelo o usarlo como precio de
    ejecucion es lookahead; por eso hay que pedirlo explicitamente.
    """
    if not allow_lookahead:
        raise LookaheadError(
            "backward_adjusted_close reescribe el pasado con eventos futuros. "
            "Para features y retornos usar total_return_index (point-in-time). "
            "Si es para un grafico, pasar allow_lookahead=True."
        )
    close = np.asarray(series.close, dtype=np.float64)
    div = np.asarray(series.cash_dividend, dtype=np.float64)
    split = np.asarray(series.split_factor, dtype=np.float64)

    n = len(close)
    factor = np.ones(n, dtype=np.float64)
    # factor[t-1] acumula los eventos ocurridos en t..n-1, que es justamente la
    # informacion futura que hace inutilizable esta serie fuera de un grafico.
    for t in range(n - 1, 0, -1):
        event = split[t] * (1.0 + div[t] / close[t])
        factor[t - 1] = factor[t] / event
    return close * factor
