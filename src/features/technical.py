"""Indicadores tecnicos. Funciones puras sobre arrays, evaluadas en la ultima barra.

Todas reciben la historia que ``MarketView.history`` entrega, o sea ``[0..t]``,
asi que no pueden ver el futuro aunque quieran. Ninguna guarda estado entre
llamados: dos evaluaciones sobre la misma ventana dan el mismo numero.

Cada funcion declara su calentamiento minimo en ``*_WARMUP``. El entorno arranca
el episodio despues del maximo de todos ellos; evaluar un indicador con menos
historia de la que necesita devuelve un numero que parece valido y no lo es.
"""

from __future__ import annotations

import numpy as np

from data.schema import FloatArray

RSI_WARMUP = 1
MACD_WARMUP = 1
ATR_WARMUP = 1
BOLLINGER_WARMUP = 1
VOLUME_WARMUP = 1


def _validar(valores: FloatArray, minimo: int, nombre: str) -> FloatArray:
    serie = np.asarray(valores, dtype=np.float64)
    if serie.ndim != 1:
        raise ValueError(f"{nombre} debe ser 1-D, tiene ndim={serie.ndim}")
    if len(serie) < minimo:
        raise ValueError(
            f"{nombre} necesita al menos {minimo} barras, recibio {len(serie)}"
        )
    if not np.all(np.isfinite(serie)):
        raise ValueError(f"{nombre} tiene valores no finitos")
    return serie


def _wilder(valores: FloatArray, period: int) -> float:
    """Suavizado de Wilder: media simple del primer tramo y luego recursion.

    Es la definicion original del RSI y del ATR. Un EMA estandar da numeros
    parecidos pero distintos, y mezclarlos hace que dos corridas del mismo
    experimento con librerias distintas no sean comparables.
    """
    suave = float(np.mean(valores[:period]))
    for x in valores[period:]:
        suave = (suave * (period - 1) + float(x)) / period
    return suave


def rsi(closes: FloatArray, period: int = 14) -> float:
    """RSI de Wilder en la ultima barra, en ``[0, 100]``.

    Necesita ``period + 1`` cierres para tener ``period`` variaciones.

    Casos degenerados explicitos en vez de ``nan``: sin perdidas devuelve 100,
    sin ganancias devuelve 0, y una serie plana devuelve 50 (ni sobrecomprada
    ni sobrevendida, que es lo que una serie plana es).
    """
    if period < 1:
        raise ValueError("period debe ser positivo")
    serie = _validar(closes, period + 1, "closes")
    cambios = np.diff(serie)
    ganancias = np.maximum(cambios, 0.0)
    perdidas = -np.minimum(cambios, 0.0)

    media_ganancia = _wilder(ganancias, period)
    media_perdida = _wilder(perdidas, period)
    if media_perdida == 0.0:
        return 100.0 if media_ganancia > 0.0 else 50.0
    if media_ganancia == 0.0:
        return 0.0
    rs = media_ganancia / media_perdida
    return float(100.0 - 100.0 / (1.0 + rs))


def _ema(valores: FloatArray, period: int) -> FloatArray:
    """EMA estandar, sembrada con el primer valor de la serie."""
    alpha = 2.0 / (period + 1.0)
    salida = np.empty_like(valores)
    salida[0] = valores[0]
    for i in range(1, len(valores)):
        salida[i] = alpha * valores[i] + (1.0 - alpha) * salida[i - 1]
    return salida


def macd(
    closes: FloatArray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """``(macd, signal, histograma)`` en la ultima barra.

    Se devuelven los tres y no solo el histograma: el nivel y el cruce son
    senales distintas, y agregarlas en una sola cifra le quita al agente la
    posibilidad de distinguirlas.
    """
    if not 0 < fast < slow:
        raise ValueError("fast debe ser positivo y menor que slow")
    if signal < 1:
        raise ValueError("signal debe ser positivo")
    serie = _validar(closes, slow, "closes")
    linea = _ema(serie, fast) - _ema(serie, slow)
    linea_signal = _ema(linea, signal)
    return (
        float(linea[-1]),
        float(linea_signal[-1]),
        float(linea[-1] - linea_signal[-1]),
    )


def atr(
    high: FloatArray, low: FloatArray, close: FloatArray, period: int = 14
) -> float:
    """Average True Range de Wilder en la ultima barra, en unidades de precio.

    El rango verdadero incluye el gap contra el cierre anterior, no solo el
    rango de la barra. En este proyecto eso importa: el gap entre ``close[t]`` y
    ``open[t+1]`` es un termino de primer nivel del desglose de costos.
    """
    if period < 1:
        raise ValueError("period debe ser positivo")
    h = _validar(high, period + 1, "high")
    lo = _validar(low, period + 1, "low")
    c = _validar(close, period + 1, "close")
    if not len(h) == len(lo) == len(c):
        raise ValueError("high, low y close deben tener la misma longitud")

    rango_barra = h[1:] - lo[1:]
    gap_arriba = np.abs(h[1:] - c[:-1])
    gap_abajo = np.abs(lo[1:] - c[:-1])
    rango_verdadero = np.maximum(rango_barra, np.maximum(gap_arriba, gap_abajo))
    return _wilder(rango_verdadero, period)


def bollinger(
    closes: FloatArray, period: int = 20, k: float = 2.0
) -> tuple[float, float]:
    """``(%B, ancho relativo)`` en la ultima barra.

    ``%B`` ubica el precio dentro de las bandas: 0 en la inferior, 1 en la
    superior, fuera de ``[0, 1]`` cuando el precio las perfora. El ancho va
    dividido por la media, asi que es adimensional y comparable entre
    instrumentos y entre regimenes de riesgo.

    Con desviacion cero (serie plana en la ventana) ``%B`` es 0.5 y el ancho 0:
    el precio esta exactamente en la media y no hay banda que perforar.
    """
    if period < 2:
        raise ValueError("period debe ser al menos 2")
    if k <= 0:
        raise ValueError("k debe ser positivo")
    serie = _validar(closes, period, "closes")
    ventana = serie[-period:]
    media = float(np.mean(ventana))
    desvio = float(np.std(ventana, ddof=0))
    if desvio == 0.0 or media == 0.0:
        return 0.5, 0.0
    inferior = media - k * desvio
    ancho_banda = 2.0 * k * desvio
    pct_b = (float(serie[-1]) - inferior) / ancho_banda
    return pct_b, ancho_banda / media


def normalized_volume(volumes: FloatArray, period: int = 20) -> float:
    """Logaritmo del volumen sobre su media movil. Cero significa "lo normal".

    Se usa el logaritmo porque el volumen tiene cola derecha larga: un dia de
    cinco veces el volumen normal y uno de un quinto deben pesar lo mismo en
    magnitud y en signo opuesto, y el cociente crudo no hace eso.
    """
    if period < 1:
        raise ValueError("period debe ser positivo")
    serie = _validar(volumes, period, "volumes")
    ventana = serie[-period:]
    media = float(np.mean(ventana))
    actual = float(serie[-1])
    if media <= 0.0 or actual <= 0.0:
        # Una barra sin volumen no se opera; el simulador la rechaza con
        # NO_VOLUME. Devolver 0.0 evita un -inf que contaminaria la observacion.
        return 0.0
    return float(np.log(actual / media))
