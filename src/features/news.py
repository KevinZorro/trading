"""Features de noticias del Agente B por barra diaria (Etapa 4, PR N4; ADR 0012).

Alinea los agregados por lote de 15 min (``data.news_aggregate``) con las barras y
calcula las siete columnas de la seccion 3 del ADR. Es el unico lugar donde un lote
se asigna a una barra.

Regla anti-lookahead (``CLAUDE.md``, convencion temporal; ADR 0012, 2.3): un lote es
de la barra ``t`` si y solo si ``close[t-1] < rotulo + latencia <= close[t]``. La
latencia se suma al rotulo, nunca se resta al cierre. Cada fila ``t`` usa lotes de
esa ventana y estadisticas de barras **anteriores**; nada posterior a ``close[t]``.

Las columnas enmascaradas valen ``NaN``. Este modulo no escala ni imputa: el
escalador ajustado en train y el "enmascarado entra como 0" son de N5 (ADR 0012, 3).
Nunca ``ffill``.

ASSUMPTION: las barras son diarias y cierran a las 00:00 UTC (``Binance 1d``). El
umbral de ``coverage_ok`` (87 de 96 lotes) esta definido para 24 h; otra duracion
de barra necesita su propio ADR.
ASSUMPTION: "las 30 barras previas validas" (``news_share_z30``) y "las 7 previas
validas" (``tone_delta7``) son las 30 y 7 barras calendario anteriores, y la mascara
cuenta cuantas de ellas tienen valor (>= 20 y >= 4). El ADR no distingue entre esa
lectura y "las ultimas 30 con valor"; se eligio la que no depende de cuan atras haya
que ir a buscar.
ASSUMPTION: ``tone_mean``, ``no_title`` y ``finbert_mean`` usan las filas de todos los
lotes ``ok`` de la barra (el ADR dice "las filas del tema en la barra"); solo
``news_share`` se restringe a lotes sanos (ADR 0012, 3, fila 1). ``coverage_ok``
informa cuando la barra es confiable.
ASSUMPTION: ``finbert_mean`` es ``sum(score) / sum(n)`` sobre los lotes ``ok`` de la
barra, con los puntajes de ``features.finbert_dataset.attach_finbert`` (``finbert_n``,
``finbert_sum``). Sin esas columnas, o sin articulos puntuados en la barra, vale
``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.errors import DataValidationError
from data.news_aggregate import COVERAGE_MIN_BATCHES, healthy_batches
from data.schema import FloatArray, TimeArray
from features.scaler import DESVIO_MINIMO

# ADR 0012, 2.2 (K2, Kevin 2026-10-25). Explicita: DEFAULT_LATENCY de data.news es 30.
NEWS_LATENCY = np.timedelta64(60, "m")
# ADR 0012, 4: 52 semanas exactas conservan el dia de la semana.
PLACEBO_LAG = np.timedelta64(364, "D")
NEWS_FEATURE_NAMES: tuple[str, ...] = (
    "news_share",
    "news_share_z30",
    "tone_mean",
    "tone_delta7",
    "finbert_mean",
    "no_title",
    "coverage_ok",
)
SCALED_FEATURES = 5  # las columnas 1 a 5; 6 y 7 son indicadores sin escalar
Z_WINDOW, Z_MIN_VALID = 30, 20
DELTA_WINDOW, DELTA_MIN_VALID = 7, 4
_DIA = np.timedelta64(1, "D")
_SIN_LAG = np.timedelta64(0, "D")
_COL = {nombre: i for i, nombre in enumerate(NEWS_FEATURE_NAMES)}


@dataclass(frozen=True)
class NewsFeatureTable:
    """Las siete columnas para cada cierre diario cubierto por los datos.

    ``values`` es ``(n, 7)`` en el orden de ``NEWS_FEATURE_NAMES``; ``NaN`` = masked.
    """

    closes: TimeArray
    values: FloatArray = field(repr=False)
    latency: np.timedelta64 = NEWS_LATENCY

    def at(self, closes: TimeArray, lag: np.timedelta64 = _SIN_LAG) -> FloatArray:
        """Filas para los cierres pedidos, mirando ``lag`` hacia atras en el tiempo.

        ``lag = PLACEBO_LAG`` da el bloque del brazo placebo: el de ``close - 364 d``.
        El lag no puede ser negativo (seria mirar el futuro), y un cierre sin fila
        (fuera del rango de datos o fuera de la rejilla diaria) falla en vez de
        rellenarse.
        """
        if lag < np.timedelta64(0, "D"):
            raise DataValidationError(f"el lag no puede ser negativo ({lag})")
        pedidos = np.asarray(closes, dtype="datetime64[ns]") - lag
        filas = np.searchsorted(self.closes, pedidos)
        fuera = (filas >= len(self.closes)) | (
            self.closes[np.minimum(filas, len(self.closes) - 1)] != pedidos
        )
        if fuera.any():
            malo = pedidos[int(np.argmax(fuera))]
            raise DataValidationError(
                f"no hay features de noticias para el cierre {malo}"
            )
        return np.asarray(self.values[filas], dtype=np.float64)


def build_news_features(
    frame: pd.DataFrame, latency: np.timedelta64 = NEWS_LATENCY
) -> NewsFeatureTable:
    """Features de todos los dias que tocan los datos (``frame``: read_aggregates)."""
    if latency < np.timedelta64(0, "s"):
        raise DataValidationError(f"la latencia no puede ser negativa ({latency})")
    sano = healthy_batches(frame)  # valida la rejilla de 15 min completa
    disponible = (frame["label"] + pd.Timedelta(latency)).to_numpy("datetime64[ns]")
    primero, ultimo = _cierre_de(disponible[0]), _cierre_de(disponible[-1])
    closes = np.arange(primero, ultimo + _DIA, _DIA).astype("datetime64[ns]")
    barra = np.searchsorted(closes, disponible, side="left")  # primer cierre >= disp.

    ok = (frame["status"] == "ok").to_numpy()
    n = len(closes)

    def suma(valor: FloatArray, mascara: npt.NDArray[np.bool_]) -> FloatArray:
        pesos = np.bincount(barra[mascara], weights=valor[mascara], minlength=n)
        return np.asarray(pesos, dtype=np.float64)

    theme = frame["rows_theme"].fillna(0).to_numpy(np.float64)
    total = frame["rows_total"].fillna(0).to_numpy(np.float64)
    tono = frame["tone_sum"].fillna(0).to_numpy(np.float64)
    titulos = frame["n_titled"].fillna(0).to_numpy(np.float64)

    sanos = suma(np.ones(len(frame)), sano)
    theme_sano, total_sano = suma(theme, sano), suma(total, sano)
    theme_ok, tono_ok, titulos_ok = suma(theme, ok), suma(tono, ok), suma(titulos, ok)

    vals = np.full((n, len(NEWS_FEATURE_NAMES)), np.nan)
    con_sanos = total_sano > 0
    share = np.full(n, np.nan)
    share[con_sanos] = np.log1p(1e4 * theme_sano[con_sanos] / total_sano[con_sanos])
    tone = np.full(n, np.nan)
    tone[theme_ok > 0] = tono_ok[theme_ok > 0] / theme_ok[theme_ok > 0]

    finbert = np.full(n, np.nan)
    if "finbert_n" in frame.columns:
        fin_n = suma(frame["finbert_n"].fillna(0).to_numpy(np.float64), ok)
        fin_suma = suma(frame["finbert_sum"].fillna(0).to_numpy(np.float64), ok)
        finbert[fin_n > 0] = fin_suma[fin_n > 0] / fin_n[fin_n > 0]

    vals[:, _COL["news_share"]] = share
    vals[:, _COL["news_share_z30"]] = _zscore(share)
    vals[:, _COL["tone_mean"]] = tone
    vals[:, _COL["tone_delta7"]] = _delta(tone)
    vals[:, _COL["finbert_mean"]] = finbert
    vals[:, _COL["no_title"]] = (titulos_ok == 0).astype(np.float64)
    vals[:, _COL["coverage_ok"]] = (sanos >= COVERAGE_MIN_BATCHES).astype(np.float64)
    return NewsFeatureTable(closes, vals, latency)


def _cierre_de(instante: np.datetime64) -> np.datetime64:
    """Primer cierre diario (00:00 UTC) ``>=`` al instante: el de su barra."""
    dia = (instante - np.timedelta64(1, "ns")).astype("datetime64[D]") + _DIA
    return np.datetime64(dia)


def _zscore(share: FloatArray) -> FloatArray:
    previas = pd.Series(share).shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_VALID)
    media = previas.mean().to_numpy(np.float64)
    desvio = previas.std(ddof=0).to_numpy(np.float64)
    z = np.full(len(share), np.nan)
    util = np.isfinite(share) & np.isfinite(media) & (desvio > DESVIO_MINIMO)
    z[util] = (share[util] - media[util]) / desvio[util]
    return z


def _delta(tone: FloatArray) -> FloatArray:
    previas = (
        pd.Series(tone).shift(1).rolling(DELTA_WINDOW, min_periods=DELTA_MIN_VALID)
    )
    media = previas.mean().to_numpy(np.float64)
    return np.asarray(tone - media, dtype=np.float64)
