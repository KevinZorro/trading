"""Errores del subsistema de datos.

Todos los problemas de datos son *ruidosos*: nunca se degradan a warning ni se
reparan en silencio. Un dataset roto que se arregla solo es un backtest optimista.
"""

from __future__ import annotations


class DataError(Exception):
    """Raiz de la jerarquia de errores de datos."""


class SchemaError(DataError):
    """Columnas faltantes, dtypes incorrectos o metadatos inconsistentes."""


class DataValidationError(DataError):
    """El contenido de las barras viola una invariante (OHLC, precios, volumen)."""


class CalendarGapError(DataValidationError):
    """Faltan o sobran barras respecto del calendario declarado del mercado."""


class AdjustedPriceError(DataError):
    """Se detectaron precios ajustados retroactivamente en la ruta de ejecucion.

    El ajuste retroactivo por dividendos y splits reescribe el pasado con
    informacion publicada despues; usarlo como precio de ejecucion es lookahead.
    """


class LookaheadError(DataError):
    """Se intento acceder a informacion no disponible en el instante consultado."""
