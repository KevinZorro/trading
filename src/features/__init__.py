"""Features: indicadores tecnicos y transformadores ajustados en train.

Separado de ``envs/`` a proposito. Calcular un indicador no es tarea del entorno
Gymnasium; el entorno ensambla la observacion, no la computa. Cuando la Etapa 4
agregue features de noticias, entraran aca y el entorno no tendra que cambiar.
"""

from features.scaler import DESVIO_MINIMO, FeatureScaler, ScalerError
from features.technical import (
    ATR_WARMUP,
    BOLLINGER_WARMUP,
    MACD_WARMUP,
    RSI_WARMUP,
    VOLUME_WARMUP,
    atr,
    bollinger,
    macd,
    normalized_volume,
    rsi,
)

__all__ = [
    "ATR_WARMUP",
    "BOLLINGER_WARMUP",
    "DESVIO_MINIMO",
    "MACD_WARMUP",
    "RSI_WARMUP",
    "VOLUME_WARMUP",
    "FeatureScaler",
    "ScalerError",
    "atr",
    "bollinger",
    "macd",
    "normalized_volume",
    "rsi",
]
