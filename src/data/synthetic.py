"""Generador de barras sinteticas: GBM con volatilidad estocastica (Heston).

Existe para poder testear el simulador sin datos reales y para construir los
tres regimenes de riesgo del estudio con parametros conocidos. Las barras se
construyen desde un camino intra-barra a sub-pasos, no muestreando OHLC de
forma independiente: eso ultimo produce barras con ``close`` fuera de
``[low, high]`` y haria pasar tests que deberian fallar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.instruments import InstrumentSpec, us_equity_spec
from data.schema import BarSeries


@dataclass(frozen=True)
class HestonParams:
    """Parametros del proceso.

    ``mu`` y las velocidades estan expresadas en unidades anuales; se escalan
    por ``bars_per_year`` al discretizar.
    """

    mu: float = 0.05
    v0: float = 0.04
    kappa: float = 3.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7

    def __post_init__(self) -> None:
        if self.v0 <= 0 or self.theta <= 0:
            raise ValueError("v0 y theta deben ser positivos")
        if self.kappa <= 0 or self.xi <= 0:
            raise ValueError("kappa y xi deben ser positivos")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho fuera de [-1, 1]")


# Regimenes de riesgo del estudio. ASSUMPTION: la volatilidad de largo plazo
# (sqrt(theta)) es la que define el regimen: 12%, 30% y 80% anualizado.
RISK_REGIMES: dict[str, HestonParams] = {
    "low": HestonParams(mu=0.05, v0=0.0144, theta=0.0144, kappa=4.0, xi=0.20, rho=-0.6),
    "medium": HestonParams(mu=0.08, v0=0.09, theta=0.09, kappa=3.0, xi=0.50, rho=-0.7),
    "high": HestonParams(mu=0.15, v0=0.64, theta=0.64, kappa=2.0, xi=1.20, rho=-0.3),
}


def generate_gbm_sv(
    n_bars: int,
    *,
    seed: int,
    instrument: InstrumentSpec | None = None,
    params: HestonParams | str = "medium",
    s0: float = 100.0,
    start: str = "2020-01-01T21:00:00Z",
    freq: str = "1D",
    bars_per_year: float = 252.0,
    sub_steps: int = 24,
    base_volume: float = 1_000_000.0,
    volume_dispersion: float = 0.35,
    volume_vol_beta: float = 2.0,
    overnight_gap_frac: float = 0.0,
    round_to_tick: bool = True,
) -> BarSeries:
    """Genera ``n_bars`` barras OHLCV coherentes.

    - Variancia por esquema de Euler con truncamiento completo (``max(v, 0)``),
      que mantiene el proceso definido sin sesgar la media hacia arriba.
    - OHLC extraidos del camino de ``sub_steps`` puntos dentro de cada barra:
      ``open`` es el primero, ``close`` el ultimo, ``high``/``low`` los extremos.
    - El volumen crece con la volatilidad realizada de la barra
      (``volume_vol_beta``), que es lo que se observa en datos reales y hace no
      trivial el modelo de participacion del simulador.

    - ``overnight_gap_frac`` inyecta un salto entre ``close[t]`` y
      ``open[t+1]``, como fraccion de la volatilidad de la propia barra. Sin el,
      el camino es continuo y ``open[t+1] == close[t]`` exactamente: el
      simulador registraria gap cero siempre y el desplazamiento de valuacion
      por latencia quedaria sin testear, que es justo el efecto que domina en el
      regimen de riesgo alto.

    Sin eventos corporativos: ``split_factor = 1`` y ``cash_dividend = 0``.

    ASSUMPTION: el gap se modela como un shock gaussiano sin correlacion con el
    movimiento intradia y sin colas pesadas. Los gaps reales son leptocurticos y
    se concentran alrededor de anuncios; para la Etapa 4 (noticias) esta
    aproximacion se queda corta y habra que condicionarlos a los eventos.
    """
    if n_bars < 2:
        raise ValueError("se necesitan al menos 2 barras")
    if sub_steps < 1:
        raise ValueError("sub_steps debe ser >= 1")
    if overnight_gap_frac < 0:
        raise ValueError("overnight_gap_frac no puede ser negativo")

    if isinstance(params, str):
        if params not in RISK_REGIMES:
            raise ValueError(
                f"regimen desconocido {params!r}; opciones: {sorted(RISK_REGIMES)}"
            )
        params = RISK_REGIMES[params]

    instrument = instrument or us_equity_spec("SYNTH")
    rng = np.random.default_rng(seed)

    dt = 1.0 / (bars_per_year * sub_steps)
    sqrt_dt = np.sqrt(dt)

    z1 = rng.standard_normal((n_bars, sub_steps))
    z2 = rng.standard_normal((n_bars, sub_steps))
    z_gap = rng.standard_normal(n_bars)
    # Correlacion entre el shock de precio y el de varianza (efecto apalancamiento).
    w_price = z1
    w_var = params.rho * z1 + np.sqrt(1.0 - params.rho**2) * z2

    log_s = np.log(s0)
    v = params.v0

    open_ = np.empty(n_bars)
    high = np.empty(n_bars)
    low = np.empty(n_bars)
    close = np.empty(n_bars)
    realized_vol = np.empty(n_bars)

    path = np.empty(sub_steps + 1)
    for t in range(n_bars):
        if t > 0 and overnight_gap_frac > 0.0:
            # Salto entre el cierre de t-1 y la apertura de t. Escala con la
            # volatilidad vigente, asi que el regimen alto gapea mas.
            bar_sigma = np.sqrt(max(v, 0.0) * dt * sub_steps)
            log_s += overnight_gap_frac * bar_sigma * z_gap[t]
        path[0] = log_s
        for k in range(sub_steps):
            v_pos = max(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            log_s += (params.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w_price[t, k]
            v += params.kappa * (params.theta - v_pos) * dt + (
                params.xi * sqrt_v * sqrt_dt * w_var[t, k]
            )
            path[k + 1] = log_s
        prices = np.exp(path)
        open_[t] = prices[0]
        close[t] = prices[-1]
        high[t] = prices.max()
        low[t] = prices.min()
        realized_vol[t] = np.std(np.diff(path)) * np.sqrt(sub_steps)

    if round_to_tick:
        # El redondeo a tick es monotono, asi que preserva high >= max(open, close)
        # y low <= min(open, close).
        for arr in (open_, high, low, close):
            np.round(arr / instrument.tick_size, out=arr)
            arr *= instrument.tick_size
        floor_price = instrument.tick_size
        for arr in (open_, high, low, close):
            np.maximum(arr, floor_price, out=arr)

    vol_z = realized_vol / max(realized_vol.mean(), 1e-12)
    volume = base_volume * np.exp(
        volume_dispersion * rng.standard_normal(n_bars)
        - 0.5 * volume_dispersion**2
        + volume_vol_beta * (vol_z - 1.0) * 0.1
    )
    volume = np.round(volume)

    timestamps = pd.date_range(start=pd.Timestamp(start), periods=n_bars, freq=freq)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")

    return BarSeries(
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
        source=f"synthetic:gbm_sv:seed={seed}",
    )
