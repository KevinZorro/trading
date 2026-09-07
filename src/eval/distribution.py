"""Reporte de una metrica sobre multiples semillas, y significancia.

El principio 5 del proyecto -"nunca reportar el mejor seed"- no se sostiene con
disciplina. Un barrido de 10 semillas produce 10 numeros y la tentacion de
mirar el mejor es estructural: es el que confirma la hipotesis. Aca la regla es
mecanica: :func:`summarize` **falla** si recibe menos de ``MIN_SEEDS`` valores,
y la unica forma de saltearla deja una marca que viaja dentro del resultado
serializado.

Sobre significancia: comparar 10 semillas contra un baseline y quedarse con el
p-value del mejor es el ejemplo de manual de comparaciones multiples. El
proyecto usa el **Deflated Sharpe Ratio** (Bailey y Lopez de Prado, 2014), que
descuenta explicitamente cuantas configuraciones se probaron: sin ese descuento,
probar 10 semillas de ruido puro produce un Sharpe "significativo" alrededor de
una vez de cada dos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from data.schema import FloatArray
from eval.metrics import DISPERSION_NULA_REL, MetricError

# Minimo de semillas para reportar una distribucion. El numero sale del
# CLAUDE.md y esta aca para que sea codigo, no una nota al pie.
MIN_SEEDS = 10

# Constante de Euler-Mascheroni, en la aproximacion del maximo esperado de N
# Sharpes independientes.
_GAMMA = 0.5772156649015329


class DistributionError(MetricError):
    """La distribucion no se puede resumir con lo que se paso."""


@dataclass(frozen=True)
class SeedDistribution:
    """Una metrica sobre varias semillas. Nunca un escalar.

    ``values`` conserva el orden de ``seeds`` para que un resultado se pueda
    rastrear hasta la semilla que lo produjo; ``None`` es un valor legitimo -una
    metrica indefinida, como ``win_rate`` sin trades cerrados- y se cuenta
    aparte en vez de convertirse en cero.
    """

    metric: str
    seeds: tuple[int, ...]
    values: tuple[float | None, ...]
    median: float | None
    p25: float | None
    p75: float | None
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    n_valid: int
    n_missing: int
    below_minimum_seeds: bool = False

    @property
    def iqr(self) -> float | None:
        if self.p25 is None or self.p75 is None:
            return None
        return self.p75 - self.p25

    def fraction_above(self, threshold: float) -> float | None:
        """Fraccion de semillas por encima de un umbral.

        Es la forma honesta de comparar contra un baseline deterministico: "7 de
        10 semillas superan a buy-and-hold" dice algo que la mediana sola no
        dice, sobre todo cuando la dispersion entre semillas es grande.
        """
        validos = [v for v in self.values if v is not None]
        if not validos:
            return None
        return sum(1.0 for v in validos if v > threshold) / len(validos)

    def describe(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "seeds": list(self.seeds),
            "values": list(self.values),
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "n_valid": self.n_valid,
            "n_missing": self.n_missing,
            "below_minimum_seeds": self.below_minimum_seeds,
        }

    def render(self) -> str:
        def n(v: float | None) -> str:
            return "n/a" if v is None else f"{v:+.4f}"

        marca = "  [MENOS DE 10 SEMILLAS]" if self.below_minimum_seeds else ""
        return (
            f"{self.metric}: mediana {n(self.median)}  "
            f"p25 {n(self.p25)}  p75 {n(self.p75)}  "
            f"min {n(self.minimum)}  max {n(self.maximum)}  "
            f"(n={self.n_valid}"
            + (f", {self.n_missing} indefinidas" if self.n_missing else "")
            + ")"
            + marca
        )


def summarize(
    metric: str,
    seeds: list[int] | tuple[int, ...],
    values: list[float | None] | tuple[float | None, ...],
    *,
    allow_fewer_seeds: bool = False,
) -> SeedDistribution:
    """Resume una metrica sobre semillas. Falla con menos de ``MIN_SEEDS``.

    ``allow_fewer_seeds`` existe para los tests y para las exploraciones
    rapidas, pero **no es gratis**: enciende ``below_minimum_seeds``, que viaja
    dentro de ``describe()`` y sale impreso en ``render()``. Un resultado con
    pocas semillas se puede producir; lo que no se puede es que parezca uno con
    muchas.

    Los percentiles usan interpolacion lineal (el metodo por defecto de
    ``numpy``). Queda dicho porque con 10 observaciones el metodo cambia el p25
    de forma visible, y dos reportes con metodos distintos no son comparables.
    """
    if len(seeds) != len(values):
        raise DistributionError(
            f"{len(seeds)} semillas y {len(values)} valores: no se pueden parear"
        )
    if len(set(seeds)) != len(seeds):
        raise DistributionError("hay semillas repetidas: el barrido no es lo que dice")
    pocas = len(values) < MIN_SEEDS
    if pocas and not allow_fewer_seeds:
        raise DistributionError(
            f"se pasaron {len(values)} semillas y el minimo es {MIN_SEEDS}. Toda "
            "metrica del estudio se reporta como distribucion; con menos "
            "semillas la mediana es indistinguible del mejor run. Si es una "
            "exploracion, pasa allow_fewer_seeds=True y quedara marcado."
        )

    validos = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if validos.size and not np.all(np.isfinite(validos)):
        raise DistributionError(
            f"la metrica {metric!r} tiene valores no finitos: un inf o un nan "
            "domina cualquier percentil sin significar nada"
        )
    if validos.size == 0:
        estadisticas: tuple[float | None, ...] = (None,) * 6
    else:
        p25, mediana, p75 = (float(x) for x in np.percentile(validos, [25, 50, 75]))
        estadisticas = (
            mediana,
            p25,
            p75,
            float(validos.min()),
            float(validos.max()),
            float(validos.mean()),
        )
    desvio = float(validos.std(ddof=1)) if validos.size >= 2 else None

    return SeedDistribution(
        metric=metric,
        seeds=tuple(seeds),
        values=tuple(values),
        median=estadisticas[0],
        p25=estadisticas[1],
        p75=estadisticas[2],
        minimum=estadisticas[3],
        maximum=estadisticas[4],
        mean=estadisticas[5],
        std=desvio,
        n_valid=int(validos.size),
        n_missing=sum(1 for v in values if v is None),
        below_minimum_seeds=pocas,
    )


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Sharpe maximo esperado de ``n_trials`` intentos bajo la hipotesis nula.

    Aproximacion de Bailey y Lopez de Prado a partir del maximo de normales
    independientes::

        E[max SR] = sqrt(V) * [ (1-gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N*e)) ]

    Es el numero contra el que hay que comparar, no cero. Probar 10 semillas y
    quedarse con la mejor produce un Sharpe positivo aunque no haya senal: este
    es exactamente cuanto.

    ``sharpe_variance`` es la varianza de los Sharpe **entre** intentos, que es
    lo que mide la dispersion del barrido de semillas.
    """
    if n_trials < 1:
        raise DistributionError("n_trials debe ser >= 1")
    if sharpe_variance < 0:
        raise DistributionError("sharpe_variance no puede ser negativa")
    if n_trials == 1 or sharpe_variance <= DISPERSION_NULA_REL:
        return 0.0
    normal = NormalDist()
    z1 = normal.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probabilidad de que el Sharpe verdadero supere a ``benchmark_sharpe``.

    Corrige por no normalidad: los retornos con asimetria negativa y colas
    pesadas -que son los de cualquier estrategia con riesgo de cola- hacen que
    el estimador del Sharpe sea mas ruidoso de lo que la formula gaussiana
    sugiere, y sin la correccion el mismo numero parece mas significativo de lo
    que es.

    ``kurtosis`` es la curtosis **no** en exceso (3 para una normal).
    """
    if n_obs < 2:
        raise DistributionError("hacen falta al menos 2 observaciones")
    varianza = (
        1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )
    if varianza <= 0:
        raise DistributionError(
            "la varianza del estimador del Sharpe sale no positiva: con esa "
            "asimetria y curtosis el estadistico no esta definido"
        )
    z = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(n_obs - 1)
        / math.sqrt(varianza)
    )
    return float(NormalDist().cdf(z))


@dataclass(frozen=True)
class DeflatedSharpe:
    """Resultado del Deflated Sharpe Ratio, con todo lo que lo produjo."""

    value: float
    observed_sharpe: float
    benchmark_sharpe: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float
    significant: bool

    def describe(self) -> dict[str, object]:
        return vars(self).copy()


def deflated_sharpe(
    returns: FloatArray,
    *,
    n_trials: int,
    sharpe_variance: float,
    alpha: float = 0.05,
) -> DeflatedSharpe:
    """DSR de una serie de retornos, descontando ``n_trials`` configuraciones.

    El Sharpe es **por periodo, sin anualizar**: la formula esta definida sobre
    el estadistico muestral y anualizarlo antes de deflactarlo mezcla dos
    escalas. Para reportar, el Sharpe anualizado sale de ``eval.metrics.sharpe``.

    ``n_trials`` tiene que ser el numero **real** de configuraciones probadas,
    incluidas las que se descartaron. Poner 1 despues de haber barrido 40
    hiperparametros es la forma mas comun de fabricar significancia.
    """
    serie = np.asarray(returns, dtype=np.float64)
    if serie.size < 2:
        raise DistributionError("hacen falta al menos 2 retornos")
    if not np.all(np.isfinite(serie)):
        raise DistributionError("la serie de retornos tiene valores no finitos")
    media = float(serie.mean())
    desvio = float(serie.std(ddof=1))
    if desvio <= abs(media) * DISPERSION_NULA_REL or desvio == 0.0:
        raise DistributionError(
            "dispersion nula: el Sharpe no esta definido. Mismo criterio que "
            "eval.metrics.sharpe"
        )
    sr = media / desvio
    centrado = (serie - media) / desvio
    asimetria = float((centrado**3).mean())
    curtosis = float((centrado**4).mean())
    referencia = expected_max_sharpe(n_trials, sharpe_variance)
    valor = probabilistic_sharpe_ratio(
        sr,
        benchmark_sharpe=referencia,
        n_obs=serie.size,
        skew=asimetria,
        kurtosis=curtosis,
    )
    return DeflatedSharpe(
        value=valor,
        observed_sharpe=sr,
        benchmark_sharpe=referencia,
        n_trials=n_trials,
        n_obs=int(serie.size),
        skew=asimetria,
        kurtosis=curtosis,
        significant=valor > 1.0 - alpha,
    )
