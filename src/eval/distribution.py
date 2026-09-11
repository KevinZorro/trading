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
from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# Varianza de mercado contra varianza de entrenamiento
#
# El hallazgo que obligo a escribir esto: sobre un fixture sintetico, diez
# semillas de entrenamiento sobre **un** camino no miden lo que uno cree que
# miden. Miden cuanto varia el resultado al reentrenar sobre la misma serie, que
# es una pregunta distinta de cuanto varia al cambiar de mercado.
#
# Medido en el nivel 4 del protocolo: el exceso del agente sobre estar invertido
# fue +0.164 con las diez semillas de acuerdo entre si, sobre un camino donde
# estar invertido rindio -0.013. El desvio de ese mismo baseline **entre 20
# caminos de Heston independientes** es 0.781. El "hallazgo" era cinco veces mas
# chico que la dispersion del sorteo, y ninguna cantidad de semillas de
# entrenamiento lo habria revelado: hay que muestrear caminos.
#
# En datos reales los caminos no se pueden muestrear -la historia es uno solo- y
# esa asimetria es la razon de fondo por la que el walk-forward fuera de muestra
# es la unica defensa que queda. No es un detalle metodologico: es la diferencia
# entre poder medir el error y solo poder acotarlo.
# ---------------------------------------------------------------------------

# Valores criticos de t de dos colas al 95%, por grados de libertad. Tabla y no
# una aproximacion normal porque con N=10 caminos la diferencia entre 2.262 y
# 1.96 decide si un exceso se declara significativo o no.
_T_CRITICO_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_T_CRITICO_INFINITO = 1.960


def t_critical_95(df: int) -> float:
    """Valor critico de t de dos colas al 95% con ``df`` grados de libertad."""
    if df < 1:
        raise DistributionError("hacen falta al menos 2 observaciones (df >= 1)")
    return _T_CRITICO_95.get(df, _T_CRITICO_INFINITO)


def drift_t_statistic(log_returns: FloatArray) -> float:
    """``t`` del drift medio: ``media / (desvio / sqrt(n))``.

    Es la pregunta "¿este drift es detectable en esta muestra?", que hay que
    contestar **antes** de exigirle a un agente que lo aprenda. Sobre el nivel 4
    del protocolo da ~1.8 con 4800 barras de entrenamiento: por debajo de
    cualquier umbral de significancia, asi que "el agente debe converger a estar
    invertido" le pide aprender algo que la muestra no contiene.

    Se reporta junto al resultado justamente para que el criterio no se pueda
    leer sin ese contexto.
    """
    serie = np.asarray(log_returns, dtype=np.float64)
    if serie.size < 2:
        raise DistributionError("hacen falta al menos 2 retornos")
    desvio = float(serie.std(ddof=1))
    if desvio <= 0.0:
        raise DistributionError("dispersion nula: el estadistico no esta definido")
    return float(serie.mean()) / (desvio / math.sqrt(serie.size))


@dataclass(frozen=True)
class VarianceDecomposition:
    """Una metrica sobre ``N`` caminos x ``M`` semillas, con las dos varianzas.

    - ``between_path_std``: desvio de las medianas por camino. Es la **varianza
      de mercado**: cuanto cambia el resultado por haber tocado otro camino.
    - ``within_path_std``: mediana de los desvios entre semillas dentro de cada
      camino. Es la **varianza de entrenamiento**: cuanto cambia el resultado
      por reentrenar sobre la misma serie.

    ``t_statistic`` contrasta la media entre caminos contra cero usando el error
    estandar **entre caminos**, que es el unico denominador honesto: usar el de
    las semillas trataria M x N observaciones como independientes cuando en
    realidad hay N.
    """

    metric: str
    n_paths: int
    n_seeds_per_path: int
    path_medians: tuple[float, ...]
    mean: float
    between_path_std: float
    within_path_std: float | None
    standard_error: float
    t_statistic: float | None
    t_critical: float
    distinguishable_from_zero: bool

    @property
    def variance_ratio(self) -> float | None:
        """Cuantas veces la varianza de mercado supera a la de entrenamiento.

        Un cociente grande dice que reportar solo semillas -que es lo que hacia
        el protocolo antes de este cambio- describe la fuente de variacion
        equivocada.
        """
        if self.within_path_std is None or self.within_path_std <= 0.0:
            return None
        return self.between_path_std / self.within_path_std

    def describe(self) -> dict[str, object]:
        salida = dict(vars(self))
        salida["path_medians"] = list(self.path_medians)
        salida["variance_ratio"] = self.variance_ratio
        return salida

    def render(self) -> str:
        t = "n/a" if self.t_statistic is None else f"{self.t_statistic:+.2f}"
        dentro = self.within_path_std
        entrenamiento = "n/a" if dentro is None else f"{dentro:.4f}"
        razon = self.variance_ratio
        veredicto = (
            "DISTINGUIBLE de cero"
            if self.distinguishable_from_zero
            else "NO distinguible de cero"
        )
        return (
            f"{self.metric}: media entre caminos {self.mean:+.4f}  "
            f"sigma_mercado {self.between_path_std:.4f}  "
            f"sigma_entrenamiento {entrenamiento}"
            + (f" (x{razon:.1f})" if razon is not None else "")
            + f"  t={t} contra {self.t_critical:.3f}  {veredicto}"
            + f"  (N={self.n_paths} caminos x M={self.n_seeds_per_path} semillas)"
        )


def decompose_variance(
    metric: str,
    per_path: Sequence[Sequence[float | None]],
    *,
    min_paths: int = 2,
) -> VarianceDecomposition:
    """Descompone una metrica medida sobre varios caminos y varias semillas.

    ``per_path[i]`` son los valores de las ``M`` semillas sobre el camino ``i``.
    Cada camino aporta **una** observacion -su mediana- porque las semillas de un
    mismo camino no son independientes entre si: comparten la serie.
    """
    if len(per_path) < min_paths:
        raise DistributionError(
            f"hacen falta al menos {min_paths} caminos para separar la varianza "
            f"de mercado de la de entrenamiento; se pasaron {len(per_path)}. Con "
            "un solo camino las dos son indistinguibles, que es exactamente el "
            "problema que esta descomposicion existe para evitar."
        )
    medianas: list[float] = []
    desvios: list[float] = []
    for valores in per_path:
        validos = np.asarray([v for v in valores if v is not None], dtype=np.float64)
        if validos.size == 0:
            raise DistributionError(
                f"un camino no tiene ni un valor valido de {metric!r}"
            )
        medianas.append(float(np.median(validos)))
        if validos.size >= 2:
            desvios.append(float(validos.std(ddof=1)))

    arreglo = np.asarray(medianas, dtype=np.float64)
    n = arreglo.size
    entre = float(arreglo.std(ddof=1))
    dentro = float(np.median(desvios)) if desvios else None
    error = entre / math.sqrt(n)
    media = float(arreglo.mean())
    critico = t_critical_95(n - 1)
    t = media / error if error > 0 else None
    return VarianceDecomposition(
        metric=metric,
        n_paths=n,
        n_seeds_per_path=len(per_path[0]),
        path_medians=tuple(medianas),
        mean=media,
        between_path_std=entre,
        within_path_std=dentro,
        standard_error=error,
        t_statistic=t,
        t_critical=critico,
        distinguishable_from_zero=bool(t is not None and abs(t) > critico),
    )


@dataclass(frozen=True)
class PairedDifference:
    """Diferencia pareada de una metrica entre dos conjuntos de caminos.

    Existe porque dos fixtures que solo difieren en un parametro **comparten la
    realizacion del ruido**: ``generate_gbm_sv`` consume los mismos shocks para
    la misma semilla, y cambiar ``mu`` solo mueve el termino deterministico del
    drift. Medido: la diferencia de log-retornos entre el nivel 4a y el 4b con
    la misma semilla es constante a 1e-15 y la correlacion entre los dos caminos
    es exactamente 1.

    Eso convierte la comparacion en un contraste **pareado**, que no es un
    refinamiento estadistico sino la unica forma de aislar el parametro: la
    varianza de mercado -que es la grande- se cancela dentro de cada par, y lo
    que queda es el efecto del cambio.

    Un contraste no pareado sobre los mismos numeros tendria que atravesar una
    dispersion entre caminos que aqui es irrelevante, y podria no detectar un
    efecto grande solo porque los caminos son ruidosos.
    """

    metric: str
    label_a: str
    label_b: str
    n_pairs: int
    differences: tuple[float, ...]
    mean_a: float
    mean_b: float
    mean: float
    std: float
    standard_error: float
    t_statistic: float | None
    t_critical: float
    distinguishable_from_zero: bool

    def describe(self) -> dict[str, object]:
        salida = dict(vars(self))
        salida["differences"] = list(self.differences)
        return salida

    def render(self) -> str:
        t = (
            "deterministica"
            if self.t_statistic is None and self.distinguishable_from_zero
            else "n/a"
            if self.t_statistic is None
            else f"{self.t_statistic:+.2f}"
        )
        veredicto = (
            "DISTINGUIBLE de cero"
            if self.distinguishable_from_zero
            else "NO distinguible de cero"
        )
        return (
            f"{self.metric}: {self.label_b} {self.mean_b:+.4f} contra "
            f"{self.label_a} {self.mean_a:+.4f}  ->  diferencia pareada "
            f"{self.mean:+.4f} (sigma {self.std:.4f}, N={self.n_pairs} pares)  "
            f"t={t} contra {self.t_critical:.3f}  {veredicto}"
        )


def paired_difference(
    a: VarianceDecomposition,
    b: VarianceDecomposition,
    *,
    label_a: str,
    label_b: str,
) -> PairedDifference:
    """Contrasta ``b - a`` camino por camino, no promedio contra promedio.

    Exige el mismo numero de caminos y la misma metrica: parear medianas de
    caminos que no se corresponden entre si produciria un numero sin
    interpretacion.
    """
    if a.metric != b.metric:
        raise DistributionError(
            f"no se pueden parear metricas distintas: {a.metric!r} y {b.metric!r}"
        )
    if a.n_paths != b.n_paths:
        raise DistributionError(
            f"{a.n_paths} caminos contra {b.n_paths}: el pareo exige "
            "correspondencia uno a uno"
        )
    if a.n_paths < 2:
        raise DistributionError("hacen falta al menos 2 pares")

    diferencias = np.asarray(b.path_medians, dtype=np.float64) - np.asarray(
        a.path_medians, dtype=np.float64
    )
    n = diferencias.size
    media = float(diferencias.mean())
    desvio = float(diferencias.std(ddof=1))
    error = desvio / math.sqrt(n)
    critico = t_critical_95(n - 1)

    # Diferencia constante entre pares: la dispersion es nula y el cociente no
    # esta definido. **Eso no la vuelve indistinguible de cero, la vuelve
    # deterministica**: si todos los pares se mueven exactamente lo mismo y ese
    # movimiento no es cero, el efecto esta ahi sin ruido que lo discuta.
    # Devolver "no distinguible" ahi seria reportar la conclusion opuesta a la
    # que los datos sostienen. Mismo criterio de tolerancia relativa que el
    # resto del proyecto para "dispersion nula".
    if desvio <= abs(media) * DISPERSION_NULA_REL:
        t = None
        distinguible = abs(media) > 0.0
    else:
        t = media / error if error > 0 else None
        distinguible = bool(t is not None and abs(t) > critico)
    return PairedDifference(
        metric=a.metric,
        label_a=label_a,
        label_b=label_b,
        n_pairs=n,
        differences=tuple(float(x) for x in diferencias),
        mean_a=a.mean,
        mean_b=b.mean,
        mean=media,
        std=desvio,
        standard_error=error,
        t_statistic=t,
        t_critical=critico,
        distinguishable_from_zero=distinguible,
    )
