"""Walk-forward del Agente A sobre datos reales. Implementa el ADR 0006, sellado.

Todo lo que decide el resultado esta fijado en ese ADR y no aca: el criterio
principal, la agregacion sin suponer independencia entre folds, la regla de
regimen y la de outliers. Este modulo solo lo ejecuta, y se niega a hacerlo si
el ADR en disco no es el que se sello (``ADR_SHA256``).

El test se toca una vez. ``acquire_seal`` crea el sello de forma exclusiva antes
de evaluar el primer fold. Si ya existe, la corrida aborta, salvo que se pida
``resume`` con exactamente la misma configuracion: en ese caso continua con los
pares (fold, semilla) que faltan, sin volver a evaluar ninguno de los que ya
estan escritos. Cada par se persiste al terminar (append), nunca solo al final.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.policy import Policy
from agents.ppo import PPOConfig
from agents.protocol import STUDY_SEEDS
from agents.runner import build_env, run_baselines, run_policy, scaler_for
from data.calendars import AlwaysOpen
from data.instruments import binance_spot_spec
from data.loaders import load_versioned_bars
from data.manifest import read_manifest
from data.schema import BarSeries, FloatArray
from envs.observation import ObservationSpec
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig, TradingEnv
from eval.distribution import DistributionError, deflated_sharpe, summarize
from eval.walkforward import Window, rolling_windows
from sim.costs import CorwinSchultzSpread, NoSlippage, SqrtSlippage, ZeroSpread
from sim.engine import SimConfig, SimResult
from sim.orders import OrderStatus, RejectReason
from sim.sizing import TargetWeightSizer

ADR_PATH = Path("docs/adr/0006-walk-forward-sobre-btcusdt-diario.md")
# Sello del ADR 0006 (resello e040707; el sello original fcc10ea se reemplazo
# antes de cualquier corrida). Cambiar el ADR, aunque sea una coma,
# obliga a cambiar esta constante, y eso queda en el diff.
ADR_SHA256 = "eea45aa640afda2053e11905c5232352af8cdbe7ab0ff067cf48fc33d411fd72"
# Commit que sello el ADR. Informativo: si el PR se mergea con squash deja de
# estar en la historia de main, y lo que prueba el sello es el SHA256.
ADR_SEAL_COMMIT = "e040707"
DATASET_MANIFEST = Path("datasets/binance/BTCUSDT-1d.csv.manifest.json")

# Constantes del ADR 0006. Viven aca y no en la configuracion porque no son
# parametros: son el criterio sellado.
PISO_T_RECIENTE = 1.0  # seccion 4.1
DSR_N_TRIALS = 10  # seccion 2: eleccion conservadora; el numero honesto es 1
DSR_UMBRAL = 0.95  # seccion 2
BARRAS_SUBSIDIARIA_3 = 50  # seccion 4.3
# Media lenta del cruce de medias (``agents.runner.baseline_strategies``). Los
# baselines necesitan ``MA_LENTA + 1`` barras de historia para su primera
# decision; con solo el calentamiento del agente (26) el cruce de medias se
# perderia sus primeras 24 decisiones de cada test por el recorte, no por la
# estrategia.
MA_LENTA = 50

PRINCIPAL = "principal"
SOLO_COMISION = "solo_comision"

PolicyFactory = Callable[[Callable[[], TradingEnv], int], Policy]


class WalkForwardError(RuntimeError):
    """El walk-forward no se puede correr tal como esta sellado."""


class SealError(WalkForwardError):
    """El ADR no es el sellado, o el test ya se toco."""


@dataclass(frozen=True)
class WalkForwardConfig:
    """Todo lo que no fija el ADR y cambia el resultado. Entra al hash del sello.

    Los valores estan fijados y argumentados en el ADR 0006, seccion 1.1: capital
    de 100 USDT (donde muerde ``min_notional``), comision del instrumento (10 bps),
    Corwin-Schultz con tope de 200 bps, slippage raiz con ``k = 0.1`` (el triple de
    la ley de raiz cuadrada con la volatilidad de BTC) y 365 barras por anio.
    """

    train: int = 800
    validation: int = 200
    test: int = 200
    step: int = 200
    seeds: tuple[int, ...] = STUDY_SEEDS
    total_timesteps: int = 60_000
    reward_scale: float = 100.0
    initial_cash: float = 100.0
    cash_rate: float = 0.0
    bars_per_year: float = 365.0
    safety: float = 0.98
    spread_max_bps: float = 200.0
    slippage_k: float = 0.1
    max_participation: float = 0.10

    def sim_config(self, scenario: str = PRINCIPAL) -> SimConfig:
        """``principal`` decide; ``solo_comision`` es la sensibilidad (seccion 1.1):
        la comision del instrumento sin spread ni slippage."""
        if scenario == PRINCIPAL:
            spread: Any = CorwinSchultzSpread(max_bps=self.spread_max_bps)
            slippage: Any = SqrtSlippage(k=self.slippage_k)
        elif scenario == SOLO_COMISION:
            spread, slippage = ZeroSpread(), NoSlippage()
        else:
            raise WalkForwardError(f"escenario desconocido: {scenario!r}")
        return SimConfig(
            initial_cash=self.initial_cash,
            spread=spread,
            slippage=slippage,
            max_participation=self.max_participation,
            cash_rate=self.cash_rate,
            bars_per_year=self.bars_per_year,
            run_id=f"walkforward_btcusdt_1d_{scenario}",
        )

    def ppo_config(self) -> PPOConfig:
        """Los demas hiperparametros son los del protocolo (opcion i)."""
        return PPOConfig(
            total_timesteps=self.total_timesteps, reward_scale=self.reward_scale
        )

    def describe(self) -> dict[str, object]:
        datos: dict[str, object] = asdict(self)
        datos["seeds"] = list(self.seeds)
        # Todos los hiperparametros de PPO entran al hash del sello: cambiar un
        # default de PPOConfig despues de sellar cambia el digest y se detecta.
        datos["ppo"] = self.ppo_config().describe()
        # El sizer define que ordenes existen y por lo tanto la observacion del
        # agente (last_order_rejected). Entra al digest: reanudar una corrida
        # sellada con otro sizer falla en vez de mezclar comportamientos.
        datos["sizer"] = TargetWeightSizer(safety=self.safety).describe()
        return datos

    def digest(self) -> str:
        crudo = json.dumps(self.describe(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(crudo).hexdigest()


# ---------------------------------------------------------------------------
# Geometria y regimen (ADR 0006, secciones 1 y 4.1)
# ---------------------------------------------------------------------------


def folds(n_bars: int, config: WalkForwardConfig) -> tuple[Window, ...]:
    return rolling_windows(
        n_bars,
        train=config.train,
        validation=config.validation,
        test=config.test,
        step=config.step,
    )


def training_span(window: Window) -> tuple[int, int]:
    """D2: se entrena con train + validacion, las 1.000 barras."""
    return window.train[0], window.validation[1]


def _log_returns(series: BarSeries) -> FloatArray:
    cierre = np.log(np.asarray(series.close, dtype=np.float64))
    previo = np.concatenate([[np.log(float(series.open[0]))], cierre[:-1]])
    retornos: FloatArray = cierre - previo
    return retornos


@dataclass(frozen=True)
class RegimeClass:
    fold: int
    r_ant: float
    r_rec: float
    t_rec: float
    label: str  # "tras_cambio" | "sin_cambio" | "ambigua"


def classify_regime(series: BarSeries, window: Window) -> RegimeClass:
    """Seccion 4.1: las 800 barras anteriores contra las 200 recientes, con piso."""
    r = _log_returns(series)
    ant = r[window.train[0] : window.validation[0]]
    rec = r[window.validation[0] : window.validation[1]]
    r_ant, r_rec = float(ant.sum()), float(rec.sum())
    t_rec = r_rec / (float(rec.std(ddof=1)) * float(np.sqrt(len(rec))))
    if abs(t_rec) < PISO_T_RECIENTE:
        label = "ambigua"
    elif np.sign(r_ant) != np.sign(r_rec):
        label = "tras_cambio"
    else:
        label = "sin_cambio"
    return RegimeClass(window.index, r_ant, r_rec, t_rec, label)


# ---------------------------------------------------------------------------
# Sellado (ADR 0006, seccion 7)
# ---------------------------------------------------------------------------


def verify_adr(root: Path, expected: str = ADR_SHA256) -> str:
    real = hashlib.sha256((root / ADR_PATH).read_bytes()).hexdigest()
    if real != expected:
        raise SealError(
            f"el ADR 0006 en disco tiene SHA256 {real} y el sellado es {expected}: "
            "el criterio cambio despues de sellarse"
        )
    return real


def acquire_seal(path: Path, payload: dict[str, str], *, resume: bool) -> None:
    """Crea el sello de forma exclusiva. Si ya existe, solo ``resume`` con el mismo."""
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return
    except FileExistsError:
        existente = json.loads(path.read_text(encoding="utf-8"))
    if not resume:
        raise SealError(
            f"el sello {path} ya existe: el test ya se toco. Una segunda corrida "
            "exige borrarlo a mano, y eso queda en la historia"
        )
    if existente != payload:
        raise SealError(
            "resume con una configuracion distinta de la sellada: "
            f"{existente} != {payload}"
        )


# ---------------------------------------------------------------------------
# Un par (fold, semilla)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSeedRecord:
    fold: int
    seed: int
    e_mark: float  # exceso log sobre B&H, seccion 2
    e_liq: float
    t_mark: float  # exceso sobre B&H de exposicion igualada, seccion 4.2
    t_liq: float
    exposure: float  # e(k,s): fraccion media del equity invertida
    exposure_first: float  # subsidiaria 3
    exposure_rest: float
    action_std: float
    n_fills: int
    n_outliers: int
    e_mark_sin_outliers: float  # seccion 5, version secundaria
    n_orders: int  # ordenes enviadas por el agente (seccion 1.1)
    n_rejected_min_notional: int
    n_rejected_other: int
    # Conteo por RejectReason. Ausente en los registros de C3, que solo guardaron
    # MIN_NOTIONAL y "otro": se lee como {} y el reporte lo dice.
    rejected_by_reason: dict[str, int]
    baselines: dict[str, dict[str, float]]
    # Sensibilidad de costos: la MISMA politica, sin reentrenar, con solo la
    # comision. Mismas claves que el escenario principal; no decide.
    sensitivity: dict[str, Any] = field(repr=False)
    excess_daily: tuple[float, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        datos = asdict(self)
        datos["excess_daily"] = list(self.excess_daily)
        return datos

    @classmethod
    def from_dict(cls, datos: dict[str, Any]) -> FoldSeedRecord:
        completo = dict(datos)
        completo["excess_daily"] = tuple(completo["excess_daily"])
        completo.setdefault("rejected_by_reason", {})
        completo["sensitivity"] = {"rejected_by_reason": {}, **completo["sensitivity"]}
        return cls(**completo)


def _crecimiento(equity: FloatArray, desde: int) -> float:
    return float(np.log(equity[-1] / equity[desde]))


def _retorno(equity: FloatArray, desde: int) -> float:
    return float(equity[-1] / equity[desde] - 1.0)


def evaluate_fold_seed(
    series: BarSeries,
    marks: FloatArray,
    window: Window,
    seed: int,
    config: WalkForwardConfig,
    policy_factory: PolicyFactory,
) -> FoldSeedRecord:
    """Entrena con las 1.000 barras del fold y evalua en su test, una vez.

    El test se corre sobre ``[test.start - warmup, test.stop)``: los indicadores
    de la primera barra de test se calculan con barras anteriores, que son pasado.
    Agente y baselines deciden por primera vez en la misma barra, ``test.start``.
    Los baselines corren sobre un prefijo mas largo (``MA_LENTA + 1`` barras) con
    su propio ``WarmupDelay``, asi que tampoco operan antes de ``test.start``;
    todas las metricas se miden desde esa barra.
    """
    observacion = ObservationSpec()
    calentamiento = observacion.warmup
    entrenamiento = series.slice(*training_span(window))
    escalador = scaler_for(entrenamiento, observacion)  # solo con el train del fold
    config_sim = config.sim_config(PRINCIPAL)
    env_config = EnvConfig(
        observation=observacion, sizer=TargetWeightSizer(safety=config.safety)
    )

    def entorno(serie: BarSeries, costos: SimConfig) -> TradingEnv:
        return build_env(
            serie,
            costos,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=config.reward_scale),
        )

    politica = policy_factory(lambda: entorno(entrenamiento, config_sim), seed)
    principal = _escenario(
        series, marks, window, seed, politica, entorno, config_sim, calentamiento
    )
    sensibilidad = _escenario(
        series,
        marks,
        window,
        seed,
        politica,
        entorno,
        config.sim_config(SOLO_COMISION),
        calentamiento,
    )
    return FoldSeedRecord(
        fold=window.index,
        seed=seed,
        **{k: v for k, v in principal.items() if k in _CAMPOS_PRINCIPALES},
        sensitivity={
            k: v for k, v in sensibilidad.items() if k in _CAMPOS_SENSIBILIDAD
        },
    )


_CAMPOS_SENSIBILIDAD = (
    "e_mark",
    "e_liq",
    "t_mark",
    "t_liq",
    "exposure",
    "n_orders",
    "n_rejected_min_notional",
    "n_rejected_other",
    "rejected_by_reason",
    "baselines",
    "excess_daily",
)
_CAMPOS_PRINCIPALES = (
    *_CAMPOS_SENSIBILIDAD,
    "exposure_first",
    "exposure_rest",
    "action_std",
    "n_fills",
    "n_outliers",
    "e_mark_sin_outliers",
)


def _rechazos(resultado: SimResult) -> tuple[int, int, int, dict[str, int]]:
    """Ordenes enviadas, rechazadas por MIN_NOTIONAL, por otro motivo, y el
    conteo por motivo.

    Los rechazos quedan en el log de fills con su motivo: el venue rechaza y
    registra, nunca redimensiona. Aca solo se cuentan. El conteo por motivo
    existe porque C3 guardo solo "otro" y no se pudo saber que era sin tocar el
    test de nuevo.
    """
    rechazadas = [f for f in resultado.fills if f.status is OrderStatus.REJECTED]
    por_motivo: dict[str, int] = {}
    for f in rechazadas:
        clave = f.reject_reason.value if f.reject_reason is not None else "SIN_MOTIVO"
        por_motivo[clave] = por_motivo.get(clave, 0) + 1
    minimo = por_motivo.get(RejectReason.MIN_NOTIONAL.value, 0)
    return len(resultado.fills), minimo, len(rechazadas) - minimo, por_motivo


def _sumar_motivos(conteos: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for conteo in conteos:
        for motivo, n in conteo.items():
            total[motivo] = total.get(motivo, 0) + n
    return dict(sorted(total.items()))


def _escenario(
    series: BarSeries,
    marks: FloatArray,
    window: Window,
    seed: int,
    politica: Policy,
    entorno: Callable[[BarSeries, SimConfig], TradingEnv],
    config_sim: SimConfig,
    c: int,
) -> dict[str, Any]:
    """Evalua una politica ya entrenada y los baselines bajo un modelo de costos."""
    inicio_eval = window.test[0] - c
    evaluacion = series.slice(inicio_eval, window.test[1])
    salida = run_policy(entorno(evaluacion, config_sim), politica, seed=seed)
    prefijo = max(c, MA_LENTA + 1)
    bases = run_baselines(
        series.slice(window.test[0] - prefijo, window.test[1]),
        config_sim,
        warmup=prefijo,
        seed=seed,
        ma_slow=MA_LENTA,
    )
    agente, bh = salida.result, bases["buy_and_hold"]

    mark = np.asarray(agente.equity, dtype=np.float64)
    liq = np.asarray(agente.equity_liquidation, dtype=np.float64)
    # Los baselines se recortan a partir de su primera decision, test.start, para
    # que las series del agente y de los baselines cubran las mismas barras.
    bh_mark = np.asarray(bh.equity, dtype=np.float64)[prefijo - c :]
    bh_liq = np.asarray(bh.equity_liquidation, dtype=np.float64)[prefijo - c :]

    invertido = (
        np.asarray(agente.position, dtype=np.float64)
        * np.asarray(evaluacion.close, dtype=np.float64)
        / mark
    )[c:]
    e = float(invertido.mean())

    def timing(agente_q: FloatArray, bh_q: FloatArray) -> float:
        return float(
            np.log(1.0 + _retorno(agente_q, c)) - np.log(1.0 + e * _retorno(bh_q, c))
        )

    ejecutados = [f for f in agente.fills if f.is_executed]
    paso = pd.Timedelta(evaluacion.freq)
    outliers = [
        f
        for f in ejecutados
        if marks[inicio_eval + f.t_decision] != 0
        or pd.Timedelta(f.timestamp_fill - f.timestamp_decision) > paso
    ]
    gap_outliers = sum(f.gap for f in outliers)
    enviadas, minimo, otros, por_motivo = _rechazos(agente)

    r_agente = np.diff(mark[c:]) / mark[c:-1]
    r_bh = np.diff(bh_mark[c:]) / bh_mark[c:-1]
    acciones = np.asarray(salida.actions, dtype=np.float64)
    n = BARRAS_SUBSIDIARIA_3
    return {
        "e_mark": _crecimiento(mark, c) - _crecimiento(bh_mark, c),
        "e_liq": _crecimiento(liq, c) - _crecimiento(bh_liq, c),
        "t_mark": timing(mark, bh_mark),
        "t_liq": timing(liq, bh_liq),
        "exposure": e,
        "exposure_first": float(invertido[:n].mean()),
        "exposure_rest": float(invertido[n:].mean()),
        "action_std": float(acciones.std()) if acciones.size else 0.0,
        "n_fills": len(ejecutados),
        "n_outliers": len(outliers),
        # Neutraliza el gap de los fills outlier: es la parte del resultado que
        # dependio de ejecutar al reanudar tras una caida o un hueco.
        "e_mark_sin_outliers": float(np.log((mark[-1] + gap_outliers) / mark[c]))
        - _crecimiento(bh_mark, c),
        "n_orders": enviadas,
        "n_rejected_min_notional": minimo,
        "n_rejected_other": otros,
        "rejected_by_reason": por_motivo,
        "baselines": {
            nombre: {
                "log_growth_mark": _crecimiento(np.asarray(r.equity), prefijo),
                "log_growth_liq": _crecimiento(
                    np.asarray(r.equity_liquidation), prefijo
                ),
                "n_rejected_min_notional": _rechazos(r)[1],
            }
            for nombre, r in bases.items()
        },
        "excess_daily": tuple(float(x) for x in r_agente - r_bh),
    }


# ---------------------------------------------------------------------------
# Veredictos (ADR 0006, secciones 2 y 4.3)
# ---------------------------------------------------------------------------


def main_verdict(
    x_mark: Sequence[float], x_liq: Sequence[float], dsr: Sequence[float | None]
) -> str:
    """Seccion 2. Un DSR indefinido cuenta como no significativo (0.0): es la
    lectura conservadora, y no puede convertir un NO SUPERA en SUPERA."""
    dsr_efectivo = [0.0 if d is None or not np.isfinite(d) else d for d in dsr]
    supera = (
        float(np.median(x_mark)) > 0.0
        and float(np.median(x_liq)) > 0.0
        and float(np.median(dsr_efectivo)) >= DSR_UMBRAL
    )
    return "SUPERA" if supera else "NO SUPERA"


def prediction_verdict(
    t_by_fold: Mapping[int, Sequence[float]], labels: Mapping[int, str]
) -> dict[str, Any]:
    """``t_by_fold[k]`` son los ``T(k,s)`` de las semillas del fold ``k``."""
    tras = [k for k, v in labels.items() if v == "tras_cambio"]
    sin = [k for k, v in labels.items() if v == "sin_cambio"]
    mediana = {k: float(np.median(t_by_fold[k])) for k in t_by_fold}
    iqr = [
        float(np.percentile(v, 75) - np.percentile(v, 25)) for v in t_by_fold.values()
    ]
    piso = float(np.median(iqr))
    if len(tras) < 2 or len(sin) < 2:
        return {"verdict": "NO CONTRASTABLE", "G": None, "F": piso}
    brecha = float(np.median([mediana[k] for k in tras])) - float(
        np.median([mediana[k] for k in sin])
    )
    if brecha >= 0.0:
        veredicto = "REFUTADA"
    elif brecha < -piso:
        veredicto = "CONSISTENTE"  # el maximo: este diseno no puede confirmar
    else:
        veredicto = "NO CONCLUYENTE"
    return {"verdict": veredicto, "G": brecha, "F": piso}


def _dsr(serie: Sequence[float], varianza: float, n_trials: int) -> float | None:
    """DSR, o ``None`` si no esta definido: un exceso sin dispersion (el agente
    replico a buy-and-hold barra a barra) no tiene Sharpe. Indefinido no es cero,
    y se reporta como tal; en el veredicto cuenta como no significativo."""
    try:
        return deflated_sharpe(
            np.asarray(serie, dtype=np.float64),
            n_trials=n_trials,
            sharpe_variance=varianza,
        ).value
    except DistributionError:
        return None


def _sharpe(serie: Sequence[float]) -> float:
    desvio = float(np.std(serie, ddof=1))
    return float(np.mean(serie)) / desvio if desvio > 0 else float("nan")


def assemble(
    records: Sequence[FoldSeedRecord],
    regimes: Sequence[RegimeClass],
    *,
    allow_fewer_seeds: bool = False,
) -> dict[str, Any]:
    """Aplica los criterios sellados sobre los registros guardados."""
    semillas = sorted({r.seed for r in records})
    por_semilla = {
        s: sorted((r for r in records if r.seed == s), key=lambda r: r.fold)
        for s in semillas
    }

    def dist(nombre: str, valores: Sequence[float | None]) -> dict[str, object]:
        return summarize(
            nombre, semillas, list(valores), allow_fewer_seeds=allow_fewer_seeds
        ).describe()

    def criterio(
        campos: Callable[[FoldSeedRecord], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Seccion 2 sobre un escenario de costos: veredicto, DSR, secundario y
        rechazos por min_notional."""

        def suma(clave: str, s: int) -> float:
            return float(sum(campos(r)[clave] for r in por_semilla[s]))

        x_mark = [suma("e_mark", s) for s in semillas]
        x_liq = [suma("e_liq", s) for s in semillas]
        exceso = {
            s: [x for r in por_semilla[s] for x in campos(r)["excess_daily"]]
            for s in semillas
        }
        sharpes = [_sharpe(v) for v in exceso.values()]
        finitos = [x for x in sharpes if np.isfinite(x)]
        varianza = float(np.var(finitos, ddof=1)) if len(finitos) > 1 else 0.0
        dsr = [_dsr(exceso[s], varianza, DSR_N_TRIALS) for s in semillas]
        enviadas = sum(campos(r)["n_orders"] for r in records)
        minimo = sum(campos(r)["n_rejected_min_notional"] for r in records)
        return {
            "verdict": main_verdict(x_mark, x_liq, dsr),
            "x_mark": dist("x_mark", x_mark),
            "x_liq": dist("x_liq", x_liq),
            "dsr_n10": dist("dsr_n10", dsr),
            "dsr_n1": dist("dsr_n1", [_dsr(exceso[s], varianza, 1) for s in semillas]),
            "secondary_x_exposure_matched": dist(
                "x_exp", [suma("t_mark", s) for s in semillas]
            ),
            "orders": {
                "sent": int(enviadas),
                "rejected_min_notional": int(minimo),
                "rejected_other": int(
                    sum(campos(r)["n_rejected_other"] for r in records)
                ),
                "fraction_rejected_min_notional": minimo / enviadas
                if enviadas
                else None,
                "rejected_by_reason": _sumar_motivos(
                    campos(r)["rejected_by_reason"] for r in records
                ),
            },
        }

    def principal(r: FoldSeedRecord) -> Mapping[str, Any]:
        return asdict(r)

    def sensibilidad(r: FoldSeedRecord) -> Mapping[str, Any]:
        return r.sensitivity

    etiquetas = {c.fold: c.label for c in regimes}
    t_por_fold = {
        k: [r.t_mark for r in records if r.fold == k] for k in sorted(etiquetas)
    }
    subsidiaria_3 = {}
    for clase in regimes:
        if clase.label != "tras_cambio":
            continue
        filas = [r for r in records if r.fold == clase.fold]
        delta = float(np.median([r.exposure_first - r.exposure_rest for r in filas]))
        subsidiaria_3[clase.fold] = {
            "delta_exposicion": delta,
            "en_direccion_del_regimen_anterior": bool(
                np.sign(delta) == np.sign(clase.r_ant)
            ),
        }
    return {
        "main": {
            **criterio(principal),
            "secondary_x_sin_outliers": dist(
                "x_sin_outliers",
                [sum(r.e_mark_sin_outliers for r in por_semilla[s]) for s in semillas],
            ),
            "n_outliers": sum(r.n_outliers for r in records),
        },
        # Seccion 1.1: las mismas politicas, sin reentrenar, con solo la comision.
        # Secundaria: no decide.
        "sensitivity_commission_only": criterio(sensibilidad),
        "prediction_adr_0004": {
            **prediction_verdict(t_por_fold, etiquetas),
            "classes": [asdict(c) for c in regimes],
            "subsidiary_3": {
                "per_fold": subsidiaria_3,
                # Seccion 4.3: sostenida si ocurre en la mayoria de los folds
                # tras cambio.
                "sostenida": bool(
                    subsidiaria_3
                    and sum(
                        v["en_direccion_del_regimen_anterior"]
                        for v in subsidiaria_3.values()
                    )
                    > len(subsidiaria_3) / 2
                ),
            },
        },
        "per_fold": {
            k: {
                m: dist(f"{m}_fold{k}", [getattr(r, m) for r in records if r.fold == k])
                for m in (
                    "e_mark",
                    "e_liq",
                    "t_mark",
                    "exposure",
                    "action_std",
                    "n_rejected_min_notional",
                )
            }
            for k in sorted(etiquetas)
        },
    }


# ---------------------------------------------------------------------------
# Corrida (Tarea C3)
# ---------------------------------------------------------------------------


def load_dataset(root: Path) -> tuple[BarSeries, FloatArray, str]:
    """Serie diaria verificada por SHA256, marcas de anomalia y SHA del dataset."""
    manifest_path = root / DATASET_MANIFEST
    serie = load_versioned_bars(
        manifest_path,
        instrument=binance_spot_spec(),
        freq="1D",
        calendar=AlwaysOpen(),
    )
    manifest = read_manifest(manifest_path)
    crudo = pd.read_csv(manifest_path.parent / manifest.file)
    marcas = crudo["close_time_desvio_ms"].to_numpy(dtype=np.float64)
    return serie, marcas, manifest.sha256


def run_walkforward(
    root: Path,
    out_dir: Path,
    policy_factory: PolicyFactory,
    *,
    config: WalkForwardConfig | None = None,
    run_commit: str = "",
    resume: bool = False,
    expected_adr_sha: str = ADR_SHA256,
) -> dict[str, Any]:
    config = config or WalkForwardConfig()
    adr_sha = verify_adr(root, expected_adr_sha)
    serie, marcas, dataset_sha = load_dataset(root)
    ventanas = folds(len(serie), config)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_seal(
        out_dir / "SEAL.json",
        {
            "adr_sha256": adr_sha,
            "dataset_sha256": dataset_sha,
            "config": config.digest(),
        },
        resume=resume,
    )

    registros_path = out_dir / "records.jsonl"
    hechos: list[FoldSeedRecord] = []
    if registros_path.exists():
        hechos = [
            FoldSeedRecord.from_dict(json.loads(linea))
            for linea in registros_path.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        ]
    listos = {(r.fold, r.seed) for r in hechos}
    for ventana in ventanas:
        for semilla in config.seeds:
            if (ventana.index, semilla) in listos:
                continue
            registro = evaluate_fold_seed(
                serie, marcas, ventana, semilla, config, policy_factory
            )
            with registros_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(registro.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            hechos.append(registro)

    reporte = assemble(
        hechos,
        [classify_regime(serie, v) for v in ventanas],
        allow_fewer_seeds=len(config.seeds) < 10,
    )
    reporte["provenance"] = {
        "adr_sha256": adr_sha,
        "adr_seal_commit": ADR_SEAL_COMMIT,
        "run_commit": run_commit,
        "dataset_sha256": dataset_sha,
        "config": config.describe(),
        "config_digest": config.digest(),
        "records_sha256": hashlib.sha256(registros_path.read_bytes()).hexdigest(),
        "limitations": (
            "politica subentrenada: valida el pipeline, no concluye sobre el mercado"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(reporte, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return reporte
