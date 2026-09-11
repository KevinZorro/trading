"""Protocolo de validacion del Agente A sobre la escalera de fixtures.

Los cinco niveles se corren **en orden** y el protocolo **para en el primero que
falla**. No es una comodidad: si el nivel 0 falla, el bug esta en el pipeline y
seguir al nivel 1 solo produce numeros que hay que descartar despues. Peor,
invita a tocar hiperparametros para "arreglar" un sintoma cuya causa esta en
otro lado.

**Los criterios se declaran antes de correr** y viajan serializados con el
resultado (:class:`ProtocolThresholds`). Un umbral elegido despues de ver los
numeros no es un criterio, es una descripcion.

Los tres veredictos posibles no son dos:

- ``PASS`` / ``FAIL``: el nivel tenia un criterio y se cumplio o no.
- ``MEASURED``: el nivel **no** tiene criterio de aprobacion porque los dos
  resultados posibles son validos. Es el caso del nivel 3, donde adaptarse y
  memorizar son dos hallazgos distintos y ambos legitimos siempre que queden
  medidos. Forzar un PASS/FAIL ahi seria inventar una hipotesis despues del
  hecho.

Sobre que tramo se usa: el agente entrena en **train** y se evalua en
**validacion**. El tramo de **test de los fixtures queda sin tocar**, igual que
el de los datos reales. No hace falta para el protocolo -los fixtures no son
datos de investigacion- y reservarlo mantiene la misma disciplina en todos lados.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

from agents.policy import Policy
from agents.ppo import PPOConfig, action_saturation, train_ppo
from agents.runner import (
    EpisodeOutcome,
    build_env,
    run_baselines,
    run_policy,
    scaler_for,
)
from data.fixtures import (
    Ceilings,
    Fixture,
    evaluate_states,
    myopic_states,
    periodic_rate_per_bar,
)
from data.schema import FloatArray
from envs.observation import ObservationSpec
from envs.rewards import NetReturnReward
from envs.trading_env import EnvConfig
from eval.distribution import (
    SeedDistribution,
    VarianceDecomposition,
    decompose_variance,
    drift_t_statistic,
    summarize,
)
from eval.report import RunReport, evaluate_run
from sim.engine import SimConfig, SimResult
from sim.sizing import TargetWeightSizer


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    MEASURED = "MEASURED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ProtocolThresholds:
    """Criterios de aprobacion. **Declarados antes de correr y serializados.**

    Cada numero con su justificacion, porque un umbral sin motivo se ajusta
    solo hasta que el experimento pasa:

    - ``level_0_min_capture = 0.80``: el nivel 0 no tiene ruido y el optimo es
      una funcion del ultimo retorno que esta literalmente en la observacion.
      Un agente sano lo resuelve casi entero; 0.80 deja margen para el
      calentamiento y la exploracion residual sin admitir un agente que solo
      aprendio la mitad.
    - ``level_1_monotonia_tol = 0.10``: la degradacion tiene que ser suave, no
      exactamente monotona. Con 10 semillas el error estandar de la mediana es
      del orden de 0.05, asi que exigir monotonia estricta haria fallar por
      ruido muestral.
    - ``level_2_min_turnover_drop = 0.10``: con costos el optimo desarrolla una
      banda de no-operar. Un 10% menos de rotacion es el minimo detectable por
      encima del ruido; si el agente opera **igual** con y sin costos, la
      penalizacion no esta llegando al reward.
    - **La rotacion no es criterio en el nivel 4, y esto es deliberado.** Se
      reporta como evidencia pero no entra en el veredicto: el unico baseline
      con rotacion comparable seria "estar siempre invertido", que rebalancea al
      peso objetivo en cada barra igual que el agente, mientras que
      ``BuyAndHold`` compra una vez y su rotacion anualizada es ~0.05. Un
      criterio contra esa referencia daria ratios de tres cifras y fallaria
      siempre, incluso para un agente perfectamente convergido. Lo que si
      captura "no rota" es ``level_4_min_time_invested``: un agente que se queda
      invertido el 95% del tiempo no esta entrando y saliendo.
    - ``level_4_min_paths = 10``: el criterio del exceso **no** es un umbral
      sobre la mediana. Es un contraste contra la dispersion **entre caminos**,
      y para eso hacen falta caminos. Diez semillas sobre uno solo miden
      varianza de entrenamiento, que es otra pregunta. Ver
      ``eval.distribution.decompose_variance``.

    Un umbral que estuvo y se quito: ``level_4_max_median_excess = 0.05``, que
    comparaba la mediana del exceso contra un numero fijo. Fallaba con +0.164
    sobre un camino cuyo baseline tenia desvio 0.781 entre caminos: declaraba
    significativo algo cinco veces mas chico que el ruido del sorteo. El umbral
    no era demasiado laxo ni demasiado estricto, estaba mal planteado.

    Otro que se movio de nivel: el tiempo invertido **salio del 4a**. Sobre un
    fixture cuyo drift no es detectable (t poblacional 0.51), abstenerse no es un
    error sino la respuesta defendible a una serie donde no hay senal que seguir.
    Exigir convergencia ahi mezclaba dos hipotesis en un solo veredicto. La
    pregunta "¿reconoce drift cuando existe?" es legitima y ahora vive en el
    **nivel 4b**, con un fixture calibrado para que la respuesta sea posible.

    Umbrales del 4b, declarados antes de correrlo (ver ADR 0005):

    - ``level_4b_min_drift_t = 3.0``: **condicion sobre el fixture, no sobre el
      agente**. Si el ``t`` mediano del drift en la ventana de entrenamiento no
      llega a 3, el fixture no tiene lo que dice tener y el nivel no puede medir
      lo que pretende. El calculo esta en ``data.fixtures.drift_t_population``.
    - ``level_4b_min_time_invested = 0.80``: con drift detectable y sin senal
      direccional, el optimo es estar invertido.
    - ``level_4b_max_action_std = 0.15``: "no rota". Una politica fija en 1.0 da
      dispersion 0; una que alterna entre 0 y 1 la mitad del tiempo da 0.5. El
      umbral admite oscilar entre 0.8 y 1.0 (dispersion ~0.1) y excluye rotar.
      Se usa la dispersion de la accion y no la rotacion anualizada porque esta
      ultima mezcla el comportamiento con el crecimiento del equity y con el
      rebalanceo al peso objetivo.
    """

    level_0_min_capture: float = 0.80
    level_1_monotonia_tol: float = 0.10
    level_1_min_capture_highest_snr: float = 0.50
    level_2_min_turnover_drop: float = 0.10
    level_4_min_paths: int = 10
    level_4b_min_drift_t: float = 3.0
    level_4b_min_time_invested: float = 0.80
    level_4b_max_action_std: float = 0.15

    def describe(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass(frozen=True)
class SeedRun:
    """Resultado de una semilla sobre un fixture. Nunca se reporta solo."""

    seed: int
    capture: float | None
    log_growth: float
    excess_log_growth_vs_always_long: float
    total_return_mark: float
    total_return_liquidation: float
    friction_gap: float
    sharpe_mark: float | None
    max_drawdown_mark: float
    turnover_annualized: float
    time_invested: float
    action_std: float
    saturation: float
    clipped_actions: int
    n_round_trips: int
    win_rate: float | None
    costs_paid: float
    gap: float
    ruined: bool

    def describe(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass(frozen=True)
class ArmResult:
    """Un fixture concreto corrido sobre varias semillas.

    Se puede reconstruir desde su forma serializada (:meth:`from_dict`). No es
    una comodidad: entrenar 10 semillas cuesta horas, y volver a aplicar los
    criterios sobre resultados guardados -por ejemplo para revisar un umbral con
    los mismos numeros- no puede exigir reentrenar. Las distribuciones se
    **recalculan** desde las corridas en vez de leerse, asi que un archivo con
    distribuciones inconsistentes con sus corridas no puede pasar inadvertido.
    """

    label: str
    fixture_name: str
    fixture_config: dict[str, Any]
    ppo_config: dict[str, Any]
    runs: tuple[SeedRun, ...]
    distributions: dict[str, SeedDistribution]
    ceiling: dict[str, Any]
    baselines: dict[str, dict[str, Any]]

    def median(self, metric: str) -> float | None:
        return self.distributions[metric].median

    def beats(self, baseline: str) -> float | None:
        """Fraccion de semillas que supera el retorno de un baseline.

        El retorno del baseline es un escalar -son deterministas salvo el
        aleatorio-, asi que la comparacion honesta es cuantas semillas lo
        superan, no si la mediana lo supera: dos distribuciones con la misma
        mediana y dispersiones distintas no son la misma evidencia.
        """
        referencia = self.baselines[baseline]["total_return_mark"]
        return self.distributions["total_return_mark"].fraction_above(float(referencia))

    @classmethod
    def from_dict(cls, datos: dict[str, Any]) -> ArmResult:
        corridas = tuple(SeedRun(**r) for r in datos["runs"])
        return cls(
            label=str(datos["label"]),
            fixture_name=str(datos["fixture"]["name"]),
            fixture_config=dict(datos["fixture"]),
            ppo_config=dict(datos["ppo"]),
            runs=corridas,
            distributions=_distribuciones(corridas, allow_fewer_seeds=True),
            ceiling=dict(datos["ceiling"]),
            baselines=dict(datos["baselines"]),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "fixture": self.fixture_config,
            "ppo": self.ppo_config,
            "n_seeds": len(self.runs),
            "distributions": {k: v.describe() for k, v in self.distributions.items()},
            "ceiling": self.ceiling,
            "baselines": self.baselines,
            "beats_baselines": {b: self.beats(b) for b in self.baselines},
            "runs": [r.describe() for r in self.runs],
        }


@dataclass(frozen=True)
class LevelResult:
    """Veredicto de un nivel, con el criterio que se declaro antes de correr."""

    level: int
    label: str
    verdict: Verdict
    statement: str
    finding: str
    arms: tuple[ArmResult, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "label": self.label,
            "verdict": str(self.verdict),
            "statement": self.statement,
            "finding": self.finding,
            "arms": [a.describe() for a in self.arms],
        }


@dataclass
class ProtocolReport:
    """Los niveles corridos, en orden, hasta el primero que fallo."""

    thresholds: ProtocolThresholds
    levels: list[LevelResult] = field(default_factory=list)

    @property
    def stopped_at(self) -> int | None:
        for nivel in self.levels:
            if nivel.verdict is Verdict.FAIL:
                return nivel.level
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.describe(),
            "stopped_at": self.stopped_at,
            "levels": [n.describe() for n in self.levels],
        }

    def render(self) -> str:
        lineas = ["PROTOCOLO DE VALIDACION - AGENTE A", "=" * 72]
        for nivel in self.levels:
            lineas.append(f"[{nivel.verdict}] Nivel {nivel.level} - {nivel.label}")
            lineas.append(f"    criterio: {nivel.statement}")
            lineas.append(f"    hallazgo: {nivel.finding}")
            for arm in nivel.arms:
                lineas.append(f"    {arm.label}:")
                for clave in ("capture", "total_return_mark", "turnover_annualized"):
                    if clave in arm.distributions:
                        lineas.append(f"      {arm.distributions[clave].render()}")
        parada = self.stopped_at
        lineas.append("=" * 72)
        lineas.append(
            "PROTOCOLO DETENIDO en el nivel " + str(parada)
            if parada is not None
            else "Protocolo completo sin fallos"
        )
        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Corrida de un brazo
# ---------------------------------------------------------------------------

PolicyFactory = Callable[[Fixture, int], Policy]


def _sim_config(fixture: Fixture, initial_cash: float, cash_rate: float) -> SimConfig:
    """Traduce los costos del fixture -que son datos- a modelos ejecutables.

    Vive aca y no en ``data.fixtures`` porque ``data`` no depende de ``sim``.
    Es el unico lugar donde se hace la traduccion, asi que ningun experimento
    puede correr el nivel 2 con costos distintos de los que se calibraron.
    """
    from sim.costs import FixedBpsSpread, NoSlippage, SchemaCommission

    return SimConfig(
        initial_cash=initial_cash,
        spread=FixedBpsSpread(bps=fixture.costs.spread_bps),
        slippage=NoSlippage(),
        commission=SchemaCommission(schema=fixture.costs.commission),
        max_participation=1.0,
        cash_rate=cash_rate,
        run_id=f"protocol_{fixture.name}",
    )


def _seed_run(
    seed: int,
    outcome: EpisodeOutcome,
    report: RunReport,
    ceilings: Ceilings,
    initial_cash: float,
) -> SeedRun:
    equity = np.asarray(outcome.result.equity, dtype=np.float64)
    crecimiento = float(np.log(equity[-1] / equity[0]))
    largo = np.asarray(ceilings.always_long, dtype=np.float64)
    crecimiento_largo = float(np.log(largo[-1] / largo[0]))
    return SeedRun(
        seed=seed,
        capture=ceilings.capture(equity),
        log_growth=crecimiento,
        excess_log_growth_vs_always_long=crecimiento - crecimiento_largo,
        total_return_mark=report.mark.total_return,
        total_return_liquidation=report.liquidation.total_return,
        friction_gap=report.friction.final_gap_pct,
        sharpe_mark=report.mark.sharpe,
        max_drawdown_mark=report.mark.drawdown.depth,
        turnover_annualized=report.turnover.annualized,
        time_invested=float(np.mean(outcome.actions)) if outcome.actions else 0.0,
        # Dispersion de la accion dentro del episodio. Es la medida de "no rota"
        # que no depende de ningun baseline: una politica fija en 1.0 da 0, y una
        # que alterna entre 0 y 1 la mitad del tiempo da 0.5. La rotacion
        # anualizada mezcla eso con el crecimiento del equity y con el
        # rebalanceo al peso objetivo, asi que sirve de evidencia y no de criterio.
        action_std=float(np.std(outcome.actions)) if outcome.actions else 0.0,
        saturation=action_saturation(outcome.actions),
        clipped_actions=outcome.clipped_actions,
        n_round_trips=report.trades.n_round_trips,
        win_rate=report.trades.win_rate,
        costs_paid=report.costs.total_costs_paid,
        gap=report.costs.gap,
        ruined=outcome.ruined,
    )


_METRICAS = (
    "capture",
    "log_growth",
    "excess_log_growth_vs_always_long",
    "total_return_mark",
    "total_return_liquidation",
    "friction_gap",
    "sharpe_mark",
    "max_drawdown_mark",
    "turnover_annualized",
    "time_invested",
    "action_std",
    "saturation",
    "n_round_trips",
    "win_rate",
)


def run_arm(
    fixture: Fixture,
    *,
    label: str,
    seeds: Sequence[int],
    ppo_config: PPOConfig,
    policy_factory: PolicyFactory | None = None,
    initial_cash: float = 100_000.0,
    cash_rate: float = 0.0,
    observation: ObservationSpec | None = None,
    safety: float = 0.98,
    allow_fewer_seeds: bool = False,
) -> ArmResult:
    """Entrena y evalua un fixture sobre varias semillas.

    Secuencia por semilla, y el orden importa:

    1. Particion train / validacion / test del fixture.
    2. **El escalador se ajusta solo con train.** Recorre ``MarketView`` sobre esa
       porcion, asi que la serie de validacion no esta en el objeto.
    3. Se entrena sobre train.
    4. Se evalua sobre validacion, de forma determinista.
    5. El techo se pide con el **mismo** ``safety`` y el mismo
       ``first_decision`` que usa el agente, o el deficit del margen de
       seguridad y del calentamiento se leerian como fallo de aprendizaje.

    ``policy_factory`` permite inyectar una politica que no sea PPO -es lo que
    usan los tests para correr el protocolo entero sin torch-.
    """
    train, validacion, _test = fixture.split()
    config_sim = _sim_config(fixture, initial_cash, cash_rate)
    escalador = scaler_for(train.series, observation)
    env_config = EnvConfig(
        observation=observation or ObservationSpec(),
        sizer=TargetWeightSizer(safety=safety),
    )

    def construir(serie_fixture: Fixture) -> Any:
        return build_env(
            serie_fixture.series,
            config_sim,
            scaler=escalador,
            env_config=env_config,
            reward=NetReturnReward(scale=ppo_config.reward_scale),
        )

    calentamiento = env_config.observation.warmup
    memorizador = _memorizer_curve(
        fixture,
        validacion,
        safety=safety,
        rate_per_bar=periodic_rate_per_bar(cash_rate, 252.0),
        initial_cash=initial_cash,
        first_decision=calentamiento,
    )
    techos = validacion.ceilings(
        safety=safety,
        cash_rate=cash_rate,
        initial_cash=initial_cash,
        first_decision=calentamiento,
    )

    corridas: list[SeedRun] = []
    for semilla in seeds:
        if policy_factory is None:
            politica: Policy = train_ppo(
                lambda: construir(train), ppo_config, seed=semilla
            )
        else:
            politica = policy_factory(train, semilla)
        salida = run_policy(construir(validacion), politica, seed=semilla)
        reporte = evaluate_run(salida.result)
        corridas.append(_seed_run(semilla, salida, reporte, techos, initial_cash))

    distribuciones = _distribuciones(
        tuple(corridas), allow_fewer_seeds=allow_fewer_seeds
    )

    referencias = run_baselines(
        validacion.series, config_sim, warmup=calentamiento, seed=seeds[0]
    )
    return ArmResult(
        label=label,
        fixture_name=fixture.name,
        fixture_config=fixture.describe(),
        ppo_config=ppo_config.describe(),
        runs=tuple(corridas),
        distributions=distribuciones,
        ceiling=_ceiling_summary(techos, initial_cash, memorizador),
        baselines={k: _baseline_summary(v) for k, v in referencias.items()},
    )


def _distribuciones(
    corridas: tuple[SeedRun, ...], *, allow_fewer_seeds: bool
) -> dict[str, SeedDistribution]:
    return {
        metrica: summarize(
            metrica,
            [c.seed for c in corridas],
            [getattr(c, metrica) for c in corridas],
            allow_fewer_seeds=allow_fewer_seeds,
        )
        for metrica in _METRICAS
    }


def _memorizer_curve(
    parent: Fixture,
    segment: Fixture,
    *,
    safety: float,
    rate_per_bar: float,
    initial_cash: float,
    first_decision: int,
) -> FloatArray | None:
    """La regla del regimen **viejo** aplicada al tramo de evaluacion.

    Es la referencia contra la que se mide "memoriza": exactamente lo que hace un
    agente que aprendio la primera mitad y no noto el cambio. Sin este numero, el
    nivel 3 reporta que el agente rindio mal y no puede decir **por que**: rendir
    mal por no haber aprendido nada y rendir mal por aplicar con conviccion la
    regla anterior son dos diagnosticos distintos, y solo el segundo es
    "memorizacion".

    Devuelve ``None`` cuando el fixture no tiene cambio de regimen, que es donde
    la referencia no significa nada.
    """
    if parent.spec.flip_at is None:
        return None
    viejo = replace(
        segment.spec,
        beta=parent.spec.beta,
        beta_after_flip=None,
        flip_at=None,
    )
    estados = myopic_states(
        viejo.conditional_mean(segment.signal),
        np.full(len(segment), viejo.sigma_for(parent.spec.beta)),
        safety=safety,
        rate_per_bar=rate_per_bar,
        first_decision=first_decision,
    )
    return evaluate_states(
        np.asarray(segment.series.close, dtype=np.float64),
        estados,
        initial_cash=initial_cash,
        safety=safety,
        rate_per_bar=rate_per_bar,
        costs=segment.costs,
        first_decision=first_decision,
    )


def _ceiling_summary(
    ceilings: Ceilings, initial_cash: float, memorizer: FloatArray | None = None
) -> dict[str, Any]:
    resumen = ceilings.describe()
    resumen["always_long_log_growth"] = float(
        np.log(ceilings.always_long[-1] / ceilings.always_long[0])
    )
    resumen["informed_log_growth"] = float(
        np.log(ceilings.informed[-1] / ceilings.informed[0])
    )
    resumen["initial_cash"] = initial_cash
    if memorizer is not None:
        resumen["memorizer_total_return"] = float(memorizer[-1] / memorizer[0] - 1.0)
        resumen["memorizer_capture"] = ceilings.capture(memorizer)
    return resumen


def _baseline_summary(result: SimResult) -> dict[str, Any]:
    reporte = evaluate_run(result)
    return {
        "total_return_mark": reporte.mark.total_return,
        "total_return_liquidation": reporte.liquidation.total_return,
        "sharpe_mark": reporte.mark.sharpe,
        "max_drawdown_mark": reporte.mark.drawdown.depth,
        "turnover_annualized": reporte.turnover.annualized,
        "costs_paid": reporte.costs.total_costs_paid,
        "gap": reporte.costs.gap,
        "n_round_trips": reporte.trades.n_round_trips,
        "win_rate": reporte.trades.win_rate,
        "friction_gap": reporte.friction.final_gap_pct,
    }


# ---------------------------------------------------------------------------
# Los cinco niveles
#
# Cada evaluador recibe brazos ya corridos y solo aplica el criterio. Separar
# "correr" de "juzgar" permite testear los criterios sin entrenar nada, que es
# lo que hace posible verificar que el protocolo **falla** cuando tiene que
# fallar: un criterio que solo se ejercita con agentes que pasan no esta testeado.
# ---------------------------------------------------------------------------


def evaluate_level_0(arm: ArmResult, thresholds: ProtocolThresholds) -> LevelResult:
    """Sin ruido, el agente tiene que acercarse al optimo analitico.

    Si falla, **el bug esta en el pipeline**: la senal es una feature cruda de la
    observacion y el optimo es una funcion escalon sobre ella. No hay
    hiperparametro que arregle una observacion mal armada, un reward
    desconectado del equity o un sizer que no llega al venue.
    """
    criterio = (
        f"la mediana de capture sobre >= 10 semillas supera "
        f"{thresholds.level_0_min_capture:.2f}"
    )
    mediana = arm.median("capture")
    if mediana is None:
        return LevelResult(
            0,
            "senal determinista",
            Verdict.FAIL,
            criterio,
            "capture indefinido: el techo coincide con estar siempre invertido, "
            "lo cual no puede pasar en el nivel 0 y delata un fixture mal armado",
            (arm,),
        )
    ok = mediana >= thresholds.level_0_min_capture
    hallazgo = (
        f"mediana de capture {mediana:.3f} "
        f"(p25 {arm.distributions['capture'].p25:.3f}, "
        f"p75 {arm.distributions['capture'].p75:.3f}); "
        f"saturacion mediana {arm.median('saturation'):.2f}"
    )
    if not ok:
        hallazgo += (
            ". PARAR: el problema esta en el pipeline, no en los "
            "hiperparametros. No pasar al nivel 1."
        )
    return LevelResult(
        0,
        "senal determinista",
        Verdict.PASS if ok else Verdict.FAIL,
        criterio,
        hallazgo,
        (arm,),
    )


def evaluate_level_1(
    arms: Sequence[ArmResult], thresholds: ProtocolThresholds
) -> LevelResult:
    """Con ruido, el desempeno tiene que degradar **suavemente** al bajar el SNR.

    Se compara ``capture`` y no el retorno crudo: al bajar el SNR **el techo
    tambien baja**, asi que una caida del retorno seria compatible con un agente
    que captura exactamente la misma fraccion de lo capturable. Sin normalizar,
    la degradacion se mediria a si misma.

    Los brazos se ordenan por SNR decreciente (``beta^2``), que es el orden en
    que se espera que caiga la fraccion capturada.
    """
    criterio = (
        "ordenados por SNR decreciente, la mediana de capture no cae mas de "
        f"{thresholds.level_1_monotonia_tol:.2f} al subir el SNR (degradacion "
        f"suave), y el brazo de mayor SNR supera "
        f"{thresholds.level_1_min_capture_highest_snr:.2f}"
    )

    def snr(arm: ArmResult) -> float:
        especificacion: dict[str, Any] = arm.fixture_config["spec"]
        return float(especificacion["r_squared"])

    ordenados = sorted(arms, key=snr, reverse=True)
    medianas = [(a.label, a.median("capture")) for a in ordenados]
    if any(m is None for _, m in medianas):
        return LevelResult(
            1,
            "senal con ruido (barrido de SNR)",
            Verdict.FAIL,
            criterio,
            f"capture indefinido en algun brazo: {medianas}",
            tuple(ordenados),
        )
    valores = [float(m) for _, m in medianas if m is not None]
    inversiones = [
        (medianas[i][0], medianas[i + 1][0], valores[i + 1] - valores[i])
        for i in range(len(valores) - 1)
        if valores[i + 1] - valores[i] > thresholds.level_1_monotonia_tol
    ]
    mejor_ok = valores[0] >= thresholds.level_1_min_capture_highest_snr
    ok = not inversiones and mejor_ok
    detalle = ", ".join(
        f"{lab}={v:.3f}" for (lab, _), v in zip(medianas, valores, strict=True)
    )
    hallazgo = f"capture mediano por SNR decreciente: {detalle}"
    if inversiones:
        hallazgo += f". Inversiones fuera de tolerancia: {inversiones}"
    if not mejor_ok:
        hallazgo += (
            f". El brazo de mayor SNR captura {valores[0]:.3f}, por debajo del "
            f"minimo {thresholds.level_1_min_capture_highest_snr:.2f}"
        )
    return LevelResult(
        1,
        "senal con ruido (barrido de SNR)",
        Verdict.PASS if ok else Verdict.FAIL,
        criterio,
        hallazgo,
        tuple(ordenados),
    )


def evaluate_level_2(
    with_costs: ArmResult, without_costs: ArmResult, thresholds: ProtocolThresholds
) -> LevelResult:
    """Con costos, el agente tiene que **rotar menos**, no solo rendir menos.

    Rendir menos es automatico: los costos se restan del equity aunque el agente
    los ignore por completo. Lo que prueba que la penalizacion llego al reward es
    el cambio de **comportamiento**, y por eso el criterio esta sobre la rotacion
    y no sobre el retorno.

    La comparacion es contra el mismo ``beta`` sin costos, no contra un numero
    absoluto: la rotacion depende del proceso.
    """
    criterio = (
        f"la rotacion anualizada mediana cae al menos "
        f"{thresholds.level_2_min_turnover_drop:.0%} respecto del mismo proceso "
        "sin costos"
    )
    con = with_costs.median("turnover_annualized")
    sin = without_costs.median("turnover_annualized")
    if con is None or sin is None or sin <= 0:
        return LevelResult(
            2,
            "senal con costos",
            Verdict.FAIL,
            criterio,
            f"rotacion no comparable: con costos {con}, sin costos {sin}",
            (with_costs, without_costs),
        )
    caida = 1.0 - con / sin
    ok = caida >= thresholds.level_2_min_turnover_drop
    hallazgo = (
        f"rotacion anualizada mediana {con:.2f} con costos contra {sin:.2f} sin "
        f"costos: caida del {caida:.1%}. Capture mediano {with_costs.median('capture')}"
    )
    if not ok:
        hallazgo += (
            ". El agente opera practicamente igual con y sin costos: la "
            "penalizacion no esta llegando al reward."
        )
    return LevelResult(
        2,
        "senal con costos",
        Verdict.PASS if ok else Verdict.FAIL,
        criterio,
        hallazgo,
        (with_costs, without_costs),
    )


def evaluate_level_3(arms: Sequence[ArmResult]) -> LevelResult:
    """Cambio de regimen: **se mide, no se aprueba**.

    Adaptarse y memorizar son dos hallazgos distintos y los dos son validos. Lo
    que no es valido es no medirlo. La referencia es ``Ceilings.memorizer``: la
    regla optima de la primera mitad aplicada a toda la serie, que es
    exactamente lo que hace un agente que memorizo.

    El veredicto es ``MEASURED`` a proposito. Poner un umbral aqui seria
    inventar una hipotesis despues del hecho.
    """
    criterio = (
        "sin criterio de aprobacion: se reporta si el agente se adapta o "
        "memoriza, contra la referencia del memorizador del fixture"
    )
    partes = []
    for arm in arms:
        capture = arm.median("capture")
        referencia = arm.ceiling.get("memorizer_capture")
        veredicto = "sin referencia de memorizador"
        if capture is not None and referencia is not None:
            # El memorizador tiene capture muy negativo: aplicar la regla vieja
            # al regimen nuevo destruye capital. Un agente que se le acerca esta
            # memorizando; uno que se acerca a 1 se adapto.
            veredicto = (
                "MEMORIZA"
                if capture < float(referencia) / 2.0
                else "se adapta parcialmente"
                if capture < 0.5
                else "SE ADAPTA"
            )
        partes.append(
            f"{arm.label}: {veredicto} (capture "
            f"{'n/a' if capture is None else f'{capture:+.3f}'}, memorizador "
            f"{'n/a' if referencia is None else f'{float(referencia):+.3f}'})"
        )
    return LevelResult(
        3,
        "cambio de regimen",
        Verdict.MEASURED,
        criterio,
        "; ".join(partes),
        tuple(arms),
    )


def assemble_protocol(
    thresholds: ProtocolThresholds,
    *,
    level_0: ArmResult,
    level_1: Sequence[ArmResult] | None = None,
    level_2: ArmResult | None = None,
    level_2_reference: ArmResult | None = None,
    level_3: Sequence[ArmResult] | None = None,
    level_4a: MultiPathResult | None = None,
    level_4b: MultiPathResult | None = None,
) -> ProtocolReport:
    """Arma el reporte aplicando los criterios en orden y **parando al fallar**.

    Los niveles que no se corrieron -porque uno anterior fallo- quedan como
    ``SKIPPED`` con su motivo. Que aparezcan en el reporte en vez de faltar es
    deliberado: un reporte con tres niveles y sin explicacion se lee como si el
    protocolo tuviera tres niveles.
    """
    reporte = ProtocolReport(thresholds=thresholds)

    def saltear(nivel: int, etiqueta: str, motivo: str) -> None:
        reporte.levels.append(
            LevelResult(nivel, etiqueta, Verdict.SKIPPED, "no se evaluo", motivo)
        )

    pendientes = (
        (1, "senal con ruido (barrido de SNR)"),
        (2, "senal con costos"),
        (3, "cambio de regimen"),
        (4, "4a control negativo puro (Heston)"),
        (4, "4b drift detectable (Heston)"),
    )

    cero = evaluate_level_0(level_0, thresholds)
    reporte.levels.append(cero)
    if cero.verdict is Verdict.FAIL:
        for nivel, etiqueta in pendientes:
            saltear(nivel, etiqueta, "el nivel 0 fallo: el bug esta en el pipeline")
        return reporte

    if level_1 is None:
        return reporte
    uno = evaluate_level_1(level_1, thresholds)
    reporte.levels.append(uno)
    if uno.verdict is Verdict.FAIL:
        for nivel, etiqueta in pendientes[1:]:
            saltear(nivel, etiqueta, "el nivel 1 fallo")
        return reporte

    if level_2 is None or level_2_reference is None:
        return reporte
    dos = evaluate_level_2(level_2, level_2_reference, thresholds)
    reporte.levels.append(dos)
    if dos.verdict is Verdict.FAIL:
        for nivel, etiqueta in pendientes[2:]:
            saltear(nivel, etiqueta, "el nivel 2 fallo")
        return reporte

    if level_3 is None:
        return reporte
    reporte.levels.append(evaluate_level_3(level_3))

    if level_4a is None:
        return reporte
    cuatro_a = evaluate_level_4a(level_4a, thresholds)
    reporte.levels.append(cuatro_a)
    if cuatro_a.verdict is Verdict.FAIL:
        saltear(
            4,
            "4b drift detectable (Heston)",
            "el nivel 4a fallo: el agente inventa senal donde no la hay, y medir "
            "si reconoce drift real solo tendria sentido despues de entender eso",
        )
        return reporte

    if level_4b is None:
        return reporte
    reporte.levels.append(evaluate_level_4b(level_4b, thresholds))
    return reporte


# ---------------------------------------------------------------------------
# Brazos sobre multiples caminos
#
# Un fixture sintetico se puede muestrear: cambiar la semilla del proceso da
# otro camino del **mismo** mercado. Eso permite separar la varianza de mercado
# de la de entrenamiento, y esa separacion es la que faltaba cuando el nivel 4
# reporto un exceso de +0.164 sobre un camino cuyo baseline tiene desvio 0.781
# entre caminos.
#
# En datos reales no se puede: la historia es un solo camino y N=1 por
# construccion. La asimetria no se arregla, se declara -y es la razon de fondo
# por la que ahi el walk-forward fuera de muestra es la unica defensa-.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiPathResult:
    """Un nivel corrido sobre ``N`` caminos independientes x ``M`` semillas."""

    label: str
    path_seeds: tuple[int, ...]
    agent_seeds: tuple[int, ...]
    arms: tuple[ArmResult, ...]
    decompositions: dict[str, VarianceDecomposition]
    drift_t_by_path: tuple[float, ...]
    ppo_config: dict[str, Any]

    @property
    def n_paths(self) -> int:
        return len(self.arms)

    @property
    def drift_detectable(self) -> bool:
        """¿El drift del proceso es detectable en la ventana de entrenamiento?

        Se contrasta la **mediana** de los ``t`` por camino contra el valor
        critico normal al 95%. Si no lo es, exigirle al agente que lo aprenda es
        exigirle que extraiga algo que la muestra no contiene, y el criterio del
        nivel hay que leerlo con eso adelante.
        """
        return bool(abs(float(np.median(self.drift_t_by_path))) > 1.96)

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_paths": self.n_paths,
            "path_seeds": list(self.path_seeds),
            "agent_seeds": list(self.agent_seeds),
            "ppo": self.ppo_config,
            "drift_t_by_path": list(self.drift_t_by_path),
            "drift_t_median": float(np.median(self.drift_t_by_path)),
            "drift_detectable": self.drift_detectable,
            "decompositions": {k: v.describe() for k, v in self.decompositions.items()},
            "arms": [a.describe() for a in self.arms],
        }

    @classmethod
    def from_dict(cls, datos: dict[str, Any]) -> MultiPathResult:
        brazos = tuple(ArmResult.from_dict(a) for a in datos["arms"])
        return cls(
            label=str(datos["label"]),
            path_seeds=tuple(int(x) for x in datos["path_seeds"]),
            agent_seeds=tuple(int(x) for x in datos["agent_seeds"]),
            arms=brazos,
            decompositions=_descomponer(brazos),
            drift_t_by_path=tuple(float(x) for x in datos["drift_t_by_path"]),
            ppo_config=dict(datos["ppo"]),
        )


#: Metricas que se descomponen en varianza de mercado y de entrenamiento.
MULTIPATH_METRICS = (
    "excess_log_growth_vs_always_long",
    "log_growth",
    "total_return_mark",
    "time_invested",
    "action_std",
    "turnover_annualized",
)


def _descomponer(arms: tuple[ArmResult, ...]) -> dict[str, VarianceDecomposition]:
    salida: dict[str, VarianceDecomposition] = {}
    for metrica in MULTIPATH_METRICS:
        por_camino = [[getattr(c, metrica) for c in brazo.runs] for brazo in arms]
        salida[metrica] = decompose_variance(metrica, por_camino)
    return salida


def run_multipath_arm(
    builder: Callable[[int], Fixture],
    *,
    label: str,
    path_seeds: Sequence[int],
    agent_seeds: Sequence[int],
    ppo_config: PPOConfig,
    policy_factory: PolicyFactory | None = None,
    initial_cash: float = 100_000.0,
    cash_rate: float = 0.0,
    observation: ObservationSpec | None = None,
    safety: float = 0.98,
    allow_fewer_seeds: bool = False,
) -> MultiPathResult:
    """Corre el mismo nivel sobre ``N`` caminos, con ``M`` semillas cada uno.

    ``builder`` recibe la semilla del **proceso** y devuelve el fixture. Que sea
    una fabrica y no una lista de fixtures es lo que hace explicito que los
    caminos son del mismo mercado con otra realizacion, y no cinco mercados
    distintos.

    El ``t`` del drift se calcula sobre la ventana de **entrenamiento** de cada
    camino, que es la muestra de la que el agente podria aprenderlo.
    """
    brazos: list[ArmResult] = []
    ts: list[float] = []
    for semilla_camino in path_seeds:
        fixture = builder(semilla_camino)
        train, _, _ = fixture.split()
        ts.append(drift_t_statistic(np.asarray(train.signal)[1:]))
        brazos.append(
            run_arm(
                fixture,
                label=f"{label}:path{semilla_camino}",
                seeds=agent_seeds,
                ppo_config=ppo_config,
                policy_factory=policy_factory,
                initial_cash=initial_cash,
                cash_rate=cash_rate,
                observation=observation,
                safety=safety,
                allow_fewer_seeds=allow_fewer_seeds,
            )
        )
    return MultiPathResult(
        label=label,
        path_seeds=tuple(path_seeds),
        agent_seeds=tuple(agent_seeds),
        arms=tuple(brazos),
        decompositions=_descomponer(tuple(brazos)),
        drift_t_by_path=tuple(ts),
        ppo_config=ppo_config.describe(),
    )


def evaluate_level_4a(
    result: MultiPathResult, thresholds: ProtocolThresholds
) -> LevelResult:
    """Control negativo puro: **una sola hipotesis, un solo criterio**.

    ¿El agente inventa senal donde no la hay? Se contesta con lo unico que se
    puede preguntar sobre una serie sin direccion predecible: si el exceso sobre
    estar invertido se distingue de cero cuando se lo mide contra el ruido del
    sorteo entre caminos.

    **El tiempo invertido ya no entra en el veredicto.** Sobre este fixture el
    drift no es detectable en la ventana de entrenamiento -``t`` poblacional
    0.51- y abstenerse es una respuesta defendible, no un fallo: no hay en la
    muestra nada que lleve al agente a invertirse. Exigir convergencia aqui
    mezclaba esa hipotesis con la del control negativo y hacia que un solo
    veredicto contestara dos preguntas. La segunda vive en el nivel 4b.

    Se sigue reportando como evidencia: un nivel que no mide algo no es un nivel
    que deba ocultarlo.
    """
    criterio = (
        f"sobre >= {thresholds.level_4_min_paths} caminos independientes, el "
        "exceso de crecimiento sobre estar siempre invertido NO se distingue de "
        "cero contrastado con la dispersion entre caminos (t de dos colas al "
        "95%). Unico criterio del nivel"
    )
    if result.n_paths < thresholds.level_4_min_paths:
        return LevelResult(
            4,
            "4a control negativo puro (Heston)",
            Verdict.FAIL,
            criterio,
            f"solo hay {result.n_paths} caminos y el criterio necesita "
            f"{thresholds.level_4_min_paths}: con menos, la varianza de mercado "
            "y la de entrenamiento no se separan",
            result.arms,
        )

    exceso = result.decompositions["excess_log_growth_vs_always_long"]
    invertido = result.decompositions["time_invested"]
    t_drift = float(np.median(result.drift_t_by_path))
    detectable = "DETECTABLE" if result.drift_detectable else "NO detectable"
    ok = not exceso.distinguishable_from_zero
    hallazgo = (
        f"exceso {exceso.render()}. Evidencia que NO entra en el veredicto: "
        f"tiempo invertido medio {invertido.mean:.2f}, t del drift en "
        f"entrenamiento mediana {t_drift:+.2f} ({detectable} al 95%), asi que "
        "abstenerse es defendible en este fixture"
    )
    if not ok:
        hallazgo += (
            ". El exceso se distingue de cero incluso contra la dispersion "
            "entre caminos: el agente extrae algo de una serie sin senal y hay "
            "que entender que antes de seguir."
        )
    return LevelResult(
        4,
        "4a control negativo puro (Heston)",
        Verdict.PASS if ok else Verdict.FAIL,
        criterio,
        hallazgo,
        result.arms,
    )


def evaluate_level_4b(
    result: MultiPathResult, thresholds: ProtocolThresholds
) -> LevelResult:
    """¿Reconoce el agente un drift **cuando existe**?

    Mismo proceso que el 4a salvo por ``mu``, calibrado para que el drift sea
    detectable en la ventana de entrenamiento. Sigue sin haber senal
    direccional, asi que el optimo es estar invertido y quedarse quieto.

    El primer chequeo es **sobre el fixture**: si el ``t`` mediano del drift no
    llega al umbral, el fixture no tiene lo que dice tener y el nivel no puede
    medir lo que pretende. Eso es un fallo de calibracion, no del agente, y el
    veredicto lo dice con esas palabras en vez de cargarselo al agente.
    """
    criterio = (
        f"con el drift detectable en entrenamiento (t mediano >= "
        f"{thresholds.level_4b_min_drift_t:.1f}), el agente converge a estar "
        f"invertido (media entre caminos >= "
        f"{thresholds.level_4b_min_time_invested:.0%}) y no rota (dispersion de "
        f"la accion <= {thresholds.level_4b_max_action_std:.2f})"
    )
    if result.n_paths < thresholds.level_4_min_paths:
        return LevelResult(
            4,
            "4b drift detectable (Heston)",
            Verdict.FAIL,
            criterio,
            f"solo hay {result.n_paths} caminos y el criterio necesita "
            f"{thresholds.level_4_min_paths}",
            result.arms,
        )

    t_drift = float(np.median(result.drift_t_by_path))
    invertido = result.decompositions["time_invested"]
    dispersion = result.decompositions["action_std"]
    rotacion = result.decompositions["turnover_annualized"]

    if t_drift < thresholds.level_4b_min_drift_t:
        return LevelResult(
            4,
            "4b drift detectable (Heston)",
            Verdict.FAIL,
            criterio,
            f"el fixture no esta bien calibrado: t mediano del drift "
            f"{t_drift:+.2f}, por debajo de {thresholds.level_4b_min_drift_t:.1f}. "
            "Es un fallo de calibracion del fixture, no del agente: sin drift "
            "detectable el nivel no puede medir lo que pretende medir",
            result.arms,
        )

    invertido_ok = invertido.mean >= thresholds.level_4b_min_time_invested
    quieto_ok = dispersion.mean <= thresholds.level_4b_max_action_std
    hallazgo = (
        f"t del drift mediana {t_drift:+.2f} (detectable). Tiempo invertido medio "
        f"entre caminos {invertido.mean:.2f} (sigma_mercado "
        f"{invertido.between_path_std:.3f}); dispersion de la accion media "
        f"{dispersion.mean:.3f} (sigma_mercado {dispersion.between_path_std:.3f}); "
        f"rotacion anualizada media {rotacion.mean:.1f}"
    )
    if not invertido_ok:
        hallazgo += (
            ". NO converge a estar invertido pese a que el drift si esta en la "
            "muestra: el agente no reconoce un drift que podria medir."
        )
    if not quieto_ok:
        hallazgo += (
            ". Rota: la dispersion de la accion supera el umbral, asi que no se "
            "queda quieto ni siquiera con el optimo a la vista."
        )
    return LevelResult(
        4,
        "4b drift detectable (Heston)",
        Verdict.PASS if (invertido_ok and quieto_ok) else Verdict.FAIL,
        criterio,
        hallazgo,
        result.arms,
    )
