"""El test que demuestra que el wrapper es delgado.

Una politica determinista ejecutada **a traves del entorno** debe producir
exactamente el mismo log de ordenes, el mismo equity y la misma posicion que esa
misma politica corrida como estrategia directa contra el simulador.

Si el entorno tuviera logica propia de ejecucion, contabilidad, costos o
dimensionamiento, los dos caminos divergirian y este archivo se pondria rojo.

Que prueba y que no: la politica lee solo las columnas de **mercado** de la
observacion. Las cuatro columnas de cuenta y resultado (brecha de friccion,
orden vetada, orden rechazada, fraccion llenada) no entran en la decision,
porque ``Strategy.on_bar`` recibe ``(view, account)`` y no la ``Decision``
completa, asi que una estrategia directa no puede reconstruirlas. Eso no debilita
el test: esas columnas son informacion **adicional** que el entorno entrega al
agente, no logica adicional que el entorno ejecute. Lo que se verifica es que,
dada la misma secuencia de acciones, el camino de ejecucion es identico.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.instruments import InstrumentSpec
from data.schema import BarSeries, FloatArray
from envs.observation import ObservationBuilder
from envs.trading_env import EnvConfig, TradingEnv
from features.scaler import FeatureScaler
from risk.layer import RiskLayer, RiskLimits
from sim.clock import SimulatedClock
from sim.engine import Decision, SimConfig, Simulator
from sim.orders import MarketOrder
from sim.sizing import TargetWeightSizer
from sim.view import AccountSnapshot, MarketView


class PoliticaDeterminista:
    """Peso objetivo en funcion de las columnas de mercado. Sin estado.

    La forma concreta no importa; lo que importa es que sea determinista y que
    dependa de la observacion, para que el test recorra caminos distintos segun
    la serie en vez de mandar siempre lo mismo.
    """

    def __init__(self, n_mercado: int) -> None:
        self.n_mercado = n_mercado

    def __call__(self, obs: FloatArray) -> float:
        mercado = obs[: self.n_mercado]
        senal = float(np.tanh(np.sum(mercado) / max(self.n_mercado, 1)))
        return float(min(max(0.5 * (senal + 1.0), 0.0), 1.0))


class AdaptadorEstrategia:
    """La misma politica, corrida como ``Strategy`` directa contra el motor.

    Construye la observacion con el **mismo** ``ObservationBuilder`` y el mismo
    escalador, y dimensiona con el **mismo** ``TargetWeightSizer``. Si el entorno
    usara otros, este adaptador y el entorno divergirian.
    """

    name = "politica"

    def __init__(
        self,
        politica: PoliticaDeterminista,
        builder: ObservationBuilder,
        scaler: FeatureScaler,
        sizer: TargetWeightSizer,
        warmup: int,
    ) -> None:
        self.politica = politica
        self.builder = builder
        self.scaler = scaler
        self.sizer = sizer
        self.warmup = warmup

    def reset(self, seed: int | None = None) -> None:
        return None

    def on_bar(self, view: MarketView, account: AccountSnapshot) -> MarketOrder | None:
        if view.t < self.warmup:
            # El entorno envia None durante el calentamiento; la estrategia
            # directa tiene que hacer lo mismo o los caminos no son comparables.
            return None
        decision = Decision(view=view, account=account, equity_liquidation=0.0)
        obs = self.scaler.transform(self.builder.build(decision))
        return self.sizer.order_for(self.politica(obs), view, account)


def _n_columnas_de_mercado(builder: ObservationBuilder) -> int:
    return sum(1 for escalar in builder.scale_mask if escalar)


def _correr_por_el_entorno(
    serie: BarSeries,
    sim_config: SimConfig,
    scaler: FeatureScaler,
    builder: ObservationBuilder,
    politica: PoliticaDeterminista,
    gate_factory=None,
):
    env = TradingEnv(
        serie,
        sim_config,
        scaler=scaler,
        env_config=EnvConfig(observation=builder.spec),
        gate_factory=gate_factory,
    )
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminado, truncado, _ = env.step(np.array([politica(obs)]))
        if terminado or truncado:
            break
    assert env.result is not None
    return env.result


def _correr_directo(
    serie: BarSeries,
    sim_config: SimConfig,
    scaler: FeatureScaler,
    builder: ObservationBuilder,
    politica: PoliticaDeterminista,
    gate=None,
):
    estrategia = AdaptadorEstrategia(
        politica, builder, scaler, TargetWeightSizer(), builder.spec.warmup
    )
    return Simulator(serie, sim_config).run(estrategia, gate=gate)


class TestEquivalencia:
    def test_el_log_de_ordenes_es_identico(
        self, serie, sim_config, scaler, builder
    ) -> None:
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        por_env = _correr_por_el_entorno(serie, sim_config, scaler, builder, politica)
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)

        assert len(por_env.fills) == len(directo.fills)
        assert por_env.fills == directo.fills

    def test_hubo_operaciones_de_verdad(
        self, serie, sim_config, scaler, builder
    ) -> None:
        """Sin esto, dos logs vacios harian pasar el test de arriba."""
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)
        assert len(directo.executed_fills) > 10

    def test_las_series_de_equity_son_identicas(
        self, serie, sim_config, scaler, builder
    ) -> None:
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        por_env = _correr_por_el_entorno(serie, sim_config, scaler, builder, politica)
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)

        np.testing.assert_array_equal(por_env.equity, directo.equity)
        np.testing.assert_array_equal(
            por_env.equity_liquidation, directo.equity_liquidation
        )
        np.testing.assert_array_equal(por_env.position, directo.position)
        np.testing.assert_array_equal(por_env.cash, directo.cash)

    def test_el_desglose_de_costos_es_identico(
        self, serie, sim_config, scaler, builder
    ) -> None:
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        por_env = _correr_por_el_entorno(serie, sim_config, scaler, builder, politica)
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)
        assert por_env.cost_breakdown() == directo.cost_breakdown()

    def test_los_identificadores_de_orden_coinciden(
        self, serie, sim_config, scaler, builder
    ) -> None:
        """Mismo generador determinista, misma secuencia, mismos IDs."""
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        por_env = _correr_por_el_entorno(serie, sim_config, scaler, builder, politica)
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)
        assert [f.client_order_id for f in por_env.fills] == [
            f.client_order_id for f in directo.fills
        ]

    def test_tambien_con_instrumento_con_minimos(
        self, spec_con_minimos: InstrumentSpec, sim_config, builder
    ) -> None:
        """Con rechazos por minimos del venue la equivalencia tiene que aguantar.

        Es el caso interesante: si el entorno pre-redondeara o se autocensurara,
        aca aparecerian menos rechazos que en la corrida directa.
        """
        from envs.observation import fit_scaler_on_train

        from .conftest import serie_deterministica

        serie = serie_deterministica(spec_con_minimos, n=120, seed=5)
        scaler = fit_scaler_on_train(serie.slice(0, 80), builder)
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))

        por_env = _correr_por_el_entorno(serie, sim_config, scaler, builder, politica)
        directo = _correr_directo(serie, sim_config, scaler, builder, politica)
        assert por_env.fills == directo.fills
        assert len(por_env.rejected_fills) == len(directo.rejected_fills)


class TestEquivalenciaConCapaDeRiesgo:
    """El entorno no puentea la capa de riesgo: la pasa al mismo punto."""

    LIMITES = RiskLimits(max_position_pct_equity=0.40, max_daily_turnover_ratio=0.30)

    def _gate(self) -> RiskLayer:
        return RiskLayer(
            self.LIMITES, SimulatedClock(np.datetime64("2020-01-01T21:00:00", "ns"))
        )

    def test_los_vetos_son_identicos(self, serie, sim_config, scaler, builder) -> None:
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        por_env = _correr_por_el_entorno(
            serie, sim_config, scaler, builder, politica, gate_factory=self._gate
        )
        directo = _correr_directo(
            serie, sim_config, scaler, builder, politica, gate=self._gate()
        )

        assert por_env.gate_rejections == directo.gate_rejections
        assert por_env.fills == directo.fills
        np.testing.assert_array_equal(por_env.equity, directo.equity)

    def test_la_capa_veto_algo(self, serie, sim_config, scaler, builder) -> None:
        """Un limite que nunca muerde no probaria nada."""
        politica = PoliticaDeterminista(_n_columnas_de_mercado(builder))
        directo = _correr_directo(
            serie, sim_config, scaler, builder, politica, gate=self._gate()
        )
        assert len(directo.gate_rejections) > 0


class TestSizerCompartido:
    def test_el_entorno_no_tiene_su_propio_sizer(self) -> None:
        """El dimensionamiento vive en sim/, no en envs/.

        Si alguien reimplementara la conversion peso -> cantidad dentro del
        entorno, este test no lo detectaria por si solo; los de equivalencia si.
        Este fija la intencion por si el codigo cambia de forma.
        """
        import envs.trading_env as modulo

        fuente = modulo.__file__
        assert fuente is not None
        with open(fuente, encoding="utf-8") as f:
            texto = f.read()
        # El entorno delega: usa el sizer, no calcula cantidades.
        assert "sizer.order_for" in texto
        assert "max_affordable_qty" not in texto
        assert "round_qty" not in texto

    def test_el_sizer_no_pre_redondea_a_cero(self, serie, spec) -> None:
        """Un delta chico se envia igual; lo rechaza el venue y queda en el log."""
        sizer = TargetWeightSizer()
        view = MarketView(serie, 30)
        cuenta = AccountSnapshot(
            t=30,
            cash=1_000.0,
            position=0.0,
            mark_price=view.close(),
            equity=1_000.0,
            pending_qty=0.0,
        )
        # Peso minusculo: la cantidad resultante es diminuta pero no cero.
        orden = sizer.order_for(1e-6, view, cuenta)
        assert orden is not None
        assert 0.0 < orden.qty < 1e-3

    def test_el_sizer_rechaza_pesos_fuera_de_rango(self, serie) -> None:
        sizer = TargetWeightSizer()
        view = MarketView(serie, 30)
        cuenta = AccountSnapshot(
            t=30,
            cash=1_000.0,
            position=0.0,
            mark_price=view.close(),
            equity=1_000.0,
            pending_qty=0.0,
        )
        with pytest.raises(ValueError, match="fuera de"):
            sizer.order_for(1.5, view, cuenta)
