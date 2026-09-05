"""Anti-leakage del entorno Gymnasium.

El generador ``Simulator.drive`` impide adelantar el cursor sin decidir, pero no
puede impedir que el codigo del entorno tenga la ``BarSeries``: la necesita para
construir el simulador. La garantia real esta en el constructor de la
observacion, que recibe solo una ``Decision``.

El test central de este archivo lo demuestra de forma empirica y no por lectura
del codigo: dos series **identicas hasta t y distintas despues** tienen que
producir observaciones bit-identicas en todos los pasos hasta t. Si la
observacion dependiera de una sola barra futura, divergirian.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.errors import LookaheadError
from data.schema import BarSeries
from envs.observation import ObservationBuilder, fit_scaler_on_train
from envs.trading_env import TradingEnv
from sim.engine import Decision, Simulator
from sim.view import AccountSnapshot, MarketView

from ..sim.conftest import make_series

# Job propio y visible del CI: si el entorno filtra informacion del futuro,
# todos los resultados de las Etapas 3 en adelante son ruido.
pytestmark = pytest.mark.leakage

BIFURCACION = 90


def _par_de_series(spec, n: int = 120, seed: int = 3) -> tuple[BarSeries, BarSeries]:
    """Dos series identicas hasta ``BIFURCACION`` y muy distintas despues."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    opens = closes * (1.0 + rng.normal(0.0, 0.002, n))

    closes_b = closes.copy()
    opens_b = opens.copy()
    # A partir de la bifurcacion, el otro camino se desploma. Si algo de la
    # observacion mirara adelante, se notaria.
    closes_b[BIFURCACION:] *= np.linspace(1.0, 0.3, n - BIFURCACION)
    opens_b[BIFURCACION:] *= np.linspace(1.0, 0.3, n - BIFURCACION)

    return (
        make_series(spec, closes, opens=opens, volume=200_000.0),
        make_series(spec, closes_b, opens=opens_b, volume=200_000.0),
    )


class TestLaObservacionNoDependeDelFuturo:
    def test_las_series_realmente_divergen(self, spec) -> None:
        """Premisa del test siguiente: sin divergencia no probaria nada."""
        a, b = _par_de_series(spec)
        np.testing.assert_allclose(a.close[:BIFURCACION], b.close[:BIFURCACION])
        assert not np.allclose(a.close[BIFURCACION:], b.close[BIFURCACION:])

    def test_observaciones_identicas_hasta_la_bifurcacion(
        self, spec, sim_config, builder
    ) -> None:
        """El test central del archivo.

        Se comparan los pasos cuya barra de decision es anterior a la
        bifurcacion. La barra ``t`` de la bifurcacion ya es legitimamente
        distinta, porque su propio ``open`` cambio.
        """
        a, b = _par_de_series(spec)
        scaler = fit_scaler_on_train(a.slice(0, 80), builder)

        env_a = TradingEnv(a, sim_config, scaler=scaler)
        env_b = TradingEnv(b, sim_config, scaler=scaler)
        obs_a, info_a = env_a.reset(seed=0)
        obs_b, info_b = env_b.reset(seed=0)

        assert info_a["t"] == info_b["t"] < BIFURCACION
        np.testing.assert_array_equal(obs_a, obs_b)

        comparados = 0
        while True:
            accion = np.array([0.5])
            obs_a, _, ta, tra, info_a = env_a.step(accion)
            obs_b, _, tb, trb, info_b = env_b.step(accion)
            if info_a["t"] >= BIFURCACION:
                break
            # Bit a bit: no `allclose`. Una diferencia de redondeo tambien seria
            # dependencia del futuro.
            np.testing.assert_array_equal(obs_a, obs_b)
            comparados += 1
            if ta or tra or tb or trb:
                break

        assert comparados > 20, "hay que comparar suficientes pasos"

    def test_el_constructor_solo_recibe_la_decision(self, spec, builder) -> None:
        """``build`` no tiene acceso a la serie: su unica entrada es Decision."""
        import inspect

        firma = inspect.signature(ObservationBuilder.build)
        assert list(firma.parameters) == ["self", "decision"]
        assert firma.parameters["decision"].annotation in (Decision, "Decision")


class TestLaVistaDelEntornoRechazaElFuturo:
    def test_la_decision_lleva_una_vista_acotada(self, serie, sim_config) -> None:
        sim = Simulator(serie, sim_config)
        gen = sim.drive()
        decision = next(gen)
        assert len(decision.view) == 1
        with pytest.raises(LookaheadError):
            decision.view.close(-1)
        gen.close()

    def test_el_entorno_no_expone_la_serie_a_la_observacion(
        self, spec, sim_config, builder
    ) -> None:
        """Corromper la serie mas alla de t no cambia la observacion en t.

        Ataque directo: se construye la observacion en ``t`` a partir de una
        vista tomada sobre una serie cuyos valores posteriores son absurdos.
        """
        a, _ = _par_de_series(spec)
        t = 60
        vista = MarketView(a, t)
        cuenta = AccountSnapshot(
            t=t,
            cash=100.0,
            position=0.0,
            mark_price=vista.close(),
            equity=100.0,
            pending_qty=0.0,
        )
        antes = builder.build(
            Decision(view=vista, account=cuenta, equity_liquidation=0.0)
        )

        rng = np.random.default_rng(1)
        closes = np.asarray(a.close, dtype=np.float64).copy()
        opens = np.asarray(a.open, dtype=np.float64).copy()
        # Solo a partir de t+1. Tocar la barra t cambiaria su high y su low, que
        # son informacion legitima del presente: el test dejaria de medir
        # dependencia del futuro y mediria un error de construccion.
        closes[t + 1 :] = rng.uniform(1.0, 10_000.0, len(closes) - t - 1)
        opens[t + 1 :] = rng.uniform(1.0, 10_000.0, len(opens) - t - 1)
        corrupta = make_series(a.instrument, closes, opens=opens, volume=200_000.0)
        vista_corrupta = MarketView(corrupta, t)
        cuenta_corrupta = AccountSnapshot(
            t=t,
            cash=100.0,
            position=0.0,
            mark_price=vista_corrupta.close(),
            equity=100.0,
            pending_qty=0.0,
        )
        despues = builder.build(
            Decision(
                view=vista_corrupta, account=cuenta_corrupta, equity_liquidation=0.0
            )
        )
        np.testing.assert_array_equal(antes, despues)


class TestElCalentamientoNoAdelanta:
    def test_el_episodio_empieza_despues_del_calentamiento(
        self, serie, sim_config, scaler, builder
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        _, info = env.reset(seed=0)
        assert info["t"] == builder.spec.warmup

    def test_durante_el_calentamiento_no_se_opera(
        self, serie, sim_config, scaler, builder
    ) -> None:
        """Las barras del calentamiento existen en el log, sin fills.

        Es la verdad: el agente no decidio nada porque sus indicadores todavia no
        estaban definidos. Saltearlas de la serie seria otra cosa.
        """
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        while True:
            _, _, terminado, truncado, _ = env.step(np.array([1.0]))
            if terminado or truncado:
                break
        resultado = env.result
        assert resultado is not None
        primeros = [f for f in resultado.fills if f.t_decision < builder.spec.warmup]
        assert primeros == []


class TestElEscaladorNoVeElTest:
    def test_ajustar_en_train_no_toca_estadisticas_de_test(self, spec, builder) -> None:
        """El escalador ajustado en train no puede cambiar si el test cambia."""
        a, b = _par_de_series(spec)
        corte = BIFURCACION - 10  # el train termina antes de la bifurcacion
        escalador_a = fit_scaler_on_train(a.slice(0, corte), builder)
        escalador_b = fit_scaler_on_train(b.slice(0, corte), builder)
        np.testing.assert_array_equal(escalador_a.mean, escalador_b.mean)
        np.testing.assert_array_equal(escalador_a.std, escalador_b.std)

    def test_ajustar_sobre_la_serie_entera_si_cambia(self, spec, builder) -> None:
        """Premisa del test anterior: si el train incluyera el futuro, cambiaria."""
        a, b = _par_de_series(spec)
        completo_a = fit_scaler_on_train(a, builder)
        completo_b = fit_scaler_on_train(b, builder)
        assert not np.allclose(completo_a.mean, completo_b.mean)
