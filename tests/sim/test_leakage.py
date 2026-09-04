"""Anti-leakage. El test mas importante de la Etapa 1.

La pregunta no es si una estrategia razonable lee el futuro por accidente, sino
si una estrategia que lo intenta a proposito puede conseguirlo. Cada test de
aqui es un intento distinto de romper la barrera.
"""

from __future__ import annotations

import contextlib
from typing import ClassVar

import numpy as np
import pytest

from data.errors import LookaheadError
from data.instruments import InstrumentSpec
from sim.engine import SimConfig, Simulator
from sim.orders import MarketOrder
from sim.view import AccountSnapshot, MarketView

from .conftest import make_series

# Job propio en CI: el fallo de un anti-leakage no puede quedar como una
# linea mas entre cientos de tests verdes.
pytestmark = pytest.mark.leakage


class _Espia:
    """Estrategia que registra lo que consigue ver y lo que le fue negado."""

    name = "espia"

    def __init__(self, intento) -> None:
        self.intento = intento
        self.errores: list[Exception] = []
        self.vistos: list[float] = []

    def reset(self, seed: int | None = None) -> None:
        self.errores = []
        self.vistos = []

    def on_bar(self, view: MarketView, account: AccountSnapshot) -> MarketOrder | None:
        try:
            self.vistos.append(self.intento(view))
        except Exception as exc:
            self.errores.append(exc)
        return None


def _correr(intento, series) -> _Espia:
    espia = _Espia(intento)
    Simulator(series, SimConfig(initial_cash=10_000.0)).run(espia)
    return espia


@pytest.fixture
def series(equity: InstrumentSpec):
    return make_series(equity, [100.0, 110.0, 120.0, 130.0, 140.0])


class TestAccesoDirectoAlFuturo:
    def test_lookback_negativo_lanza_lookahead_error(self, series) -> None:
        """El error clasico: pedir la barra siguiente con un indice negativo."""
        espia = _correr(lambda v: v.close(-1), series)
        assert len(espia.errores) == len(series)
        assert all(isinstance(e, LookaheadError) for e in espia.errores)
        assert espia.vistos == []

    def test_lookback_mas_alla_del_inicio_lanza_index_error(self, series) -> None:
        espia = _correr(lambda v: v.close(10), series)
        assert all(isinstance(e, IndexError) for e in espia.errores)

    def test_history_no_puede_exceder_la_ventana(self, series) -> None:
        espia = _correr(lambda v: v.history("close", 100), series)
        assert all(isinstance(e, IndexError) for e in espia.errores)

    def test_la_ventana_termina_exactamente_en_t(self, series) -> None:
        espia = _correr(lambda v: (len(v), v.close()), series)
        assert espia.vistos == [
            (1, 100.0),
            (2, 110.0),
            (3, 120.0),
            (4, 130.0),
            (5, 140.0),
        ]


class TestNavegacionPorElArraySubyacente:
    def test_history_devuelve_una_copia_sin_base(self, series) -> None:
        """Un slice de numpy expone ``.base``; una copia no.

        Sin esto, ``view.history('close', 1).base`` entregaria la serie entera,
        futuro incluido.
        """
        espia = _correr(lambda v: v.history("close", 1).base, series)
        assert espia.vistos == [None] * len(series)

    def test_history_es_de_solo_lectura(self, series) -> None:
        def mutar(view: MarketView) -> float:
            arr = view.history("close", 1)
            arr[0] = 999.0  # debe fallar
            return arr[0]

        espia = _correr(mutar, series)
        assert all(isinstance(e, ValueError) for e in espia.errores)

    def test_la_vista_no_referencia_la_serie_completa(self, series) -> None:
        """Ni por atributos privados se llega mas alla de t."""
        espia = _correr(lambda v: max(len(a) for a in v._fields.values()), series)
        assert espia.vistos == [1, 2, 3, 4, 5]

    def test_la_vista_no_expone_la_barseries(self, series) -> None:
        espia = _correr(lambda v: v._series, series)
        assert all(isinstance(e, AttributeError) for e in espia.errores)


class TestSuperficieDelMotor:
    # Superficie publica permitida: `run`, que conduce el bucle, mas los cinco
    # metodos de ExecutionVenue. Ninguno de los seis recibe ni devuelve una
    # barra, y ninguno mueve el cursor `_t`; los tests de abajo lo verifican en
    # vez de confiar en la lectura del codigo.
    #
    # La igualdad exacta es deliberada: si alguien agrega un metodo publico al
    # motor, este test falla y obliga a justificarlo aca antes de que la Etapa 2
    # pueda apoyarse en el.
    SUPERFICIE_PERMITIDA: ClassVar[set[str]] = {
        "run",
        "submit_order",
        "cancel_order",
        "get_positions",
        "get_fills",
        "reconcile",
    }

    def test_el_motor_no_expone_step(self, series) -> None:
        """Sin ``step()`` publico no hay forma de adelantar el cursor y mirar.

        La Etapa 2 (entorno Gymnasium) tendra que ceder el control desde dentro
        del bucle, no conducirlo desde fuera.
        """
        simulator = Simulator(series, SimConfig(initial_cash=1_000.0))
        publicos = {a for a in dir(simulator) if not a.startswith("_")}
        assert publicos == self.SUPERFICIE_PERMITIDA

    def test_ningun_metodo_publico_adelanta_el_cursor(self, series) -> None:
        """Lo que el test anterior protege, dicho como invariante.

        No importa cuantos metodos publicos tenga el venue mientras ninguno
        mueva ``_t``: el cursor lo mueve el bucle de ``run`` y nadie mas.
        """
        simulator = Simulator(series, SimConfig(initial_cash=1_000.0))
        antes = simulator._t
        simulator.get_positions()
        simulator.get_fills()
        simulator.reconcile()
        simulator.cancel_order("inexistente")
        simulator.submit_order(MarketOrder(qty=1.0, client_order_id="x-1"))
        assert simulator._t == antes

    def test_el_venue_detenido_no_ejecuta_nada(self, series) -> None:
        """Enviar fuera del bucle se rechaza; no hay ejecucion sin barra."""
        simulator = Simulator(series, SimConfig(initial_cash=1_000.0))
        ack = simulator.submit_order(MarketOrder(qty=1.0, client_order_id="x-1"))
        assert not ack.accepted
        assert simulator.get_fills() == []
        assert simulator.get_positions() == {}

    def test_los_metodos_del_venue_no_devuelven_barras(self, series) -> None:
        """Ninguna salida del protocolo referencia la serie ni una barra futura."""
        simulator = Simulator(series, SimConfig(initial_cash=1_000.0))
        estado = simulator.reconcile()
        assert not isinstance(estado.timestamp, np.ndarray)
        assert isinstance(estado.positions, dict)
        assert isinstance(simulator.get_fills(), list)

    def test_la_estrategia_no_recibe_la_serie(self, series) -> None:
        recibidos: list[type] = []

        class Inspector:
            name = "inspector"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                recibidos.append(type(view))
                return None

        Simulator(series, SimConfig(initial_cash=1_000.0)).run(Inspector())
        assert set(recibidos) == {MarketView}


class TestConsistenciaTemporalDeLaEjecucion:
    def test_la_decision_de_t_se_ejecuta_al_open_de_t_mas_uno(
        self, equity: InstrumentSpec
    ) -> None:
        """Nunca en la misma barra que genero la senal."""
        series = make_series(
            equity,
            closes=[100.0, 100.0, 100.0, 100.0],
            opens=[100.0, 105.0, 110.0, 115.0],
        )

        class CompraEnCero:
            name = "compra_en_cero"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return MarketOrder(qty=1.0) if view.t == 0 else None

        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(CompraEnCero())
        fill = result.fills[0]
        assert fill.t_decision == 0
        assert fill.t_fill == 1
        assert fill.decision_price == 100.0  # close de t
        assert fill.ref_price == 105.0  # open de t+1, no de t

    def test_una_orden_en_la_ultima_barra_expira(self, equity: InstrumentSpec) -> None:
        """No hay barra siguiente donde ejecutarla: no se pierde, se registra."""
        series = make_series(equity, [100.0, 101.0, 102.0])

        class CompraSiempre:
            name = "compra_siempre"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return MarketOrder(qty=1.0)

        result = Simulator(series, SimConfig(initial_cash=1_000.0)).run(CompraSiempre())
        assert result.fills[-1].status.value == "EXPIRED"
        assert result.fills[-1].t_decision == len(series) - 1

    def test_el_spread_no_usa_la_barra_de_ejecucion(
        self, equity: InstrumentSpec
    ) -> None:
        """Corwin-Schultz debe estimarse con barras <= t, no con la de t+1.

        Se construyen dos series identicas salvo por el rango de la barra de
        ejecucion. Si el estimador mirara esa barra, el spread cobrado cambiaria.
        """
        import pandas as pd

        from data.loaders import bars_from_frame
        from sim.costs import CorwinSchultzSpread

        def build(rango_barra_2: float):
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01T21:00:00Z", periods=3, freq="1D"
                    ),
                    "open": [100.0, 100.0, 100.0],
                    "high": [101.0, 101.0, 100.0 + rango_barra_2],
                    "low": [99.0, 99.0, 100.0 - rango_barra_2],
                    "close": [100.0, 100.0, 100.0],
                    "volume": [1e6, 1e6, 1e6],
                    "symbol": equity.symbol,
                }
            )
            return bars_from_frame(frame, instrument=equity, freq="1D")

        class CompraEnUno:
            name = "compra_en_uno"

            def reset(self, seed: int | None = None) -> None:
                return None

            def on_bar(self, view, account):
                return MarketOrder(qty=1.0) if view.t == 1 else None

        config = SimConfig(initial_cash=1_000.0, spread=CorwinSchultzSpread())
        estrecha = Simulator(build(0.5), config).run(CompraEnUno())
        ancha = Simulator(build(20.0), config).run(CompraEnUno())

        assert estrecha.fills[0].t_fill == 2
        assert estrecha.fills[0].spread_cost == pytest.approx(
            ancha.fills[0].spread_cost
        )


def test_una_estrategia_no_puede_corromper_la_serie(
    equity: InstrumentSpec,
) -> None:
    """Aunque mute lo que reciba, la serie original queda intacta."""
    series = make_series(equity, [100.0, 101.0, 102.0, 103.0])
    original = np.array(series.close, copy=True)

    class Vandalo:
        name = "vandalo"

        def reset(self, seed: int | None = None) -> None:
            return None

        def on_bar(self, view, account):
            with contextlib.suppress(ValueError):
                view.history("close", 1)[0] = 0.0
            return None

    Simulator(series, SimConfig(initial_cash=1_000.0)).run(Vandalo())
    np.testing.assert_array_equal(series.close, original)
