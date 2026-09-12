"""El optimo expuesto por cada fixture esta bien calculado.

Sin techo no hay evaluacion posible: "el agente rindio 12%" no significa nada si
no se sabe si el maximo alcanzable era 13% o 400%. Estos tests fijan tres cosas
distintas:

1. **La contabilidad del techo es la del motor.** ``evaluate_states`` se compara
   contra ``Simulator`` corriendo la misma secuencia de decisiones. Si divergen,
   el techo mide un mundo que el backtest no reproduce.
2. **La regla optima es la que se dice.** Oraculos calculados a mano, nunca la
   propia implementacion como referencia.
3. **Las cotas se ordenan.** ``always_flat <= informed <= clairvoyant`` siempre.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.fixtures import (
    Fixture,
    FixtureCosts,
    FixtureError,
    SignalSpec,
    _commission_cost,
    _umbral,
    clairvoyant_states,
    evaluate_states,
    fixture_instrument,
    level_0_deterministic,
    level_1_noisy,
    level_2_costly,
    level_3_regime_flip,
    level_4_control,
    myopic_states,
    periodic_rate_per_bar,
    solve_optimal_policy,
)
from data.instruments import CommissionSchema
from data.schema import FloatArray
from eval.metrics import max_drawdown, periodic_rate, sharpe
from sim.costs import FixedBpsSpread, NoSlippage, SchemaCommission
from sim.engine import SimConfig, Simulator
from sim.orders import MarketOrder, RejectReason
from sim.sizing import TargetWeightSizer
from sim.view import AccountSnapshot, MarketView

SEED = 20240115
CAPITAL = 100_000.0


class SigueEstados:
    """Estrategia que reproduce una secuencia de pesos dada.

    Usa el mismo ``TargetWeightSizer`` que el entorno de la Etapa 2, asi que lo
    que corre dentro del motor es exactamente lo que ``evaluate_states`` afirma
    estar contabilizando.
    """

    name = "sigue_estados"

    def __init__(self, estados: np.ndarray, safety: float) -> None:
        self.estados = estados
        self.sizer = TargetWeightSizer(safety=safety)

    def reset(self, seed: int | None = None) -> None:
        return None

    def on_bar(self, view: MarketView, account: AccountSnapshot) -> MarketOrder | None:
        return self.sizer.order_for(float(self.estados[view.t]), view, account)


def config_para(fixture: Fixture, *, cash_rate: float = 0.0) -> SimConfig:
    """La traduccion de ``FixtureCosts`` a ``SimConfig``, en un solo lugar."""
    return SimConfig(
        initial_cash=CAPITAL,
        spread=FixedBpsSpread(bps=fixture.costs.spread_bps),
        slippage=NoSlippage(),
        commission=SchemaCommission(schema=fixture.costs.commission),
        max_participation=1.0,
        cash_rate=cash_rate,
    )


# ---------------------------------------------------------------------------
# La contabilidad del techo es la del motor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("safety", [0.98, 0.90])
@pytest.mark.parametrize("cash_rate", [0.0, 0.04])
def test_evaluate_states_reproduce_al_simulador(
    safety: float, cash_rate: float
) -> None:
    """El techo y el motor tienen que dar la misma curva de equity.

    Es el test que ata toda la maquinaria de techos al backtest validado. Si
    esto falla, el techo mide un mundo distinto del que el agente habita y
    cualquier comparacion contra el es ruido.
    """
    fixture = level_2_costly(300, seed=SEED)
    estados = fixture.ceilings(safety=safety, cash_rate=cash_rate).informed_states
    resultado = Simulator(
        fixture.series, config_para(fixture, cash_rate=cash_rate)
    ).run(SigueEstados(estados, safety))
    esperado = evaluate_states(
        np.asarray(fixture.series.close, dtype=np.float64),
        estados,
        initial_cash=CAPITAL,
        safety=safety,
        rate_per_bar=periodic_rate_per_bar(cash_rate, 252.0),
        costs=fixture.costs,
    )
    np.testing.assert_allclose(resultado.equity, esperado, rtol=1e-9, atol=1e-6)

    # Los unicos rechazos admisibles son los del rebalanceo sub-lote: mientras
    # la posicion se mantiene, el sizer manda el delta chiquisimo que la deja en
    # el peso exacto y el venue lo rechaza por quedar en cero al redondear. Es
    # el comportamiento buscado -no se pre-redondea a cero, el rechazo queda en
    # el log- y no mueve el equity. Cualquier otro motivo si seria una
    # divergencia real entre el techo y el motor.
    motivos = {f.reject_reason for f in resultado.rejected_fills}
    assert motivos <= {RejectReason.ZERO_AFTER_ROUNDING}


def test_el_techo_con_safety_uno_es_teorico_y_el_venue_lo_rechaza() -> None:
    """Estar exactamente all-in exigiria conocer el precio de ejecucion.

    La orden dimensionada con ``safety=1.0`` pide todo el cash y no deja con que
    pagar spread y comision, asi que el motor la rechaza entera por
    ``INSUFFICIENT_CASH`` y la curva de equity se queda plana. El techo con
    ``safety=1.0`` existe como referencia sin friccion de ejecucion, no como
    barra alcanzable: leerlo como objetivo del agente seria pedirle lookahead.
    """
    fixture = level_2_costly(60, seed=SEED)
    estados = fixture.ceilings(safety=1.0).informed_states
    resultado = Simulator(fixture.series, config_para(fixture)).run(
        SigueEstados(estados, 1.0)
    )
    assert resultado.executed_fills == []
    assert {f.reject_reason for f in resultado.rejected_fills} == {
        RejectReason.INSUFFICIENT_CASH
    }
    assert np.all(resultado.equity == CAPITAL)


def test_la_comision_del_fixture_es_la_del_simulador() -> None:
    """La formula esta duplicada porque ``data`` no puede importar ``sim``.

    La duplicacion no se justifica con un comentario: se guarda con un test que
    compara las dos implementaciones sobre una grilla.
    """
    esquemas = [
        CommissionSchema(kind="percent", value=0.001),
        CommissionSchema(kind="fixed", value=1.5, minimum=0.5),
        CommissionSchema(kind="per_share", value=0.005, minimum=1.0),
        CommissionSchema(kind="percent", value=0.01, max_pct_notional=0.002),
    ]
    for esquema in esquemas:
        modelo = SchemaCommission(schema=esquema)
        for qty in (-1000.0, -0.5, 0.5, 3.0, 1e5):
            for price in (0.01, 1.0, 137.42, 50_000.0):
                assert _commission_cost(esquema, qty, price) == pytest.approx(
                    modelo.compute(qty, price), rel=1e-12
                )


def test_la_tasa_por_barra_coincide_en_los_tres_modulos() -> None:
    """``data``, ``sim`` y ``eval`` tienen que descontar con la misma formula."""
    for anual in (0.0, 0.01, 0.05, 0.25):
        propia = periodic_rate_per_bar(anual, 252.0)
        assert propia == pytest.approx(periodic_rate(anual, 252.0), rel=1e-15)
        assert propia == pytest.approx(
            SimConfig(initial_cash=1.0, cash_rate=anual).rate_per_bar, rel=1e-15
        )


# ---------------------------------------------------------------------------
# Oraculos a mano
# ---------------------------------------------------------------------------


def test_la_regla_miope_es_el_signo_del_edge() -> None:
    """Oraculo a mano: con ``safety=1`` y sin cash rate, ``w=1`` sii ``m_t > 0``."""
    m = np.array([-0.01, -1e-9, 0.0, 1e-9, 0.01])
    estados = myopic_states(m, np.full(5, 0.012))
    assert list(estados) == [0, 0, 0, 1, 1]


def test_la_regla_miope_descuenta_la_tasa_del_cash() -> None:
    """Con cash rate el umbral sube: mantener tiene que ganarle a estar en cash.

    Descontar cero le regalaria alfa a una estrategia que pasa la mitad del
    tiempo afuera cobrando esa misma tasa.
    """
    tasa = periodic_rate_per_bar(0.05, 252.0)
    umbral = math.log1p(tasa)
    m = np.array([umbral - 1e-9, umbral + 1e-9])
    assert list(myopic_states(m, np.zeros(2), rate_per_bar=tasa)) == [0, 1]


def test_clairvoyant_con_costos_oraculo_a_mano() -> None:
    """Camino de 5 barras, costo de 20 bps por lado, optimo calculado a mano.

    Retornos por barra: +1%, -0.990%, +2%, -0.980%. Con ``one_way=0.002`` la
    mejor secuencia es entrar y salir en cada barra ganadora -cuatro cambios de
    estado- porque cada giro cuesta 0.2% y las barras ganadoras dan 1% y 2%.
    """
    close = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
    costos = FixtureCosts(spread_bps=40.0)  # medio spread = 20 bps por lado
    assert costos.one_way == pytest.approx(0.002)

    estados = clairvoyant_states(close, costs=costos)
    assert list(estados) == [1, 0, 1, 0, 0]

    # Oraculo cerrado de una vuelta completa comprando con todo el equity: se
    # compra a P0*(1+c) y se vende a P1*(1-c), asi que
    #     equity_final = equity_inicial * (P1/P0 * (1-c) - c).
    # Se deriva a mano y no se copia de la implementacion.
    def vuelta(p0: float, p1: float, c: float) -> float:
        return p1 / p0 * (1.0 - c) - c

    esperado = vuelta(100.0, 101.0, 0.002) * vuelta(100.0, 102.0, 0.002)
    curva = evaluate_states(close, estados, costs=costos)
    assert float(curva[-1]) == pytest.approx(esperado, rel=1e-14)


def test_clairvoyant_sin_costos_mantiene_las_barras_positivas() -> None:
    close = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
    assert list(clairvoyant_states(close)) == [1, 0, 1, 0, 0]


def test_un_costo_alto_deja_al_clarividente_afuera() -> None:
    """Con el costo bastante por encima del rango, ni la vision perfecta opera."""
    close = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
    estados = clairvoyant_states(close, costs=FixtureCosts(spread_bps=400.0))
    assert list(estados) == [0, 0, 0, 0, 0]


def test_nivel_0_alcanza_el_optimo_analitico_exacto() -> None:
    """Techo cerrado: ``exp(a * floor((n-1)/2))``.

    Los retornos son ``a*(-1)^t``, positivos en los ``t`` pares, y el optimo
    esta invertido exactamente en esas barras. No hay nada que estimar.
    """
    n, a = 2_000, 0.002
    techos = level_0_deterministic(n, amplitude=a).ceilings()
    barras_ganadoras = (n - 1) // 2
    assert float(techos.informed[-1]) == pytest.approx(
        math.exp(a * barras_ganadoras), rel=1e-12
    )
    # Sin ruido, conocer el proceso equivale a conocer el futuro.
    np.testing.assert_allclose(techos.informed, techos.clairvoyant, rtol=1e-12)


def test_el_crecimiento_esperado_tiene_forma_cerrada() -> None:
    """``E[max(m, 0)]`` de una normal, contra la cuadratura del fixture.

    Con ``safety=1`` y sin cash rate el crecimiento logaritmico de mantener es
    exactamente ``m``, asi que el crecimiento del optimo es ``E[max(m,0)]`` con
    ``m ~ N(drift, (|beta|*sigma_target)^2)``, que se integra en cerrado.
    """
    fixture = level_1_noisy(500, seed=SEED)
    spec = fixture.spec
    mu_m = spec.drift
    s_m = abs(spec.beta) * spec.sigma_target
    z = mu_m / s_m
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    cerrado = mu_m * Phi + s_m * phi

    # La tolerancia es el error de la regla del trapecio en el quiebre del
    # `max`, O(h^2) con h = 20 desvios / 20000 puntos. No es un margen elegido a
    # ojo: con cuadratura gaussiana el error era del 0.5%, cuatro ordenes de
    # magnitud peor, y el test lo detecto.
    obtenido = fixture.ceilings().expected_growth_per_bar
    assert obtenido is not None
    assert obtenido == pytest.approx(cerrado, rel=1e-6)


# ---------------------------------------------------------------------------
# Orden de las cotas
# ---------------------------------------------------------------------------


def niveles() -> list[Fixture]:
    return [
        level_0_deterministic(600),
        level_1_noisy(600, seed=SEED),
        level_2_costly(600, seed=SEED),
        level_3_regime_flip(600, seed=SEED),
        level_4_control(600, seed=SEED),
    ]


@pytest.mark.parametrize("fixture", niveles(), ids=lambda f: f.name)
def test_el_informado_nunca_supera_al_clarividente(fixture: Fixture) -> None:
    """El optimo ex-ante no puede ganarle al que vio el ruido.

    Si se invierte, la regla informada esta usando informacion del futuro.
    """
    techos = fixture.ceilings()
    assert float(techos.informed[-1]) <= float(techos.clairvoyant[-1]) * (1.0 + 1e-12)


@pytest.mark.parametrize("fixture", niveles(), ids=lambda f: f.name)
def test_las_curvas_del_techo_son_series_de_equity(fixture: Fixture) -> None:
    """Se exponen como curvas para que ``eval.metrics`` se aplique tal cual.

    Comparar solo retornos totales contra un techo esconde que el techo puede
    alcanzarse por caminos con riesgo muy distinto.
    """
    techos = fixture.ceilings(initial_cash=CAPITAL)
    for curva in (techos.informed, techos.clairvoyant, techos.always_long):
        assert len(curva) == len(fixture)
        assert np.all(curva > 0.0)
        assert math.isfinite(sharpe(curva, 252.0))
        assert max_drawdown(curva).depth >= 0.0
    assert np.all(techos.always_flat == CAPITAL)


@pytest.mark.parametrize("fixture", niveles(), ids=lambda f: f.name)
def test_el_techo_depende_del_margen_de_seguridad(fixture: Fixture) -> None:
    """El techo cambia con ``safety``, y por eso hay que pedirlo con el propio.

    **La direccion no esta garantizada y la asercion no la afirma.** Sobre un
    camino realizado, exponerse menos puede terminar mejor: cuando los retornos
    realizados salen peores que su media condicional, el 2% que queda en cash es
    el 2% que no perdio. Medido en el nivel 1: ``safety=0.98`` termina 1% por
    **encima** de ``safety=1.0``. Escribir aqui "menor" seria una asercion
    invertida que pasaria en casi todos los fixtures y fallaria sin motivo real.
    """
    completo = float(fixture.ceilings(safety=1.0).informed[-1])
    recortado = float(fixture.ceilings(safety=0.98).informed[-1])
    assert recortado != pytest.approx(completo, rel=1e-6)


def test_sin_ruido_menos_exposicion_rinde_menos() -> None:
    """Donde si esta garantizada: con retornos ciertos y positivos.

    Es el unico caso en que la direccion se puede afirmar, porque no hay ruido
    que pueda hacer que exponerse menos salga mejor.
    """
    fixture = level_0_deterministic(400)
    assert float(fixture.ceilings(safety=0.98).informed[-1]) < float(
        fixture.ceilings(safety=1.0).informed[-1]
    )


# ---------------------------------------------------------------------------
# Nivel 2: la banda de no operar
# ---------------------------------------------------------------------------


def test_sin_costos_no_hay_banda_de_no_operar() -> None:
    """Sin friccion las decisiones se desacoplan: el optimo vuelve a ser miope.

    Es la validacion del programa dinamico contra algo cerrado: en el limite de
    costo cero tiene que reproducir el umbral ``m_t > 0``, o sea ``r_t < -mu/beta``.
    """
    spec = SignalSpec(drift=0.0003, beta=-0.3, sigma_target=0.012)
    politica = solve_optimal_policy(spec, FixtureCosts())
    assert politica.no_trade_band == pytest.approx(0.0, abs=1e-12)
    umbral_cerrado = -spec.mu_for(spec.beta) / spec.beta
    paso = float(politica.grid[1] - politica.grid[0])
    assert politica.enter_below == pytest.approx(umbral_cerrado, abs=paso)


def test_con_costos_aparece_una_banda_de_no_operar() -> None:
    """Se entra con un edge mas exigente del que hace falta para quedarse.

    Esa brecha **es** "operar selectivamente". Si el agente del nivel 2 opera
    igual que en el nivel 1, la penalizacion no esta llegando al reward.
    """
    fixture = level_2_costly(600, seed=SEED)
    politica = fixture.ceilings().policy
    assert politica is not None
    assert politica.decreasing is True
    assert politica.enter_below < politica.exit_above
    assert politica.no_trade_band > fixture.costs.round_trip


def test_el_optimo_con_costos_opera_menos_y_gana_mas() -> None:
    """La regla que ignora los costos opera de mas y rinde menos.

    Es el contraste que la Parte B tiene que medir en el agente: no basta con
    que rinda, tiene que **rotar menos** cuando hay costos.
    """
    fixture = level_2_costly(2_000, seed=SEED)
    techos = fixture.ceilings()
    ingenua = myopic_states(fixture.expected_next_log_return, fixture.sigma_by_bar)
    curva = evaluate_states(
        np.asarray(fixture.series.close, dtype=np.float64),
        ingenua,
        costs=fixture.costs,
    )
    cambios = int(np.count_nonzero(np.diff(ingenua.astype(np.int64))))
    assert techos.turnover_count < cambios
    assert float(techos.informed[-1]) > float(curva[-1])


def test_el_programa_dinamico_rechaza_lo_que_no_puede_resolver() -> None:
    with pytest.raises(FixtureError, match="cambio de\n?\\s*regimen"):
        solve_optimal_policy(
            SignalSpec(
                drift=0.0,
                beta=-0.3,
                sigma_target=0.01,
                beta_after_flip=0.3,
                flip_at=10,
            ),
            FixtureCosts(spread_bps=10.0),
        )
    with pytest.raises(FixtureError, match="determinista"):
        solve_optimal_policy(
            SignalSpec(drift=0.0, beta=-1.0, sigma_target=0.0, r0=0.01),
            FixtureCosts(spread_bps=10.0),
        )


def test_una_politica_no_monotona_es_un_error_y_no_un_promedio() -> None:
    """Un cruce doble delata una grilla demasiado gruesa, no un umbral raro."""
    grid = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(FixtureError, match="no es monotona"):
        _umbral(grid, np.array([1, 0, 1, 0, 0], dtype=np.int8), True)
    assert _umbral(grid, np.ones(5, dtype=np.int8), True) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Niveles 3 y 4
# ---------------------------------------------------------------------------


def test_el_memorizador_se_derrumba_despues_del_flip() -> None:
    """Aplicar la regla de la primera mitad a la segunda destruye capital.

    Convierte "se adapta o memoriza" en un numero medido en vez de una
    impresion: ambos resultados son validos, pero solo si quedan medidos.
    """
    n = 4_000
    fixture = level_3_regime_flip(n, seed=SEED)
    techos = fixture.ceilings()
    assert techos.memorizer is not None
    corte = n // 2
    segunda_mitad_memorizador = float(techos.memorizer[-1] / techos.memorizer[corte])
    segunda_mitad_informado = float(techos.informed[-1] / techos.informed[corte])
    assert segunda_mitad_memorizador < 1.0
    assert segunda_mitad_informado > 1.0


def test_los_niveles_sin_flip_no_tienen_memorizador() -> None:
    assert level_1_noisy(300, seed=SEED).ceilings().memorizer is None


def test_nivel_4_el_optimo_es_estar_siempre_invertido() -> None:
    """El control negativo: el techo **coincide** con estar siempre invertido.

    ``E[r_{t+1}|F_t] = mu*dt > 0`` constante, asi que la regla informada -que no
    sabe que el proceso es Heston, solo aplica el umbral- decide mantener en
    todas las barras y no rota nunca. Ese es el criterio de la Parte B.
    """
    techos = level_4_control(1_000, seed=SEED).ceilings()
    assert techos.turnover_count == 0
    assert techos.time_invested == 1.0
    np.testing.assert_array_equal(techos.informed, techos.always_long)


def test_nivel_4_la_captura_es_indefinida_y_no_cero() -> None:
    """Sin brecha entre el techo y estar invertido, la fraccion no existe.

    Un cero se leeria como "no capturo nada", que es distinto de "no habia nada
    que capturar", y esa diferencia **es** el resultado del control negativo.
    Mismo criterio que ``win_rate`` sin trades cerrados.
    """
    techos = level_4_control(500, seed=SEED).ceilings()
    assert techos.capture(techos.informed) is None


def test_la_captura_normaliza_entre_estar_invertido_y_el_techo() -> None:
    techos = level_1_noisy(1_000, seed=SEED).ceilings()
    assert techos.capture(techos.informed) == pytest.approx(1.0)
    assert techos.capture(techos.always_long) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Capacidad
# ---------------------------------------------------------------------------


def capacidad(
    fixture: Fixture,
    initial_cash: float,
    *,
    safety: float = 0.98,
    max_participation: float = 0.10,
) -> dict[str, float | bool]:
    """Participacion y nocional minimo que el techo informado exigiria al venue.

    Vive en los tests y no en ``Fixture`` a proposito: es una verificacion de
    que el techo esta bien calculado, no algo que el estudio consuma. Un techo
    que la capacidad de la barra no permite alcanzar es un techo mal calculado,
    porque el agente se quedaria corto por llenados parciales y el diagnostico
    culparia al aprendizaje.
    """
    techos = fixture.ceilings(safety=safety, initial_cash=initial_cash)
    close = np.asarray(fixture.series.close, dtype=np.float64)
    volumen = np.asarray(fixture.series.volume, dtype=np.float64)
    qty_objetivo = techos.informed_states.astype(np.float64) * safety * techos.informed
    qty_objetivo = qty_objetivo / close
    delta = np.abs(np.diff(np.concatenate([[0.0], qty_objetivo])))[1:]
    participacion = delta / volumen[1:]
    operadas = delta[delta > 0]
    nocional_min = (
        float((operadas * close[1:][delta > 0]).min()) if len(operadas) else 0.0
    )
    pico = float(participacion.max()) if len(participacion) else 0.0
    return {
        "peak_participation": pico,
        "capacity_limit": max_participation,
        "min_trade_notional": nocional_min,
        "min_notional_required": fixture.series.instrument.min_notional,
        "binds": bool(
            pico > max_participation
            or nocional_min < fixture.series.instrument.min_notional
        ),
    }


def test_el_techo_es_alcanzable_dentro_del_venue() -> None:
    """Un techo que la capacidad de la barra no deja alcanzar esta mal calculado.

    Sin este test, un agente que se queda corto por llenados parciales se veria
    como un agente que no aprendio.
    """
    for fixture in (
        level_0_deterministic(600),
        level_1_noisy(600, seed=SEED),
        level_2_costly(600, seed=SEED),
    ):
        reporte = capacidad(fixture, CAPITAL)
        assert reporte["binds"] is False, fixture.name
        assert float(reporte["peak_participation"]) < 0.01


def test_la_capacidad_detecta_que_el_capital_no_entra() -> None:
    """Con capital suficientemente grande, la barra deja de poder absorberlo."""
    fixture = level_1_noisy(600, seed=SEED)
    reporte = capacidad(fixture, 1e12)
    assert reporte["binds"] is True


# ---------------------------------------------------------------------------
# Guardas de la contabilidad
# ---------------------------------------------------------------------------


def test_evaluate_states_valida_sus_entradas() -> None:
    close = np.array([100.0, 101.0, 102.0])
    with pytest.raises(FixtureError, match="states tiene"):
        evaluate_states(close, np.zeros(2, dtype=np.int8))
    with pytest.raises(FixtureError, match="safety debe estar"):
        evaluate_states(close, np.zeros(3, dtype=np.int8), safety=1.5)
    with pytest.raises(FixtureError, match="initial_cash debe ser positivo"):
        evaluate_states(close, np.zeros(3, dtype=np.int8), initial_cash=0.0)


def test_first_decision_deja_el_calentamiento_afuera() -> None:
    """Antes del calentamiento el agente no decide: el techo tampoco."""
    fixture = level_1_noisy(300, seed=SEED)
    techos = fixture.ceilings(first_decision=26)
    assert not np.any(techos.informed_states[:26])
    assert np.all(techos.informed[:27] == techos.informed[0])


def test_el_instrumento_del_fixture_no_impone_minimos() -> None:
    """Deliberado: las restricciones de venue se miden en el barrido de capital.

    Un rechazo por nocional minimo aca haria ambiguo el diagnostico de la Parte
    B: el agente no llegaria al techo por un motivo ajeno a aprender la senal.
    """
    spec = fixture_instrument()
    assert spec.min_notional == 0.0
    assert spec.allow_fractional is True
    qty, motivo = spec.check_tradable(1e-6, 100.0)
    assert motivo is None
    assert qty == pytest.approx(1e-6)


def _sin_uso(x: FloatArray) -> None:  # pragma: no cover
    return None


# ---------------------------------------------------------------------------
# El cambio de regimen y la particion
#
# Los tres tests de abajo fijan un bug que **no** encontro ningun test: lo
# encontro correr el protocolo de la Parte B. `split()` no trasladaba el indice
# del cambio de regimen a las coordenadas del tramo, asi que el tramo de
# validacion -entero posterior al flip- conservaba `flip_at` del padre, un
# indice que su propia serie ni alcanza. `betas()` devolvia el beta **anterior**
# al cambio para datos que ya eran del regimen nuevo, y la regla optima quedaba
# exactamente invertida: el techo perdia el 89% del capital mientras estar
# siempre invertido perdia el 1.3%.
# ---------------------------------------------------------------------------


def test_split_traslada_el_cambio_de_regimen_al_tramo() -> None:
    """Cada tramo tiene que saber que regimen le toca, en su propio indice."""
    fixture = level_3_regime_flip(1_000, seed=SEED, flip_at=500)
    train, validacion, test = fixture.split(train=0.4, validation=0.3, test=0.3)

    # El train (0..400) es entero anterior al cambio: no hay flip que aplicar.
    assert train.spec.flip_at is None
    assert train.spec.beta_after_flip is None
    assert train.spec.beta == fixture.spec.beta

    # Validacion (400..700) contiene el corte, que cae en su indice 100.
    assert validacion.spec.flip_at == 100
    assert validacion.spec.beta == fixture.spec.beta

    # Test (700..1000) es entero posterior: el beta pasa a ser el de despues.
    assert test.spec.flip_at is None
    assert test.spec.beta == fixture.spec.beta_after_flip


def test_el_beta_del_tramo_es_el_que_los_datos_tienen() -> None:
    """La comprobacion empirica: el beta declarado coincide con el medido."""
    fixture = level_3_regime_flip(8_000, seed=SEED)
    _, validacion, _ = fixture.split()
    r = np.asarray(validacion.signal)[1:]
    x, y = r[:-1], r[1:]
    medido = float(
        ((x - x.mean()) * (y - y.mean())).mean() / ((x - x.mean()) ** 2).mean()
    )
    assert validacion.spec.beta > 0
    assert abs(medido - validacion.spec.beta) < 4.0 * math.sqrt(
        (1 - validacion.spec.beta**2) / len(r)
    )


def test_el_techo_de_un_tramo_posterior_al_flip_le_gana_a_estar_invertido() -> None:
    """La consecuencia observable del bug, fijada como test.

    Con el beta equivocado el "optimo" era peor que no hacer nada. Estar siempre
    invertido es una politica factible: el optimo informado tiene que ganarle.
    """
    fixture = level_3_regime_flip(8_000, seed=SEED)
    _, validacion, _ = fixture.split()
    techos = validacion.ceilings(safety=0.98, first_decision=26, initial_cash=CAPITAL)
    assert techos.informed_below_reference is False
    assert float(techos.informed[-1]) > float(techos.always_long[-1])


def test_un_techo_por_debajo_de_la_referencia_queda_marcado() -> None:
    """Aplicar a proposito la regla del regimen equivocado enciende la bandera.

    Es la contraprueba: sin esto, el test anterior podria estar en verde porque
    la bandera nunca se enciende con nada.
    """
    from dataclasses import replace as reemplazar

    fixture = level_3_regime_flip(8_000, seed=SEED)
    _, validacion, _ = fixture.split()
    invertido = reemplazar(validacion, spec=reemplazar(validacion.spec, beta=-0.3))
    techos = invertido.ceilings(safety=0.98, first_decision=26, initial_cash=CAPITAL)
    assert techos.informed_below_reference is True
    assert techos.capture(techos.informed) is None


def test_capture_no_devuelve_un_cociente_con_denominador_negativo() -> None:
    """Premiar el alejarse del techo seria reportar un bug como desempeno."""
    from dataclasses import replace as reemplazar

    fixture = level_3_regime_flip(4_000, seed=SEED)
    _, validacion, _ = fixture.split()
    roto = reemplazar(validacion, spec=reemplazar(validacion.spec, beta=-0.3))
    techos = roto.ceilings(safety=0.98, first_decision=26, initial_cash=CAPITAL)
    assert techos.capture(techos.always_long) is None
    assert techos.describe()["informed_below_reference"] is True


def test_el_edge_usa_el_beta_de_la_transicion_y_no_el_de_la_barra() -> None:
    """Oraculo a mano en la barra del cambio de regimen.

    El generador produce ``r[t]`` con el beta indexado en ``t``, asi que el que
    gobierna la transicion de ``t`` a ``t+1`` es el de ``t+1``. Dentro de un
    regimen coinciden; en la barra anterior al cambio, usar el de ``t`` aplica la
    regla del regimen viejo a una transicion que ya pertenece al nuevo.
    """
    spec = SignalSpec(
        drift=0.0,
        beta=-0.5,
        sigma_target=0.01,
        beta_after_flip=0.5,
        flip_at=3,
    )
    senal = np.array([0.02] * 5)
    esperado = np.array(
        [
            -0.5 * 0.02,  # t=0 -> t=1, ambos en el regimen viejo
            -0.5 * 0.02,  # t=1 -> t=2
            +0.5 * 0.02,  # t=2 -> t=3: la transicion YA es del regimen nuevo
            +0.5 * 0.02,  # t=3 -> t=4
            +0.5 * 0.02,  # t=4: no hay t+1, la orden expira
        ]
    )
    np.testing.assert_allclose(spec.conditional_mean(senal), esperado, atol=1e-15)
