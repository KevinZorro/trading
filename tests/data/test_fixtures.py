"""Propiedades estadisticas de los fixtures: cada nivel es lo que dice ser.

Todo lo que se verifica aca es comprobable **sin un agente**: correlaciones,
autocorrelaciones, varianzas marginales, ausencia de senal. Si estos tests
pasan, un fallo de la Parte B es del agente o del pipeline, no del fixture.

Las tolerancias se **derivan del tamano de muestra**, no se tantean. El error
estandar de un coeficiente AR(1) estimado por OLS es ``sqrt((1-beta^2)/n)`` y la
banda de Bartlett de una autocorrelacion nula es ``1.96/sqrt(n)``; una tolerancia
elegida a ojo esconde tanto un generador roto como uno sano.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from data.fixtures import (
    DEFAULT_SIGMA,
    LEVEL_0_AMPLITUDE,
    LEVELS,
    SNR_SWEEP_BETAS,
    Fixture,
    FixtureCosts,
    FixtureError,
    SignalSpec,
    calibrate_costs,
    drift_t_population,
    generate_signal_bars,
    level_0_deterministic,
    level_1_noisy,
    level_2_costly,
    level_3_regime_flip,
    level_4_control,
    level_4b_detectable_drift,
    log_drift_per_bar,
)
from data.instruments import CommissionSchema
from data.schema import FloatArray

N_TEST = 4_000
SEED = 20240115


def ols_ar1(r: FloatArray) -> tuple[float, float]:
    """Pendiente y R^2 de ``r_{t+1} ~ a + b*r_t``. Cerrado, sin librerias.

    Se calcula a mano y no con ``np.polyfit`` porque el oraculo de un test no
    puede ser una caja negra: la formula de la pendiente de una regresion simple
    es ``cov(x,y)/var(x)`` y es la que hay que verificar.
    """
    x, y = r[:-1], r[1:]
    xm, ym = x.mean(), y.mean()
    cov = float(((x - xm) * (y - ym)).mean())
    var = float(((x - xm) ** 2).mean())
    b = cov / var
    residuos = y - (ym + b * (x - xm))
    r2 = 1.0 - float((residuos**2).mean()) / float(((y - ym) ** 2).mean())
    return b, r2


def se_ar1(beta: float, n: int) -> float:
    return math.sqrt((1.0 - beta**2) / n)


def log_returns(fixture: Fixture) -> FloatArray:
    close = np.asarray(fixture.series.close, dtype=np.float64)
    return np.asarray(np.log(close[1:] / close[:-1]))


def todos_los_niveles() -> list[Fixture]:
    return [
        level_0_deterministic(N_TEST),
        level_1_noisy(N_TEST, seed=SEED),
        level_2_costly(N_TEST, seed=SEED),
        level_3_regime_flip(N_TEST, seed=SEED),
        level_4_control(N_TEST, seed=SEED),
    ]


# ---------------------------------------------------------------------------
# Invariantes comunes a los cinco niveles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", todos_los_niveles(), ids=lambda f: f.name)
def test_no_hay_gap_entre_barras(fixture: Fixture) -> None:
    """``open[t] == close[t-1]`` exactamente.

    Es la condicion que hace exacto el techo: el precio con el que se dimensiona
    la orden y el de referencia de la ejecucion son el mismo numero, asi que
    mantener la posicion durante ``t+1`` captura exactamente ``r_{t+1}``.
    """
    apertura = np.asarray(fixture.series.open)[1:]
    cierre = np.asarray(fixture.series.close)[:-1]
    assert np.array_equal(apertura, cierre)


@pytest.mark.parametrize("fixture", todos_los_niveles(), ids=lambda f: f.name)
def test_ohlc_coherente(fixture: Fixture) -> None:
    s = fixture.series
    o, h, low, c = (np.asarray(getattr(s, f)) for f in ("open", "high", "low", "close"))
    assert np.all(h >= np.maximum(o, c))
    assert np.all(low <= np.minimum(o, c))
    assert np.all(low > 0.0)
    assert np.all(np.asarray(s.volume) > 0.0)
    # Sin camino intra-barra, high y low colapsarian a los extremos y
    # CorwinSchultz devolveria spread cero: el nivel 2 correria sin spread.
    assert np.any(h > np.maximum(o, c))
    assert np.any(low < np.minimum(o, c))


@pytest.mark.parametrize("fixture", todos_los_niveles(), ids=lambda f: f.name)
def test_la_senal_es_el_ultimo_log_retorno(fixture: Fixture) -> None:
    """``signal[t] == log(close[t]/close[t-1])``, que es lo que el agente ve.

    La senal tiene que estar en el precio: la observacion del entorno es cerrada
    y una columna exogena seria invisible. ``signal[0]`` queda fuera: la barra 0
    no tiene barra anterior.
    """
    np.testing.assert_allclose(
        np.asarray(fixture.signal)[1:], log_returns(fixture), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("fixture", todos_los_niveles(), ids=lambda f: f.name)
def test_el_source_marca_el_fixture(fixture: Fixture) -> None:
    """La etiqueta viaja con los datos hasta ``SimResult.config``."""
    assert fixture.series.source.startswith("fixture:")
    assert fixture.validation_only is True
    assert fixture.series.meta()["source"] == fixture.series.source


@pytest.mark.parametrize("fixture", todos_los_niveles(), ids=lambda f: f.name)
def test_describe_es_serializable(fixture: Fixture) -> None:
    """La config se guarda junto a cada resultado, asi que tiene que ser JSON."""
    texto = json.dumps(fixture.describe(), default=str)
    assert "validation_only" in texto


def test_fixture_rechaza_una_serie_sin_etiqueta() -> None:
    base = level_1_noisy(100, seed=1)
    from dataclasses import replace

    serie = replace(base.series, source="yahoo:AAPL")
    with pytest.raises(FixtureError, match="fixture:"):
        Fixture(
            level=1,
            name="mentira",
            series=serie,
            spec=base.spec,
            costs=base.costs,
            signal=base.signal,
            seed=1,
        )


def test_el_registro_cubre_los_cinco_niveles_y_el_4b() -> None:
    """El 4b no es un sexto nivel: es la segunda mitad del 4, separada.

    El 4a pregunta si el agente inventa senal donde no la hay; el 4b, si
    reconoce drift cuando existe. Estaban mezcladas en un solo veredicto.
    """
    assert sorted(LEVELS) == [f"level_{i}" for i in range(5)] + ["level_4b"]


# ---------------------------------------------------------------------------
# Nivel 0
# ---------------------------------------------------------------------------


def test_nivel_0_no_tiene_aleatoriedad() -> None:
    """Dos generaciones son bit-identicas. Ni siquiera el high es aleatorio."""
    a, b = level_0_deterministic(500), level_0_deterministic(500)
    for campo in ("open", "high", "low", "close", "volume"):
        assert np.array_equal(
            np.asarray(getattr(a.series, campo)), np.asarray(getattr(b.series, campo))
        )
    assert a.seed is None


def test_nivel_0_alterna_con_amplitud_exacta() -> None:
    """Oraculo a mano: los retornos son ``-a, +a, -a, ...`` sin excepcion."""
    a = 0.003
    fixture = level_0_deterministic(9, amplitude=a)
    esperado = np.array([-a, a, -a, a, -a, a, -a, a])
    np.testing.assert_allclose(log_returns(fixture), esperado, rtol=0, atol=1e-14)


def test_nivel_0_tiene_autocorrelacion_menos_uno() -> None:
    r = log_returns(level_0_deterministic(N_TEST))
    assert np.corrcoef(r[:-1], r[1:])[0, 1] == pytest.approx(-1.0, abs=1e-12)


def test_nivel_0_no_tiene_deriva() -> None:
    """El precio oscila entre dos valores: todo el retorno del optimo viene de
    la senal, ninguno del drift. Sin esto, "el agente supera a estar invertido"
    seria trivialmente cierto por construccion."""
    fixture = level_0_deterministic(N_TEST)
    close = np.asarray(fixture.series.close)
    assert close[-1] == pytest.approx(close[0] * math.exp(-LEVEL_0_AMPLITUDE), rel=1e-9)


def test_nivel_0_rechaza_amplitud_no_positiva() -> None:
    with pytest.raises(FixtureError, match="amplitude debe ser positiva"):
        level_0_deterministic(100, amplitude=0.0)


# ---------------------------------------------------------------------------
# Nivel 1: SNR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", SNR_SWEEP_BETAS)
def test_nivel_1_recupera_beta(beta: float) -> None:
    """El proceso tiene el ``beta`` que dice tener, dentro de 4 errores estandar."""
    fixture = level_1_noisy(N_TEST, seed=SEED, beta=beta)
    estimado, _ = ols_ar1(np.asarray(fixture.signal)[1:])
    assert abs(estimado - beta) < 4.0 * se_ar1(beta, N_TEST)


@pytest.mark.parametrize("beta", SNR_SWEEP_BETAS)
def test_el_r_cuadrado_es_beta_al_cuadrado(beta: float) -> None:
    """El SNR del fixture es exactamente ``beta^2``: el barrido no estima nada."""
    fixture = level_1_noisy(N_TEST, seed=SEED, beta=beta)
    _, r2 = ols_ar1(np.asarray(fixture.signal)[1:])
    assert r2 == pytest.approx(beta**2, abs=8.0 * se_ar1(beta, N_TEST) * abs(beta))


def test_el_barrido_de_snr_no_mueve_la_varianza_marginal() -> None:
    """Sin esto, el barrido confundiria SNR con regimen de volatilidad.

    Un agente que rinde peor con menos SNR podria estar rindiendo peor por
    operar en otro regimen de volatilidad, y las dos explicaciones serian
    inseparables. ``sigma = sigma_target*sqrt(1-beta^2)`` existe por esto.
    """
    desvios = [
        float(np.asarray(level_1_noisy(N_TEST, seed=SEED, beta=b).signal)[1:].std())
        for b in SNR_SWEEP_BETAS
    ]
    for d in desvios:
        assert d == pytest.approx(DEFAULT_SIGMA, rel=0.03)
    assert max(desvios) - min(desvios) < 0.02 * DEFAULT_SIGMA


def test_el_barrido_de_snr_no_mueve_la_media_marginal() -> None:
    medias = [
        float(np.asarray(level_1_noisy(N_TEST, seed=SEED, beta=b).signal)[1:].mean())
        for b in SNR_SWEEP_BETAS
    ]
    error = DEFAULT_SIGMA / math.sqrt(N_TEST)
    for m in medias:
        assert abs(m - 0.0003) < 4.0 * error


def test_el_volumen_no_lleva_senal() -> None:
    """La senal tiene un solo canal, y es el ultimo retorno.

    Si el volumen correlacionara con la magnitud del retorno -como en el
    generador de Heston, donde es deliberado-, "el agente aprendio del ultimo
    retorno" pasaria a ser una de dos posibilidades en vez de un hecho.
    """
    fixture = level_1_noisy(N_TEST, seed=SEED)
    r = np.abs(np.asarray(fixture.signal)[1:])
    v = np.asarray(fixture.series.volume)[1:]
    assert abs(float(np.corrcoef(r, v)[0, 1])) < 4.0 / math.sqrt(N_TEST)


def test_un_fixture_con_ruido_exige_semilla() -> None:
    spec = SignalSpec(drift=0.0, beta=-0.3, sigma_target=0.01)
    with pytest.raises(FixtureError, match="semilla explicita"):
        generate_signal_bars(100, spec=spec, seed=None)


# ---------------------------------------------------------------------------
# Nivel 2: calibracion de costos
# ---------------------------------------------------------------------------


def test_la_calibracion_pone_el_costo_en_el_cuantil_pedido() -> None:
    """El edge supera al costo en la fraccion de barras que se pidio.

    Banda binomial: con ``n`` barras y probabilidad ``q``, el desvio de la
    frecuencia es ``sqrt(q(1-q)/n)``.
    """
    q = 0.30
    fixture = level_2_costly(N_TEST, seed=SEED, target_quantile=q)
    edge = np.asarray(fixture.expected_next_log_return)[1:]
    frecuencia = float(np.mean(edge > fixture.costs.round_trip))
    error = math.sqrt(q * (1.0 - q) / N_TEST)
    assert abs(frecuencia - q) < 4.0 * error


def test_el_costo_de_ida_y_vuelta_tiene_la_forma_cerrada() -> None:
    """Oraculo a mano: ``drift + |beta|*sigma*Phi^{-1}(1-q)``."""
    spec = SignalSpec(drift=0.0003, beta=-0.3, sigma_target=0.012)
    costos = calibrate_costs(spec, target_quantile=0.30, commission_value=0.0005)
    # Phi^{-1}(0.70) = 0.5244005127080407
    esperado = 0.0003 + 0.3 * 0.012 * 0.5244005127080407
    assert costos.round_trip == pytest.approx(esperado, rel=1e-9)
    assert costos.spread_bps == pytest.approx((esperado - 0.001) * 1e4, rel=1e-9)


def test_la_calibracion_falla_si_la_comision_ya_supera_el_objetivo() -> None:
    spec = SignalSpec(drift=0.0, beta=-0.05, sigma_target=0.005)
    with pytest.raises(FixtureError, match="ya\n?\\s*supera el costo objetivo"):
        calibrate_costs(spec, target_quantile=0.30, commission_value=0.01)


def test_el_instrumento_del_nivel_2_lleva_la_misma_comision() -> None:
    """Olvidarse de pasar ``SimConfig.commission`` no puede cambiar el costo."""
    fixture = level_2_costly(200, seed=SEED)
    esquema = fixture.series.instrument.commission_schema
    assert esquema == fixture.costs.commission


def test_los_costos_solo_admiten_comision_porcentual() -> None:
    with pytest.raises(FixtureError, match="comision porcentual"):
        FixtureCosts(
            spread_bps=1.0, commission=CommissionSchema(kind="fixed", value=1.0)
        )
    with pytest.raises(FixtureError, match="comision minima"):
        FixtureCosts(
            spread_bps=1.0,
            commission=CommissionSchema(kind="percent", value=0.001, minimum=1.0),
        )


# ---------------------------------------------------------------------------
# Nivel 3: cambio de regimen
# ---------------------------------------------------------------------------


def test_nivel_3_invierte_el_signo_de_beta_a_mitad_de_serie() -> None:
    n = 8_000
    fixture = level_3_regime_flip(n, seed=SEED)
    r = np.asarray(fixture.signal)
    corte = n // 2
    b1, _ = ols_ar1(r[1:corte])
    b2, _ = ols_ar1(r[corte:])
    assert b1 < 0 < b2
    assert abs(b1 - fixture.spec.beta) < 4.0 * se_ar1(fixture.spec.beta, corte)
    esperado = fixture.spec.beta_after_flip
    assert esperado is not None
    assert abs(b2 - esperado) < 4.0 * se_ar1(esperado, corte)


def test_nivel_3_no_mueve_la_distribucion_marginal_al_invertir() -> None:
    """Solo cambia la predictibilidad. Un flip que ademas moviera el drift
    superpondria dos efectos inseparables."""
    n = 8_000
    r = np.asarray(level_3_regime_flip(n, seed=SEED).signal)[1:]
    corte = n // 2
    primera, segunda = r[:corte], r[corte:]
    error_media = DEFAULT_SIGMA / math.sqrt(corte)
    assert abs(float(primera.mean()) - float(segunda.mean())) < 4.0 * error_media
    assert float(primera.std()) == pytest.approx(float(segunda.std()), rel=0.05)


def test_nivel_3_exige_declarar_el_flip_completo() -> None:
    with pytest.raises(FixtureError, match="van juntos"):
        SignalSpec(drift=0.0, beta=-0.3, sigma_target=0.01, flip_at=100)


# ---------------------------------------------------------------------------
# Nivel 4: control negativo
# ---------------------------------------------------------------------------


def se_autocorrelacion_robusta(r: FloatArray, k: int) -> float:
    """Error estandar de una autocorrelacion muestral con heterocedasticidad.

    **La banda de Bartlett (``1.96/sqrt(n)``) no sirve aca y usarla seria un
    error, no una aproximacion.** Bartlett supone observaciones iid; los
    retornos de Heston son una diferencia de martingala con varianza
    condicional que se agrupa, y bajo esa estructura la varianza del estimador
    de la autocorrelacion es mayor que ``1/n``. Con la banda ingenua el nivel 4
    falla en lags aislados con series perfectamente sanas -medido: rho=-0.057 en
    el lag 3 con banda 0.031- y "el control negativo tiene senal" seria una
    conclusion falsa producida por el yardstick, no por los datos.

    Se usa el error estandar robusto habitual para diferencias de martingala::

        se(rho_k)^2 = sum_t x_t^2 x_{t+k}^2 / (sum_t x_t^2)^2,  x_t = r_t - media
    """
    x = r - r.mean()
    numerador = float(((x[:-k] ** 2) * (x[k:] ** 2)).sum())
    denominador = float((x**2).sum()) ** 2
    return math.sqrt(numerador / denominador)


def test_nivel_4_no_tiene_autocorrelacion() -> None:
    """La afirmacion del control negativo: la direccion es impredecible.

    ``E[r_{t+1} | F_t] = mu*dt`` por construccion del proceso. Un agente que
    gana aca esta sobreajustando ruido, y hay que entenderlo antes de seguir.
    """
    fixture = level_4_control(N_TEST, seed=SEED)
    r = np.asarray(fixture.signal)[1:]
    for k in range(1, 11):
        rho = float(np.corrcoef(r[:-k], r[k:])[0, 1])
        z = rho / se_autocorrelacion_robusta(r, k)
        assert abs(z) < 4.0, f"lag {k}: rho={rho:.4f} z={z:.2f}"


def test_nivel_4_deja_la_volatilidad_predecible() -> None:
    """Lo unico explotable de Heston es la volatilidad, y si es predecible.

    El test existe para que "no hay senal" no se lea como "no hay estructura":
    el fixture tiene reversion a la media en la varianza, y eso es deliberado.
    """
    fixture = level_4_control(N_TEST, seed=SEED)
    r2 = np.abs(np.asarray(fixture.signal)[1:])
    assert float(np.corrcoef(r2[:-1], r2[1:])[0, 1]) > 4.0 / math.sqrt(len(r2))


def test_nivel_4_rechaza_un_regimen_desconocido() -> None:
    with pytest.raises(FixtureError, match="regimen desconocido"):
        level_4_control(100, seed=1, regime="extremo")


# ---------------------------------------------------------------------------
# Particion
# ---------------------------------------------------------------------------


def test_split_es_contiguo_y_no_solapa() -> None:
    fixture = level_1_noisy(1_000, seed=SEED)
    train, val, test = fixture.split()
    assert len(train) + len(val) + len(test) == len(fixture)
    reconstruido = np.concatenate(
        [np.asarray(p.series.close) for p in (train, val, test)]
    )
    assert np.array_equal(reconstruido, np.asarray(fixture.series.close))
    assert train.series.timestamp[-1] < val.series.timestamp[0]
    assert val.series.timestamp[-1] < test.series.timestamp[0]
    assert train.name.endswith(":train")


def test_split_rechaza_fracciones_que_no_suman_uno() -> None:
    with pytest.raises(FixtureError, match="deben sumar 1"):
        level_1_noisy(1_000, seed=SEED).split(train=0.5, validation=0.2, test=0.2)


def test_split_rechaza_tramos_vacios() -> None:
    with pytest.raises(FixtureError, match="queda con"):
        level_1_noisy(20, seed=SEED).split(train=0.98, validation=0.01, test=0.01)


# ---------------------------------------------------------------------------
# Validaciones de la spec y del generador
# ---------------------------------------------------------------------------


def test_spec_rechaza_beta_no_estacionario() -> None:
    with pytest.raises(FixtureError, match="rango estacionario"):
        SignalSpec(drift=0.0, beta=1.0, sigma_target=0.01)


def test_spec_admite_beta_unitario_solo_sin_ruido() -> None:
    spec = SignalSpec(drift=0.0, beta=-1.0, sigma_target=0.0, r0=0.001)
    assert spec.is_deterministic


def test_el_generador_rechaza_series_degeneradas() -> None:
    spec = SignalSpec(drift=0.0, beta=-0.3, sigma_target=0.01)
    with pytest.raises(FixtureError, match="al menos 3 barras"):
        generate_signal_bars(2, spec=spec, seed=1)
    with pytest.raises(FixtureError, match="sub_steps debe ser"):
        generate_signal_bars(10, spec=spec, seed=1, sub_steps=1)


# ---------------------------------------------------------------------------
# Nivel 4b: drift detectable
# ---------------------------------------------------------------------------


def test_el_drift_logaritmico_no_es_el_aritmetico() -> None:
    """Oraculo a mano: ``(mu - theta/2)/bpy``, no ``mu/bpy``.

    La diferencia decide si el drift es detectable: con mu=0.08 y theta=0.09 el
    aritmetico es mas del doble del logaritmico, y un ``t`` calculado con el
    equivocado da 1.16 en vez de 0.51.
    """
    assert log_drift_per_bar(0.08, 0.09, 252.0) == pytest.approx((0.08 - 0.045) / 252)
    assert log_drift_per_bar(0.08, 0.09, 252.0) < 0.08 / 252


def test_el_t_poblacional_del_drift_tiene_la_forma_cerrada() -> None:
    """``t = (mu - theta/2) * sqrt(n) / sqrt(theta * bpy)``, calculado a mano."""
    mu, theta, n, bpy = 0.42, 0.09, 4_800, 252.0
    esperado = (mu - theta / 2) * math.sqrt(n) / math.sqrt(theta * bpy)
    assert drift_t_population(mu, theta, n, bpy) == pytest.approx(esperado, rel=1e-12)
    assert drift_t_population(mu, theta, n, bpy) == pytest.approx(5.46, abs=0.01)
    # El del nivel 4a, con el que se decidio que no era detectable.
    assert drift_t_population(0.08, theta, n, bpy) == pytest.approx(0.51, abs=0.01)


def test_el_t_crece_con_la_raiz_de_la_muestra() -> None:
    cuadruple = drift_t_population(0.42, 0.09, 4 * 4_800)
    simple = drift_t_population(0.42, 0.09, 4_800)
    assert cuadruple == pytest.approx(2.0 * simple, rel=1e-12)


def test_el_nivel_4b_tiene_drift_detectable_en_todos_sus_caminos() -> None:
    """La calibracion se verifica sobre la muestra, no sobre la formula.

    El umbral del nivel es ``t >= 3`` en la ventana de entrenamiento; si algun
    camino no llegara, el fixture no tendria lo que dice tener.
    """
    from eval.distribution import drift_t_statistic

    ts = []
    for semilla in (701, 709, 719, 727, 733, 739, 743, 751, 757, 761):
        train, _, _ = level_4b_detectable_drift(8_000, seed=semilla).split()
        ts.append(drift_t_statistic(np.asarray(train.signal)[1:]))
    assert min(ts) > 3.0
    assert float(np.median(ts)) > 4.0


def test_el_nivel_4b_difiere_del_4a_solo_en_el_drift() -> None:
    """Una sola variable de diferencia.

    Si el agente se comporta distinto entre 4a y 4b, la unica explicacion
    disponible tiene que ser el drift. Cambiar tambien la volatilidad haria que
    el contraste midiera dos cosas.
    """
    a = level_4_control(1_000, seed=SEED)
    b = level_4b_detectable_drift(1_000, seed=SEED)
    assert a.spec.sigma_target == pytest.approx(b.spec.sigma_target)
    assert a.spec.beta == b.spec.beta == 0.0
    assert b.spec.drift > a.spec.drift
    assert b.series.source.startswith("fixture:level_4b:")


def test_el_optimo_del_nivel_4b_sigue_siendo_estar_invertido() -> None:
    """Mas drift no crea senal direccional: el techo no rota."""
    _, validacion, _ = level_4b_detectable_drift(2_000, seed=SEED).split()
    techos = validacion.ceilings(safety=0.98, first_decision=26)
    np.testing.assert_array_equal(techos.informed, techos.always_long)
    assert techos.capture(techos.informed) is None


def test_el_t_poblacional_rechaza_parametros_imposibles() -> None:
    with pytest.raises(FixtureError, match="al menos 2 barras"):
        drift_t_population(0.4, 0.09, 1)
    with pytest.raises(FixtureError, match="deben ser positivos"):
        drift_t_population(0.4, 0.0, 100)
