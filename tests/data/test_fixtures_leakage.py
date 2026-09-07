"""Anti-leakage de los fixtures: el pasado no puede depender del futuro.

Un fixture con fuga es peor que ningun fixture. Si la barra 10 cambiara segun
cuantas barras se pidieron, o si el techo de un tramo usara datos posteriores,
la Parte B validaria el pipeline contra una regla imposible y el resultado
-bueno o malo- no significaria nada.

Dos garantias distintas se verifican aca:

1. **Estructural, en el generador.** Cada componente aleatorio tiene su propio
   stream, asi que los numeros que consume no dependen de ``n_bars``. Generar
   4000 barras produce, en las primeras 2000, exactamente las mismas barras que
   generar 2000.
2. **En el techo.** La regla informada mira ``r_t`` y nada mas, asi que el techo
   de un prefijo es el prefijo del techo. El clarividente **si** cambia, y el
   test lo exige: sin eso, la invariancia del informado podria estar pasando
   por vacuidad.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.fixtures import (
    level_0_deterministic,
    level_1_noisy,
    level_2_costly,
    level_3_regime_flip,
    level_4_control,
)
from sim.view import MarketView

pytestmark = pytest.mark.leakage

SEED = 20240115
CORTO, LARGO = 1_000, 2_000


def pares() -> list[tuple[str, object, object]]:
    return [
        ("level_0", level_0_deterministic(CORTO), level_0_deterministic(LARGO)),
        (
            "level_1",
            level_1_noisy(CORTO, seed=SEED),
            level_1_noisy(LARGO, seed=SEED),
        ),
        (
            "level_2",
            level_2_costly(CORTO, seed=SEED),
            level_2_costly(LARGO, seed=SEED),
        ),
    ]


@pytest.mark.parametrize(
    ("nombre", "corto", "largo"), pares(), ids=lambda x: str(x)[:16]
)
def test_el_pasado_del_fixture_no_depende_de_cuanto_futuro_se_pidio(
    nombre: str, corto: object, largo: object
) -> None:
    """Serie de 1000 barras == prefijo de la de 2000, bit a bit.

    Con un solo generador aleatorio compartido esto seria falso: cuantos numeros
    consume el camino intra-barra depende de ``n_bars``, y el volumen de la
    barra 10 cambiaria segun cuantas barras se hayan pedido. Cada componente
    tiene su propio stream justamente por esto.
    """
    for campo in ("open", "high", "low", "close", "volume", "timestamp"):
        a = np.asarray(getattr(corto.series, campo))  # type: ignore[attr-defined]
        b = np.asarray(getattr(largo.series, campo))  # type: ignore[attr-defined]
        assert np.array_equal(a, b[:CORTO]), f"{nombre}.{campo}"
    assert np.array_equal(
        np.asarray(corto.signal),  # type: ignore[attr-defined]
        np.asarray(largo.signal)[:CORTO],  # type: ignore[attr-defined]
    )


def test_el_generador_de_heston_no_conserva_el_prefijo() -> None:
    """Limitacion conocida del nivel 4, fijada por test en vez de supuesta.

    ``generate_gbm_sv`` (Etapa 1) usa **un solo** generador para los shocks de
    precio, los de varianza, el gap y el volumen. Cuantos numeros consume el
    primer bloque depende de ``n_bars``, asi que pedir 2000 barras no produce en
    las primeras 1000 las mismas barras que pedir 1000.

    **No es una fuga de informacion**: dentro de una serie el camino se
    construye hacia adelante barra por barra, y ninguna barra depende de una
    posterior. Es una limitacion de reproducibilidad entre longitudes distintas,
    y se deja fijada aqui para que nadie construya un experimento suponiendo lo
    contrario. Cambiar el generador de la Etapa 1 moveria todas las semillas ya
    usadas, que cuesta mas de lo que esta propiedad vale.
    """
    corto = level_4_control(CORTO, seed=SEED)
    largo = level_4_control(LARGO, seed=SEED)
    assert not np.array_equal(
        np.asarray(corto.series.close), np.asarray(largo.series.close)[:CORTO]
    )


def test_el_nivel_3_conserva_el_prefijo_con_el_flip_en_su_sitio() -> None:
    """El nivel 3 va aparte: el flip esta en ``n//2`` y se mueve con ``n_bars``.

    Fijando ``flip_at`` la propiedad vuelve a valer, y que haga falta fijarlo es
    la prueba de que el corte es un parametro del proceso y no un accidente.
    """
    corto = level_3_regime_flip(CORTO, seed=SEED, flip_at=400)
    largo = level_3_regime_flip(LARGO, seed=SEED, flip_at=400)
    assert np.array_equal(
        np.asarray(corto.series.close), np.asarray(largo.series.close)[:CORTO]
    )


@pytest.mark.parametrize(
    ("nombre", "corto", "largo"), pares(), ids=lambda x: str(x)[:16]
)
def test_la_senal_se_recupera_de_los_precios_hasta_t(
    nombre: str, corto: object, largo: object
) -> None:
    """``signal[t]`` sale de ``close[:t+1]``: es lo que el agente ve en ``t``.

    Se reconstruye desde ``MarketView``, que es el unico canal por el que una
    estrategia toca los datos y que por construccion no referencia nada
    posterior a ``t``. Si la senal no se pudiera reconstruir asi, el fixture
    estaria pidiendole al agente que adivine algo que no puede observar.
    """
    fixture = corto  # type: ignore[assignment]
    serie = fixture.series  # type: ignore[attr-defined]
    senal = np.asarray(fixture.signal)  # type: ignore[attr-defined]
    for t in (1, 7, 100, CORTO - 1):
        visto = float(MarketView(serie, t).log_returns(1)[-1])
        assert visto == pytest.approx(float(senal[t]), abs=1e-12), f"{nombre} t={t}"


@pytest.mark.parametrize(
    ("nombre", "corto", "largo"), pares(), ids=lambda x: str(x)[:16]
)
def test_el_edge_esperado_no_usa_barras_posteriores(
    nombre: str, corto: object, largo: object
) -> None:
    """``E[r_{t+1}|F_t]`` de un prefijo coincide con el del total, salvo el flip."""
    a = np.asarray(corto.expected_next_log_return)  # type: ignore[attr-defined]
    b = np.asarray(largo.expected_next_log_return)  # type: ignore[attr-defined]
    np.testing.assert_allclose(a, b[:CORTO], rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "fixture",
    [
        level_0_deterministic(LARGO),
        level_1_noisy(LARGO, seed=SEED),
        level_2_costly(LARGO, seed=SEED),
        level_3_regime_flip(LARGO, seed=SEED),
        level_4_control(LARGO, seed=SEED),
    ],
    ids=lambda f: f.name,
)
def test_el_techo_informado_de_un_prefijo_es_el_prefijo_del_techo(
    fixture: object,
) -> None:
    """La regla informada decide con ``r_t`` y nada mas.

    Se compara el techo calculado sobre la serie recortada contra el prefijo del
    techo calculado sobre la serie entera. Vale tambien para el nivel 2, donde
    la politica sale de un programa dinamico: la solucion es estacionaria
    -depende del estado, no del instante-, asi que tampoco mira hacia adelante.

    **La ultima barra queda fuera de la comparacion, y el motivo es el propio
    anti-leakage.** La regla informada conoce el proceso, incluido cuando cambia
    de regimen, asi que en su ultima barra el techo de la serie entera usa el
    beta que gobierna la transicion hacia la barra **siguiente** -que existe para
    el padre y no para el hijo-. El hijo no puede saber que regimen viene
    despues del final de sus datos, y que difieran ahi es exactamente lo que
    tiene que pasar: si coincidieran, el techo del tramo estaria usando
    informacion de fuera de su ventana. La decision de esa barra ademas expira
    sin ejecutarse.
    """
    train, _, _ = fixture.split(train=0.5, validation=0.25, test=0.25)  # type: ignore[attr-defined]
    n = len(train)
    entero = fixture.ceilings()  # type: ignore[attr-defined]
    prefijo = train.ceilings()
    np.testing.assert_array_equal(
        prefijo.informed_states[: n - 1], entero.informed_states[: n - 1]
    )
    np.testing.assert_allclose(
        prefijo.informed[: n - 1], entero.informed[: n - 1], rtol=1e-12
    )


def test_el_techo_de_un_tramo_no_sabe_que_regimen_viene_despues() -> None:
    """La contraparte del test anterior: en la frontera **tienen** que diferir.

    El padre sabe que en la barra siguiente cambia el regimen; el hijo no puede
    saberlo, porque esa barra esta fuera de su serie. Sin este test, recortar la
    comparacion en ``n-1`` pareceria una tolerancia de conveniencia en vez de la
    consecuencia de la garantia.
    """
    fixture = level_3_regime_flip(LARGO, seed=SEED, flip_at=LARGO // 2)
    train, _, _ = fixture.split(train=0.5, validation=0.25, test=0.25)
    n = len(train)
    assert n == fixture.spec.flip_at  # el corte cae justo en el borde del tramo
    entero = fixture.ceilings()
    prefijo = train.ceilings()
    assert prefijo.informed_states[n - 1] != entero.informed_states[n - 1]


def test_el_clarividente_con_costos_si_mira_hacia_adelante() -> None:
    """Contraprueba: la invariancia del informado no pasa por vacuidad.

    El clarividente es ex-post por definicion. **Sin costos no se nota**, y eso
    no es un descuido del test: sin friccion las decisiones se desacoplan y su
    eleccion en ``t`` depende solo de ``R_{t+1}``, asi que el prefijo tambien se
    conserva. Con costos aparece el acoplamiento -conviene aguantar una barra
    mala si la siguiente compensa el giro- y ahi si, agregar barras al final
    cambia decisiones anteriores.

    Que la diferencia sea de pocas barras es lo esperado: solo se mueven las
    decisiones cercanas al horizonte y las que estaban al borde de la banda.
    """
    corto = level_2_costly(CORTO, seed=SEED)
    largo = level_2_costly(LARGO, seed=SEED)
    a = corto.ceilings().clairvoyant_states
    b = largo.ceilings().clairvoyant_states
    assert not np.array_equal(a, b[:CORTO])

    # Sin costos, en cambio, el clarividente es miope y el prefijo se conserva.
    sin_costos_corto = level_1_noisy(CORTO, seed=SEED).ceilings().clairvoyant_states
    sin_costos_largo = level_1_noisy(LARGO, seed=SEED).ceilings().clairvoyant_states
    np.testing.assert_array_equal(sin_costos_corto, sin_costos_largo[:CORTO])


def test_el_techo_de_un_tramo_no_ve_los_otros_tramos() -> None:
    """El techo de train sale igual calculado sobre train o sobre la serie entera.

    Es la version de la particion: si el techo de train dependiera de test, la
    comparacion del agente contra su techo en validacion ya seria optimista
    antes de que el agente existiera.
    """
    fixture = level_1_noisy(LARGO, seed=SEED)
    train, _, _ = fixture.split()
    entero = fixture.ceilings()
    solo_train = train.ceilings()
    n = len(train)
    np.testing.assert_array_equal(
        solo_train.informed_states, entero.informed_states[:n]
    )
    np.testing.assert_allclose(solo_train.informed, entero.informed[:n], rtol=1e-12)
