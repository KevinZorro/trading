"""Ventanas rodantes. Los tests que importan son los de no-solapamiento.

El proyecto prohibe el split simple, y la razon por la que lo prohibe -una sola
observacion fuera de muestra- se pierde si las ventanas se solapan sin que nadie
lo diga: entonces hay muchas observaciones pero no son independientes, y
cualquier estadistico calculado sobre ellas sobreestima su propia precision.
"""

from __future__ import annotations

import pytest

from data.fixtures import level_1_noisy
from eval.walkforward import WalkForwardError, Window, coverage, rolling_windows

SEED = 20240115


def test_los_tramos_son_contiguos_y_en_orden() -> None:
    ventanas = rolling_windows(1_000, train=400, validation=100, test=100)
    for v in ventanas:
        assert v.train[1] == v.validation[0]
        assert v.validation[1] == v.test[0]
        assert v.train[0] < v.train[1] < v.test[0] < v.test[1]


def test_el_train_siempre_precede_al_test_de_su_ventana() -> None:
    """Entrenar con datos posteriores al periodo evaluado no rompe nada: da
    resultados buenisimos. Por eso hay que verificarlo y no suponerlo."""
    for anclado in (False, True):
        ventanas = rolling_windows(
            2_000, train=600, validation=200, test=200, anchored=anclado
        )
        for v in ventanas:
            assert v.train[1] <= v.test[0]
            assert v.validation[1] <= v.test[0]
            assert set(range(*v.train)).isdisjoint(range(*v.test))
            assert set(range(*v.validation)).isdisjoint(range(*v.test))


def test_con_el_paso_por_defecto_los_tests_particionan_la_serie() -> None:
    """Cada barra fuera de muestra se evalua exactamente una vez."""
    ventanas = rolling_windows(1_000, train=400, validation=100, test=100)
    cobertura = coverage(ventanas, 1_000)
    assert cobertura["overlapping_test_bars"] == 0
    vistos = [t for v in ventanas for t in range(*v.test)]
    assert len(vistos) == len(set(vistos))


def test_un_paso_corto_solapa_los_tests_y_queda_reportado() -> None:
    """Solapar no esta prohibido, esta **contado**.

    Con tests solapados hay mas observaciones que informacion independiente, y
    un p-value calculado sobre ellas sobreestima. El numero tiene que estar en
    el reporte antes de calcular cualquier estadistico.
    """
    ventanas = rolling_windows(1_000, train=400, validation=100, test=100, step=50)
    cobertura = coverage(ventanas, 1_000)
    assert cobertura["overlapping_test_bars"] > 0


def test_las_ventanas_ancladas_expanden_el_train() -> None:
    ventanas = rolling_windows(
        2_000, train=600, validation=200, test=200, anchored=True
    )
    assert all(v.train[0] == 0 for v in ventanas)
    longitudes = [v.train[1] - v.train[0] for v in ventanas]
    assert longitudes == sorted(longitudes)
    assert longitudes[-1] > longitudes[0]


def test_las_ventanas_rodantes_mantienen_el_train_del_mismo_largo() -> None:
    ventanas = rolling_windows(2_000, train=600, validation=200, test=200)
    assert {v.train[1] - v.train[0] for v in ventanas} == {600}


def test_ninguna_ventana_se_pasa_del_final_de_la_serie() -> None:
    ventanas = rolling_windows(1_050, train=400, validation=100, test=100)
    assert all(v.stop <= 1_050 for v in ventanas)
    assert coverage(ventanas, 1_050)["unused_tail"] >= 0


def test_una_serie_demasiado_corta_falla_con_su_motivo() -> None:
    with pytest.raises(WalkForwardError, match="una ventana necesita 600 barras"):
        rolling_windows(500, train=400, validation=100, test=100)


def test_el_minimo_por_tramo_cubre_el_calentamiento() -> None:
    """Un tramo mas corto que el calentamiento no da ni una observacion util."""
    with pytest.raises(WalkForwardError, match="no alcanza ni para el calentamiento"):
        rolling_windows(
            1_000, train=400, validation=10, test=100, min_bars_per_segment=26
        )


def test_los_parametros_no_positivos_se_rechazan() -> None:
    with pytest.raises(WalkForwardError, match="train debe ser positivo"):
        rolling_windows(1_000, train=0, validation=100, test=100)
    with pytest.raises(WalkForwardError, match="step debe ser positivo"):
        rolling_windows(1_000, train=400, validation=100, test=100, step=0)


def test_una_ventana_con_tramos_no_contiguos_se_rechaza() -> None:
    with pytest.raises(WalkForwardError, match="no son contiguos"):
        Window(index=0, train=(0, 100), validation=(120, 200), test=(200, 300))


def test_una_ventana_con_un_tramo_vacio_se_rechaza() -> None:
    with pytest.raises(WalkForwardError, match="el tramo test esta vacio"):
        Window(index=0, train=(0, 100), validation=(100, 200), test=(200, 200))


def test_el_tramo_se_pide_por_nombre_y_no_por_indice() -> None:
    fixture = level_1_noisy(1_000, seed=SEED)
    ventana = rolling_windows(1_000, train=400, validation=100, test=100)[0]
    train = ventana.slice(fixture.series, "train")
    assert len(train) == 400
    with pytest.raises(WalkForwardError, match="tramo desconocido"):
        ventana.slice(fixture.series, "todo")


def test_describe_es_serializable() -> None:
    import json

    ventanas = rolling_windows(1_000, train=400, validation=100, test=100)
    assert json.loads(json.dumps([v.describe() for v in ventanas]))[0]["index"] == 0
