"""El escalador se ajusta solo con train. Nunca ve el test.

Normalizar con estadisticas de todo el dataset filtra informacion del test al
train: la media y el desvio de la serie completa dependen de barras que en el
momento de entrenar todavia no ocurrieron. Los tests de aca fijan que eso no
pasa, con oraculos calculados aparte de la implementacion.
"""

from __future__ import annotations

import numpy as np
import pytest

from features.scaler import DESVIO_MINIMO, FeatureScaler, ScalerError

NOMBRES = ("a", "b", "c")
MASCARA = (True, True, False)

# Matriz elegida para que la media y el desvio salgan a mano.
# columna a: [1, 3, 5]      -> media 3, desvio poblacional 2*sqrt(2)/sqrt(3)... se
#                              calcula abajo desde la definicion
# columna b: [10, 10, 10]   -> constante: desvio cero
# columna c: [0, 1, 0]      -> fuera de la mascara: no se escala
MATRIZ = np.array([[1.0, 10.0, 0.0], [3.0, 10.0, 1.0], [5.0, 10.0, 0.0]])


def _media_y_desvio(columna: np.ndarray) -> tuple[float, float]:
    """Oraculo independiente: la definicion, no la implementacion."""
    n = len(columna)
    media = sum(columna) / n
    varianza = sum((x - media) ** 2 for x in columna) / n
    return media, varianza**0.5


class TestAjuste:
    def test_media_y_desvio_calculados_a_mano(self) -> None:
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        media_a, desvio_a = _media_y_desvio(MATRIZ[:, 0])
        assert media_a == pytest.approx(3.0)
        assert desvio_a == pytest.approx(np.sqrt(8.0 / 3.0))
        assert escalador.mean[0] == pytest.approx(media_a)
        assert escalador.std[0] == pytest.approx(desvio_a)

    def test_una_columna_constante_no_se_divide_por_cero(self) -> None:
        """Desvio cero en train: la columna pasa sin escalar en vez de estallar."""
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        assert escalador.std[1] == 1.0
        transformada = escalador.transform(np.array([1.0, 10.0, 0.0]))
        assert np.isfinite(transformada).all()

    def test_las_columnas_fuera_de_la_mascara_pasan_intactas(self) -> None:
        """Un indicador binario normalizado sale de su rango natural sin ganar nada."""
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        assert escalador.mean[2] == 0.0
        assert escalador.std[2] == 1.0
        entrada = np.array([3.0, 10.0, 1.0])
        assert escalador.transform(entrada)[2] == 1.0

    def test_la_transformacion_centra_y_escala(self) -> None:
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        media_a, desvio_a = _media_y_desvio(MATRIZ[:, 0])
        salida = escalador.transform(np.array([5.0, 10.0, 0.0]))
        assert salida[0] == pytest.approx((5.0 - media_a) / desvio_a)

    def test_guarda_cuantas_muestras_uso(self) -> None:
        assert FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA).n_samples == 3


class TestNoSeReajusta:
    def test_transform_no_cambia_las_estadisticas(self) -> None:
        """Un escalador que se reajusta en test es un modelo distinto del validado."""
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        media_antes = escalador.mean.copy()
        desvio_antes = escalador.std.copy()
        for _ in range(5):
            escalador.transform(np.array([1e6, 1e6, 1e6]))
        np.testing.assert_array_equal(escalador.mean, media_antes)
        np.testing.assert_array_equal(escalador.std, desvio_antes)

    def test_el_escalador_es_inmutable(self) -> None:
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        with pytest.raises((AttributeError, TypeError)):
            escalador.n_samples = 99  # type: ignore[misc]


class TestTrainNoVeTest:
    def test_ajustar_en_train_ignora_lo_que_venga_despues(self) -> None:
        """El oraculo del anti-patron: las estadisticas de train no se mueven."""
        train = MATRIZ
        test_suave = np.array([[2.0, 10.0, 0.0]])
        test_extremo = np.array([[1e9, 10.0, 0.0]])

        solo_train = FeatureScaler.fit(train, NOMBRES, MASCARA)
        con_test_suave = FeatureScaler.fit(
            np.vstack([train, test_suave]), NOMBRES, MASCARA
        )
        con_test_extremo = FeatureScaler.fit(
            np.vstack([train, test_extremo]), NOMBRES, MASCARA
        )

        # Ajustar sobre train + test cambia las estadisticas: eso es el leakage.
        assert not np.allclose(solo_train.mean, con_test_suave.mean)
        assert not np.allclose(solo_train.mean, con_test_extremo.mean)
        # Y el escalador de train no se entera de ninguno de los dos.
        assert solo_train.mean[0] == pytest.approx(3.0)


class TestSerializacion:
    def test_va_y_vuelve_identico(self) -> None:
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        copia = FeatureScaler.from_dict(escalador.to_dict())
        np.testing.assert_array_equal(copia.mean, escalador.mean)
        np.testing.assert_array_equal(copia.std, escalador.std)
        assert copia.names == escalador.names
        assert copia.scale_mask == escalador.scale_mask
        assert copia.n_samples == escalador.n_samples

    def test_la_copia_transforma_igual(self) -> None:
        """El mismo transformador tiene que aplicarse en test y, luego, en vivo."""
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        copia = FeatureScaler.from_dict(escalador.to_dict())
        entrada = np.array([4.0, 10.0, 1.0])
        np.testing.assert_array_equal(
            escalador.transform(entrada), copia.transform(entrada)
        )

    def test_to_dict_es_json_serializable(self) -> None:
        import json

        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        assert json.loads(json.dumps(escalador.to_dict()))["n_samples"] == 3


class TestIdentidad:
    def test_no_transforma_nada(self) -> None:
        escalador = FeatureScaler.identity(NOMBRES)
        entrada = np.array([7.0, -3.0, 0.5])
        np.testing.assert_array_equal(escalador.transform(entrada), entrada)


class TestEntradasInvalidas:
    def test_matriz_unidimensional(self) -> None:
        with pytest.raises(ScalerError, match="2-D"):
            FeatureScaler.fit(np.array([1.0, 2.0]), NOMBRES, MASCARA)

    def test_sin_muestras(self) -> None:
        with pytest.raises(ScalerError, match="no hay muestras"):
            FeatureScaler.fit(np.empty((0, 3)), NOMBRES, MASCARA)

    def test_columnas_que_no_coinciden_con_los_nombres(self) -> None:
        with pytest.raises(ScalerError, match="columnas"):
            FeatureScaler.fit(MATRIZ, ("a", "b"), (True, True))

    def test_mascara_de_otro_largo(self) -> None:
        with pytest.raises(ScalerError, match="un valor por columna"):
            FeatureScaler.fit(MATRIZ, NOMBRES, (True, True))

    def test_valores_no_finitos_en_el_ajuste(self) -> None:
        malo = MATRIZ.copy()
        malo[0, 0] = np.nan
        with pytest.raises(ScalerError, match="no finitos"):
            FeatureScaler.fit(malo, NOMBRES, MASCARA)

    def test_transformar_un_vector_de_otro_largo(self) -> None:
        escalador = FeatureScaler.fit(MATRIZ, NOMBRES, MASCARA)
        with pytest.raises(ScalerError, match="se esperaban 3 features"):
            escalador.transform(np.array([1.0, 2.0]))

    def test_desvio_no_positivo_en_la_construccion(self) -> None:
        with pytest.raises(ScalerError, match="desvios deben ser positivos"):
            FeatureScaler(
                names=NOMBRES,
                mean=np.zeros(3),
                std=np.array([1.0, 0.0, 1.0]),
                scale_mask=MASCARA,
                n_samples=1,
            )

    def test_el_umbral_de_desvio_minimo_es_relativo_a_cero(self) -> None:
        """Un desvio por debajo del umbral se trata como columna constante."""
        casi_constante = np.array([[1.0], [1.0 + DESVIO_MINIMO / 10]])
        escalador = FeatureScaler.fit(casi_constante, ("x",), (True,))
        assert escalador.std[0] == 1.0
