"""Columnas exogenas en el entorno (N5, ADR 0012 seccion 3).

Oraculos a mano. El test central es el de siempre: dos series con el mismo pasado
y distinto futuro exogeno producen observaciones bit-identicas hasta ``t``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from data.errors import LookaheadError
from data.schema import BarSeries, ExogColumn, SchemaError
from envs.observation import ObservationBuilder, ObservationSpec, fit_scaler_on_train
from envs.trading_env import EnvConfig, TradingEnv
from features.scaler import FeatureScaler, ScalerError
from sim.view import MarketView

pytestmark = pytest.mark.leakage

BIFURCACION = 90
COLUMNAS = (ExogColumn("x", scaled=True), ExogColumn("ind", scaled=False))
SPEC = ObservationSpec(exog=COLUMNAS)


def con_exog(serie: BarSeries, matriz: np.ndarray) -> BarSeries:
    from dataclasses import replace

    return replace(serie, exog=matriz, exog_names=("x", "ind"))


def matriz_base(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    m = np.column_stack([rng.normal(5.0, 2.0, n), (rng.random(n) > 0.5) * 1.0])
    m[3, 0] = np.nan  # un enmascarado en el pasado
    return m


class TestSerieYVista:
    def test_el_slice_corta_las_exogenas_con_el_precio(self, serie) -> None:
        s = con_exog(serie, matriz_base(len(serie))).slice(10, 20)
        assert s.exog is not None
        assert s.exog.shape == (10, 2)
        assert_array_equal(s.exog[:, 1], matriz_base(len(serie))[10:20, 1])

    def test_forma_incorrecta_falla(self, serie) -> None:
        with pytest.raises(SchemaError, match=r"exog debe ser \(120, 2\)"):
            con_exog(serie, np.zeros((119, 2)))

    def test_infinito_falla_pero_nan_no(self, serie) -> None:
        m = matriz_base(len(serie))
        m[5, 0] = np.inf
        with pytest.raises(SchemaError, match="infinitos"):
            con_exog(serie, m)

    def test_la_vista_no_tiene_filas_posteriores_a_t(self, serie) -> None:
        m = matriz_base(len(serie))
        vista = MarketView(con_exog(serie, m), 50)
        assert_array_equal(vista.exog_row(), m[50])
        assert not vista.exog_row().flags.writeable

    def test_sin_exogenas_la_vista_lo_dice(self, serie) -> None:
        with pytest.raises(KeyError, match="no tiene columnas exogenas"):
            MarketView(serie, 5).exog_row()

    def test_meta_de_a_no_cambia(self, serie) -> None:
        assert "exog_names" not in serie.meta()
        assert con_exog(serie, matriz_base(len(serie))).meta()["exog_names"] == [
            "x",
            "ind",
        ]


class TestAgenteANoCambia:
    def test_layout_y_serializacion_de_a_son_los_de_siempre(self) -> None:
        a = ObservationBuilder(ObservationSpec())
        assert "exog" not in ObservationSpec().describe()
        b = ObservationBuilder(SPEC)
        assert b.names[: len(a)] == a.names
        assert b.names[len(a) :] == ("x", "ind")
        assert b.scale_mask[len(a) :] == (True, False)
        assert b.maskable_mask == (False,) * len(a) + (True, False)

    def test_observacion_de_a_identica_con_o_sin_exogenas_en_la_serie(
        self, serie, sim_config
    ) -> None:
        sc = fit_scaler_on_train(serie.slice(0, 80), ObservationBuilder())
        e_a = TradingEnv(serie, sim_config, scaler=sc)
        obs_a, _ = e_a.reset(seed=0)
        # B sin sus columnas de mas: mismas primeras columnas bit a bit.
        sb = con_exog(serie, matriz_base(len(serie)))
        sc_b = fit_scaler_on_train(sb.slice(0, 80), ObservationBuilder(SPEC))
        e_b = TradingEnv(sb, sim_config, scaler=sc_b, env_config=EnvConfig(SPEC))
        obs_b, _ = e_b.reset(seed=0)
        assert_array_equal(obs_b[: len(obs_a)], obs_a)

    def test_serie_con_exogenas_no_entra_a_un_entorno_de_a(self, serie, sim_config):
        sb = con_exog(serie, matriz_base(len(serie)))
        sc = fit_scaler_on_train(serie.slice(0, 80), ObservationBuilder())
        with pytest.raises(ValueError, match="la serie trae las exogenas"):
            TradingEnv(sb, sim_config, scaler=sc)


class TestAntiLeakage:
    def test_exogenas_con_distinto_futuro_dan_observaciones_identicas(
        self, serie, sim_config
    ) -> None:
        m1 = matriz_base(len(serie))
        m2 = m1.copy()
        m2[BIFURCACION:] = [1e6, 1.0]  # futuro exogeno disparatado
        s1, s2 = con_exog(serie, m1), con_exog(serie, m2)
        sc = fit_scaler_on_train(s1.slice(0, 80), ObservationBuilder(SPEC))
        e1 = TradingEnv(s1, sim_config, scaler=sc, env_config=EnvConfig(SPEC))
        e2 = TradingEnv(s2, sim_config, scaler=sc, env_config=EnvConfig(SPEC))
        o1, i1 = e1.reset(seed=0)
        o2, _ = e2.reset(seed=0)
        pasos = 0
        while i1["t"] < BIFURCACION - 1:
            assert_array_equal(o1, o2)
            o1, _, _, _, i1 = e1.step(0.5)
            o2, _, _, _, _ = e2.step(0.5)
            pasos += 1
        assert pasos > 10

    def test_el_escalador_de_train_no_ve_el_test(self, serie) -> None:
        m = matriz_base(len(serie))
        m[80:, 0] = 1e9  # el test exogeno no puede mover las estadisticas
        sc1 = fit_scaler_on_train(
            con_exog(serie, matriz_base(len(serie))).slice(0, 80),
            ObservationBuilder(SPEC),
        )
        sc2 = fit_scaler_on_train(
            con_exog(serie, m).slice(0, 80), ObservationBuilder(SPEC)
        )
        assert_array_equal(sc1.mean, sc2.mean)
        assert_array_equal(sc1.std, sc2.std)

    def test_lookback_negativo_sigue_siendo_lookahead(self, serie) -> None:
        with pytest.raises(LookaheadError, match="posterior"):
            MarketView(serie, 5).close(-1)


class TestEscaladorConMascara:
    def test_ajuste_sin_nan_y_nan_entra_como_cero(self) -> None:
        # Oraculo a mano: valores validos 1, 3 -> media 2, desvio 1.
        x = np.array([[1.0, 0.0], [np.nan, 1.0], [3.0, 1.0]])
        sc = FeatureScaler.fit(x, ("v", "i"), (True, False), (True, False))
        assert sc.mean[0] == 2.0
        assert sc.std[0] == 1.0
        assert sc.n_samples == 3
        salida = sc.transform(np.array([np.nan, 1.0]))
        assert_array_equal(salida, [0.0, 1.0])
        assert_array_equal(sc.transform(np.array([4.0, 1.0])), [2.0, 1.0])

    def test_columna_toda_enmascarada_queda_en_cero(self) -> None:
        x = np.array([[np.nan], [np.nan]])
        sc = FeatureScaler.fit(x, ("v",), (True,), (True,))
        assert_array_equal(sc.transform(np.array([np.nan])), [0.0])
        assert_array_equal(sc.transform(np.array([7.0])), [7.0])

    def test_nan_en_columna_no_enmascarable_sigue_siendo_error(self) -> None:
        x = np.array([[1.0, np.nan], [2.0, 1.0]])
        with pytest.raises(ScalerError, match="no finitos"):
            FeatureScaler.fit(x, ("a", "b"), (True, True), (True, False))

    def test_serializacion_ida_y_vuelta(self) -> None:
        x = np.array([[1.0], [np.nan], [3.0]])
        sc = FeatureScaler.fit(x, ("v",), (True,), (True,))
        assert FeatureScaler.from_dict(sc.to_dict()).maskable == (True,)
        sin = FeatureScaler.fit(x[[0, 2]], ("v",), (True,))
        assert "maskable" not in sin.to_dict()

    def test_maskable_exige_columna_escalada(self) -> None:
        with pytest.raises(ScalerError, match="escalada"):
            FeatureScaler(
                ("a",), np.zeros(1), np.ones(1), (False,), 1, maskable=(True,)
            )
