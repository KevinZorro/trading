"""Contrato del entorno: determinismo, recorte registrado, rechazos visibles.

Los casos borde de este archivo son los que el estudio va a pisar de verdad:
capital insuficiente en el barrido de la Etapa 6, la ultima barra que expira, y
el episodio de longitud 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.instruments import CommissionSchema, InstrumentSpec, us_equity_spec
from envs.observation import ObservationBuilder, fit_scaler_on_train
from envs.rewards import DifferentialSharpeReward, NetReturnReward
from envs.trading_env import EnvConfig, TradingEnv
from features.scaler import FeatureScaler
from risk.layer import RiskLayer, RiskLimits
from sim.clock import SimulatedClock
from sim.engine import LONG_ONLY_ACTION_RANGE, SimConfig, Simulator
from sim.orders import OrderStatus, RejectReason
from sim.sizing import TargetWeightSizer

from ..sim.conftest import make_series
from .conftest import serie_deterministica

T0 = np.datetime64("2020-01-01T21:00:00", "ns")


def correr(env: TradingEnv, accion: float = 0.5, seed: int = 0):
    obs, info = env.reset(seed=seed)
    trayectoria = [(obs.copy(), 0.0, dict(info))]
    while True:
        obs, r, terminado, truncado, info = env.step(np.array([accion]))
        trayectoria.append((obs.copy(), r, dict(info)))
        if terminado or truncado:
            return trayectoria


class TestEspacios:
    def test_la_accion_usa_la_constante_del_motor(
        self, serie, sim_config, scaler
    ) -> None:
        """No se redeclara el rango: se importa de sim.engine."""
        env = TradingEnv(serie, sim_config, scaler=scaler)
        bajo, alto = LONG_ONLY_ACTION_RANGE
        assert float(env.action_space.low[0]) == bajo
        assert float(env.action_space.high[0]) == alto

    def test_la_observacion_declara_su_tamano(
        self, serie, sim_config, scaler, builder
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        assert env.observation_space.shape == (len(builder),)
        assert env.feature_names == builder.names

    def test_ninguna_feature_es_un_precio_crudo(self, builder) -> None:
        """Los precios no son estacionarios y no entran en la observacion."""
        prohibidos = {"close", "open", "high", "low", "price", "precio"}
        assert not (set(builder.names) & prohibidos)

    def test_el_escalador_incompatible_se_rechaza(self, serie, sim_config) -> None:
        malo = FeatureScaler.identity(("a", "b"))
        with pytest.raises(ValueError, match="no corresponde a esta observacion"):
            TradingEnv(serie, sim_config, scaler=malo)


class TestDeterminismo:
    def test_misma_semilla_misma_trayectoria(self, serie, sim_config, scaler) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        a = correr(env, accion=0.7, seed=42)
        b = correr(env, accion=0.7, seed=42)
        assert len(a) == len(b)
        for (obs_a, r_a, _), (obs_b, r_b, _) in zip(a, b, strict=True):
            np.testing.assert_array_equal(obs_a, obs_b)
            assert r_a == r_b

    def test_dos_instancias_dan_lo_mismo(self, serie, sim_config, scaler) -> None:
        a = correr(TradingEnv(serie, sim_config, scaler=scaler), accion=0.3, seed=1)
        b = correr(TradingEnv(serie, sim_config, scaler=scaler), accion=0.3, seed=1)
        for (obs_a, r_a, _), (obs_b, r_b, _) in zip(a, b, strict=True):
            np.testing.assert_array_equal(obs_a, obs_b)
            assert r_a == r_b

    def test_reset_no_arrastra_estado_entre_episodios(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        correr(env, accion=1.0, seed=0)
        primera = env.result
        correr(env, accion=1.0, seed=0)
        segunda = env.result
        assert primera is not None
        assert segunda is not None
        np.testing.assert_array_equal(primera.equity, segunda.equity)


class TestRecorteRegistrado:
    """Un agente entrenado sobre un rango que el entorno recorta en silencio
    aprende sobre un mundo que no existe."""

    def test_la_accion_por_encima_del_rango_se_recorta_y_se_registra(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([3.5]))
        assert info["action_clipped"] is True
        assert len(env.clip_events) == 1
        evento = env.clip_events[0]
        assert evento.raw_action == 3.5
        assert evento.clipped_action == LONG_ONLY_ACTION_RANGE[1]

    def test_la_accion_por_debajo_del_rango_tambien(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        env.step(np.array([-2.0]))
        assert env.clip_events[0].raw_action == -2.0
        assert env.clip_events[0].clipped_action == LONG_ONLY_ACTION_RANGE[0]

    def test_una_accion_valida_no_registra_nada(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.5]))
        assert info["action_clipped"] is False
        assert env.clip_events == []

    def test_el_registro_de_recortes_se_limpia_en_reset(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        env.step(np.array([9.0]))
        assert len(env.clip_events) == 1
        env.reset(seed=0)
        assert env.clip_events == []

    def test_una_accion_no_finita_es_un_error_no_un_recorte(
        self, serie, sim_config, scaler
    ) -> None:
        """NaN no se recorta en silencio a 0 o a 1: es un bug del agente."""
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="no finita"):
            env.step(np.array([np.nan]))


class TestRechazosVisibles:
    """El agente debe poder distinguir "no quise operar" de "no pude"."""

    def _indice(self, builder: ObservationBuilder, nombre: str) -> int:
        return builder.names.index(nombre)

    def test_un_rechazo_del_venue_llega_a_la_observacion_siguiente(
        self, builder
    ) -> None:
        """Capital muy bajo contra una comision minima: el venue rechaza."""
        spec = us_equity_spec(
            "TEST",
            commission=CommissionSchema(kind="per_share", value=0.005, minimum=5.0),
        )
        serie = serie_deterministica(spec, n=80, seed=9)
        scaler = fit_scaler_on_train(serie.slice(0, 60), builder)
        env = TradingEnv(serie, SimConfig(initial_cash=60.0), scaler=scaler)

        env.reset(seed=0)
        vio_rechazo = False
        columna = self._indice(builder, "last_order_rejected")
        while True:
            obs, _, terminado, truncado, info = env.step(np.array([1.0]))
            if info["fill_status"] == OrderStatus.REJECTED.value:
                assert obs[columna] == 1.0
                assert info["fill_reject_reason"] is not None
                vio_rechazo = True
                break
            if terminado or truncado:
                break
        assert vio_rechazo, "el caso de capital insuficiente no se ejercito"

    def test_el_rechazo_queda_en_el_log_con_su_motivo(self, builder) -> None:
        spec = us_equity_spec(
            "TEST",
            commission=CommissionSchema(kind="per_share", value=0.005, minimum=5.0),
        )
        serie = serie_deterministica(spec, n=80, seed=9)
        scaler = fit_scaler_on_train(serie.slice(0, 60), builder)
        env = TradingEnv(serie, SimConfig(initial_cash=60.0), scaler=scaler)
        correr(env, accion=1.0)
        resultado = env.result
        assert resultado is not None
        motivos = {f.reject_reason for f in resultado.rejected_fills}
        assert motivos
        assert None not in motivos

    def test_un_veto_de_riesgo_llega_a_la_observacion_siguiente(
        self, serie, sim_config, scaler, builder
    ) -> None:
        def gate() -> RiskLayer:
            return RiskLayer(
                RiskLimits(max_position_pct_equity=0.10), SimulatedClock(T0)
            )

        env = TradingEnv(serie, sim_config, scaler=scaler, gate_factory=gate)
        env.reset(seed=0)
        columna = self._indice(builder, "last_order_blocked")
        vio_veto = False
        while True:
            obs, _, terminado, truncado, info = env.step(np.array([1.0]))
            if info["gate_reason"] is not None:
                assert obs[columna] == 1.0
                vio_veto = True
                break
            if terminado or truncado:
                break
        assert vio_veto

    def test_el_veto_y_el_rechazo_son_columnas_distintas(self, builder) -> None:
        """ "El venue no pudo" y "nuestra capa no dejo" son diagnosticos distintos."""
        assert "last_order_blocked" in builder.names
        assert "last_order_rejected" in builder.names

    def test_la_posicion_de_la_observacion_es_la_realmente_llenada(
        self, builder
    ) -> None:
        """Tras un fill parcial la intencion y la realidad divergen.

        Se fuerza el llenado parcial con un volumen de barra minusculo: el
        simulador recorta por participacion y cancela el remanente.
        """
        spec = InstrumentSpec(
            symbol="FRAC",
            venue="TEST",
            tick_size=0.01,
            lot_size=1e-8,
            allow_fractional=True,
            qty_precision=8,
            min_order_qty=0.0,
            min_notional=0.0,
            commission_schema=CommissionSchema(kind="fixed", value=0.0),
            asset_class="crypto",
        )
        closes = np.full(80, 100.0)
        serie = make_series(spec, closes, volume=5.0)
        scaler = fit_scaler_on_train(serie.slice(0, 60), builder)
        env = TradingEnv(serie, SimConfig(initial_cash=100_000.0), scaler=scaler)

        env.reset(seed=0)
        columna = self._indice(builder, "position_weight")
        obs, _, _, _, info = env.step(np.array([1.0]))
        parcial = info["fill_partial"]
        sin_llenar = info["fill_qty"] == 0.0
        assert parcial or sin_llenar
        # El peso observado es el de la posicion realmente llenada, no el 0.98
        # que el sizer pidio.
        assert obs[columna] < 0.5
        resultado_parcial = info["fill_qty"]
        assert obs[columna] * info["equity"] == pytest.approx(
            resultado_parcial * 100.0, rel=1e-6
        )


class TestRecompensa:
    def test_el_retorno_neto_sale_del_equity_del_simulador(
        self, serie, sim_config, scaler
    ) -> None:
        """Los costos ya estan dentro: el equity los pago al asentar el fill."""
        env = TradingEnv(serie, sim_config, scaler=scaler, reward=NetReturnReward())
        trayectoria = correr(env, accion=0.8)
        resultado = env.result
        assert resultado is not None

        equities = [info["equity"] for _, _, info in trayectoria]
        for i in range(1, len(trayectoria) - 1):
            esperado = (equities[i] - equities[i - 1]) / equities[i - 1]
            assert trayectoria[i][1] == pytest.approx(esperado)

    def test_el_ultimo_paso_no_paga_recompensa(self, serie, sim_config, scaler) -> None:
        """La orden de la ultima barra expira: no produjo nada."""
        env = TradingEnv(serie, sim_config, scaler=scaler)
        trayectoria = correr(env, accion=0.8)
        assert trayectoria[-1][2]["order_expired"] is True
        assert trayectoria[-1][1] == 0.0

    def test_el_differential_sharpe_se_reinicia_entre_episodios(
        self, serie, sim_config, scaler
    ) -> None:
        reward = DifferentialSharpeReward(eta=0.05)
        env = TradingEnv(serie, sim_config, scaler=scaler, reward=reward)
        correr(env, accion=0.5)
        assert reward.state != (0.0, 0.0)
        env.reset(seed=0)
        assert reward.state == (0.0, 0.0)


class TestCasosBorde:
    def test_primera_barra_del_episodio(self, serie, sim_config, scaler, builder):
        """En la primera decision no hay posicion ni orden anterior."""
        env = TradingEnv(serie, sim_config, scaler=scaler)
        obs, info = env.reset(seed=0)
        assert info["position"] == 0.0
        assert info["fill_status"] is None
        assert info["gate_reason"] is None
        assert obs[builder.names.index("last_order_rejected")] == 0.0
        assert obs[builder.names.index("last_order_blocked")] == 0.0

    def test_ultima_barra_expira(self, serie, sim_config, scaler) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        trayectoria = correr(env, accion=1.0)
        resultado = env.result
        assert resultado is not None
        expiradas = [f for f in resultado.fills if f.status is OrderStatus.EXPIRED]
        assert len(expiradas) == 1
        assert expiradas[0].t_decision == len(serie) - 1
        assert trayectoria[-1][2]["order_expired"] is True

    def test_episodio_de_longitud_uno(self, spec, builder, sim_config) -> None:
        """Serie con exactamente una barra despues del calentamiento."""
        n = builder.spec.warmup + 1
        serie = serie_deterministica(spec, n=n, seed=4)
        scaler = fit_scaler_on_train(
            serie_deterministica(spec, n=120, seed=4).slice(0, 80), builder
        )
        env = TradingEnv(serie, sim_config, scaler=scaler)
        _, info = env.reset(seed=0)
        assert info["t"] == builder.spec.warmup

        _, recompensa, terminado, truncado, info = env.step(np.array([1.0]))
        assert truncado is True
        assert terminado is False
        assert recompensa == 0.0
        assert info["order_expired"] is True

    def test_serie_mas_corta_que_el_calentamiento(
        self, spec, builder, sim_config, scaler
    ) -> None:
        serie = serie_deterministica(spec, n=builder.spec.warmup, seed=4)
        with pytest.raises(ValueError, match="calentamiento"):
            TradingEnv(serie, sim_config, scaler=scaler)

    def test_capital_insuficiente_para_operar(self, builder) -> None:
        """Con capital por debajo del minimo del venue no se llena nada.

        La orden se envia igual y el rechazo queda en el log: es lo que
        distingue "no quiso" de "no pudo" en el barrido de capital.
        """
        spec = us_equity_spec(
            "TEST",
            commission=CommissionSchema(kind="per_share", value=0.005, minimum=10.0),
        )
        serie = serie_deterministica(spec, n=70, seed=6)
        scaler = fit_scaler_on_train(serie.slice(0, 50), builder)
        env = TradingEnv(serie, SimConfig(initial_cash=50.0), scaler=scaler)
        correr(env, accion=1.0)
        resultado = env.result
        assert resultado is not None
        assert resultado.executed_fills == []
        assert len(resultado.rejected_fills) > 0

    def test_step_despues_del_final_es_un_error(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        correr(env, accion=0.5)
        with pytest.raises(RuntimeError, match="llama a reset"):
            env.step(np.array([0.5]))


class TestSerializacion:
    def test_describe_incluye_todo_lo_que_hace_falta_para_reproducir(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(
            serie,
            sim_config,
            scaler=scaler,
            env_config=EnvConfig(sizer=TargetWeightSizer(safety=0.95, deadband=0.01)),
        )
        d = env.describe()
        assert set(d) == {"env", "sim", "reward", "scaler", "series"}
        assert d["env"]["sizer"]["safety"] == 0.95  # type: ignore[index]
        assert d["env"]["sizer"]["deadband"] == 0.01  # type: ignore[index]
        assert d["scaler"]["n_samples"] == scaler.n_samples  # type: ignore[index]

    def test_el_escalador_va_y_vuelve(self, scaler) -> None:
        copia = FeatureScaler.from_dict(scaler.to_dict())
        np.testing.assert_array_equal(copia.mean, scaler.mean)
        np.testing.assert_array_equal(copia.std, scaler.std)
        assert copia.names == scaler.names
        assert copia.scale_mask == scaler.scale_mask


class TestSinteticoDeHeston:
    def test_el_entorno_corre_sobre_las_tres_regimenes(self, builder) -> None:
        """Requisito de la Etapa 3: el entorno tiene que andar sobre sintetico.

        Es una prueba de humo, no de aprendizaje. Ver el ADR y la nota del PR:
        el generador de Heston no tiene estructura direccional explotable, asi
        que "el agente aprende algo aca" no es un criterio valido de sanidad.
        """
        from data.synthetic import generate_gbm_sv

        for regimen in ("low", "medium", "high"):
            serie = generate_gbm_sv(120, seed=2, params=regimen)
            scaler = fit_scaler_on_train(serie.slice(0, 80), builder)
            env = TradingEnv(serie, SimConfig(initial_cash=100_000.0), scaler=scaler)
            trayectoria = correr(env, accion=0.6)
            assert len(trayectoria) > 1
            assert env.result is not None

    def test_tambien_con_datos_construidos_a_mano(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        assert len(correr(env, accion=0.4)) > 1


class TestNoPuenteaLaCapaDeRiesgo:
    def test_la_capa_se_construye_por_episodio(self, serie, sim_config, scaler) -> None:
        """Reusar la instancia arrastraria el kill switch de un episodio al otro."""
        creadas: list[RiskLayer] = []

        def gate() -> RiskLayer:
            capa = RiskLayer(
                RiskLimits(max_position_pct_equity=0.5), SimulatedClock(T0)
            )
            creadas.append(capa)
            return capa

        env = TradingEnv(serie, sim_config, scaler=scaler, gate_factory=gate)
        env.reset(seed=0)
        env.reset(seed=0)
        assert len(creadas) == 2
        assert creadas[0] is not creadas[1]

    def test_sin_capa_el_entorno_funciona_igual(
        self, serie, sim_config, scaler
    ) -> None:
        env = TradingEnv(serie, sim_config, scaler=scaler)
        correr(env, accion=0.5)
        assert env.result is not None
        assert env.result.gate_rejections == []

    def test_el_motor_recibe_la_capa_no_el_entorno(
        self, serie, sim_config, scaler
    ) -> None:
        """El veto lo aplica drive(), en el mismo punto que en vivo."""

        def gate() -> RiskLayer:
            return RiskLayer(
                RiskLimits(max_position_pct_equity=0.10), SimulatedClock(T0)
            )

        env = TradingEnv(serie, sim_config, scaler=scaler, gate_factory=gate)
        correr(env, accion=1.0)
        resultado = env.result
        assert resultado is not None
        # Los vetos salen del SimResult del motor, no de un registro del entorno.
        assert len(resultado.gate_rejections) > 0


class TestSimuladorNoContaminado:
    def test_el_entorno_no_deja_el_simulador_a_medio_correr(
        self, serie, sim_config, scaler
    ) -> None:
        """Cerrar el entorno cierra el generador sin dejar el venue colgado."""
        env = TradingEnv(serie, sim_config, scaler=scaler)
        env.reset(seed=0)
        env.step(np.array([0.5]))
        env.close()
        # Un simulador nuevo sobre la misma serie sigue corriendo normalmente.
        from agents.baselines import BuyAndHold

        resultado = Simulator(serie, sim_config).run(BuyAndHold())
        assert len(resultado.executed_fills) == 1

    def test_el_reject_reason_del_venue_es_del_enum(self, builder) -> None:
        spec = us_equity_spec(
            "TEST",
            commission=CommissionSchema(kind="per_share", value=0.005, minimum=10.0),
        )
        serie = serie_deterministica(spec, n=70, seed=6)
        scaler = fit_scaler_on_train(serie.slice(0, 50), builder)
        env = TradingEnv(serie, SimConfig(initial_cash=50.0), scaler=scaler)
        correr(env, accion=1.0)
        resultado = env.result
        assert resultado is not None
        for fill in resultado.rejected_fills:
            assert isinstance(fill.reject_reason, RejectReason)
