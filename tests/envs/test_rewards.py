"""Recompensas, con el Differential Sharpe calculado a mano desde su definicion."""

from __future__ import annotations

import pytest

from envs.rewards import (
    DifferentialSharpeReward,
    NetReturnReward,
    RewardFn,
    make_reward,
)


class TestRetornoNeto:
    def test_es_el_retorno_simple_del_periodo(self) -> None:
        assert NetReturnReward().compute(100.0, 110.0) == pytest.approx(0.10)
        assert NetReturnReward().compute(100.0, 90.0) == pytest.approx(-0.10)

    def test_los_costos_ya_estan_dentro(self) -> None:
        """No hay nada que restar despues porque nunca estuvo sumado.

        El equity que llega aca ya pago comision, spread y slippage cuando el
        ledger asento el fill. La recompensa es la diferencia de ese equity.
        """
        sin_costos = NetReturnReward().compute(100.0, 110.0)
        con_costos = NetReturnReward().compute(100.0, 109.0)
        assert con_costos < sin_costos
        assert con_costos == pytest.approx(0.09)

    def test_escala_configurable(self) -> None:
        assert NetReturnReward(scale=100.0).compute(100.0, 110.0) == pytest.approx(10.0)

    def test_equity_previo_no_positivo_da_cero(self) -> None:
        """Sin base no hay retorno; un inf envenenaria el resto del episodio."""
        assert NetReturnReward().compute(0.0, 10.0) == 0.0
        assert NetReturnReward().compute(-5.0, 10.0) == 0.0

    def test_no_guarda_estado(self) -> None:
        reward = NetReturnReward()
        primero = reward.compute(100.0, 110.0)
        reward.compute(100.0, 50.0)
        assert reward.compute(100.0, 110.0) == primero

    def test_escala_invalida(self) -> None:
        with pytest.raises(ValueError, match="scale debe ser positivo"):
            NetReturnReward(scale=0.0)


class TestDifferentialSharpe:
    """Oraculo: la recursion de Moody & Saffell escrita a mano.

    dA = R - A ; dB = R^2 - B ; var = B - A^2
    D  = (B*dA - A*dB/2) / var^{3/2}
    A += eta*dA ; B += eta*dB
    """

    def test_los_primeros_pasos_dan_cero(self) -> None:
        """Con A = B = 0 el denominador es cero y el ratio no esta definido.

        Inventar un numero ahi le daria al agente una senal enorme y arbitraria
        en el arranque de cada episodio.
        """
        reward = DifferentialSharpeReward(eta=0.5)
        assert reward.compute(100.0, 110.0) == 0.0

    def test_segundo_paso_calculado_a_mano(self) -> None:
        # eta = 0.5. Primer retorno R1 = 0.1:
        #   dA = 0.1 ; dB = 0.01 ; var = 0 -> D = 0
        #   A = 0.05 ; B = 0.005
        # Segundo retorno R2 = 0.1:
        #   dA = 0.1 - 0.05 = 0.05 ; dB = 0.01 - 0.005 = 0.005
        #   var = 0.005 - 0.05^2 = 0.005 - 0.0025 = 0.0025
        #   numerador = 0.005*0.05 - 0.5*0.05*0.005 = 0.00025 - 0.000125 = 0.000125
        #   var^{3/2} = 0.0025^{1.5} = 0.000125
        #   D = 0.000125 / 0.000125 = 1.0 exacto
        reward = DifferentialSharpeReward(eta=0.5)
        reward.compute(100.0, 110.0)
        assert reward.compute(100.0, 110.0) == pytest.approx(1.0)

    def test_el_estado_avanza_como_dicta_la_recursion(self) -> None:
        reward = DifferentialSharpeReward(eta=0.5)
        reward.compute(100.0, 110.0)
        assert reward.state == pytest.approx((0.05, 0.005))
        reward.compute(100.0, 110.0)
        assert reward.state == pytest.approx((0.075, 0.0075))

    def test_un_retorno_peor_que_la_media_da_recompensa_negativa(self) -> None:
        reward = DifferentialSharpeReward(eta=0.5)
        reward.compute(100.0, 110.0)
        reward.compute(100.0, 110.0)
        assert reward.compute(100.0, 95.0) < 0.0

    def test_premia_el_retorno_ajustado_por_riesgo_no_el_crudo(self) -> None:
        """Dos series con el mismo retorno acumulado y distinta volatilidad.

        La que llega con menos varianza acumula mas Differential Sharpe. Es la
        diferencia con el retorno neto, que las puntuaria igual.
        """
        estable = DifferentialSharpeReward(eta=0.1)
        volatil = DifferentialSharpeReward(eta=0.1)
        suma_estable = sum(estable.compute(100.0, 100.0 * (1 + r)) for r in [0.01] * 8)
        suma_volatil = sum(
            volatil.compute(100.0, 100.0 * (1 + r))
            for r in [0.09, -0.07, 0.09, -0.07, 0.09, -0.07, 0.09, -0.07]
        )
        assert suma_estable > suma_volatil

    def test_reset_vuelve_al_arranque(self) -> None:
        reward = DifferentialSharpeReward(eta=0.5)
        reward.compute(100.0, 110.0)
        assert reward.state != (0.0, 0.0)
        reward.reset()
        assert reward.state == (0.0, 0.0)
        assert reward.compute(100.0, 110.0) == 0.0

    def test_eta_invalida(self) -> None:
        with pytest.raises(ValueError, match="eta debe estar"):
            DifferentialSharpeReward(eta=0.0)
        with pytest.raises(ValueError, match="eta debe estar"):
            DifferentialSharpeReward(eta=1.5)

    def test_una_serie_constante_no_explota(self) -> None:
        """Sin dispersion el ratio no esta definido: cero, no 1e15."""
        reward = DifferentialSharpeReward(eta=0.5)
        for _ in range(10):
            assert reward.compute(100.0, 100.0) == 0.0


class TestFabrica:
    def test_construye_por_nombre(self) -> None:
        assert isinstance(make_reward("net_return"), NetReturnReward)
        assert isinstance(
            make_reward("differential_sharpe", eta=0.02), DifferentialSharpeReward
        )

    def test_nombre_desconocido(self) -> None:
        with pytest.raises(ValueError, match="reward desconocido"):
            make_reward("sortino_diferencial")

    def test_ambos_satisfacen_el_protocolo(self) -> None:
        assert isinstance(NetReturnReward(), RewardFn)
        assert isinstance(DifferentialSharpeReward(), RewardFn)

    def test_describe_es_serializable(self) -> None:
        import json

        assert json.loads(json.dumps(DifferentialSharpeReward(eta=0.02).describe()))
