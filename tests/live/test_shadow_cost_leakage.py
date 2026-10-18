"""Anti-leakage de ``model_cost``: el modelo no ve barras que aun no cerraron.

El ``timestamp`` de una barra es su cierre (convencion del repo). Una barra con
``timestamp > as_of`` tiene high, low y volumen que en ``as_of`` no existen:
recortarla en silencio haria pasar por valida una llamada con datos del futuro.
"""

from __future__ import annotations

import numpy as np
import pytest

from live.shadow_cost import ShadowCostError, model_cost

from .test_shadow_cost import SIM, barras

pytestmark = pytest.mark.leakage

CIERRE_ULTIMA = np.datetime64("2026-10-16T21:00:00", "ns")


def llamar(series: object, as_of: np.datetime64) -> None:
    model_cost(
        series,  # type: ignore[arg-type]
        mid=100.0,
        notional=10.0,
        as_of=as_of,
        spread=SIM.spread,
        slippage=SIM.slippage,
    )


def test_una_barra_posterior_a_as_of_lanza_y_no_se_recorta() -> None:
    series = barras([(101.0, 100.0)] * 3, volumen=1000.0)  # cierran 15, 16 y 17
    with pytest.raises(ShadowCostError, match=r"lookahead: 1 barra\(s\)"):
        llamar(series, CIERRE_ULTIMA)


def test_un_nanosegundo_antes_del_cierre_ya_es_futuro() -> None:
    series = barras([(101.0, 100.0)] * 2, volumen=1000.0)
    with pytest.raises(ShadowCostError, match="lookahead"):
        llamar(series, CIERRE_ULTIMA - np.timedelta64(1, "ns"))


def test_la_barra_que_cierra_justo_en_as_of_es_usable() -> None:
    llamar(barras([(101.0, 100.0)] * 2, volumen=1000.0), CIERRE_ULTIMA)


def test_con_una_sola_barra_lanza() -> None:
    with pytest.raises(ShadowCostError, match=r">= 2 barras cerradas.*hay 1"):
        llamar(barras([(101.0, 100.0)], volumen=1000.0), CIERRE_ULTIMA)


def test_el_resultado_no_depende_de_barras_posteriores_a_as_of() -> None:
    # Cambiar high/low/volumen de una barra futura no puede mover el resultado:
    # no puede ni entrar (lanza), asi que el calculo solo ve el pasado.
    cerradas = barras([(101.0, 100.0)] * 2, volumen=1000.0)
    a = model_cost(
        cerradas,
        mid=100.0,
        notional=10.0,
        as_of=AS_OF_DESPUES,
        spread=SIM.spread,
        slippage=SIM.slippage,
    )
    b = model_cost(
        cerradas,
        mid=100.0,
        notional=10.0,
        as_of=AS_OF_DESPUES + np.timedelta64(5, "D"),
        spread=SIM.spread,
        slippage=SIM.slippage,
    )
    assert a == b


AS_OF_DESPUES = np.datetime64("2026-10-16T21:00:30", "ns")
