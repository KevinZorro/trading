"""Senales de recompensa. Se calculan **desde el equity del simulador**.

El principio 3 del proyecto dice que los costos entran dentro de la recompensa,
no restados en post-procesamiento. Aca eso sale gratis y no por disciplina: el
reward se calcula sobre ``equity_mark``, y el equity ya pago las comisiones, el
spread y el slippage cuando el ledger asento el fill. No hay nada que restar
despues porque nunca estuvo sumado.

Se usa ``equity_mark`` y no ``equity_liquidation``. La serie de liquidacion
cobra la friccion de salida en **cada** barra, cuando en la realidad se paga una
sola vez al salir: recompensar sobre ella penalizaria de mas a las posiciones
grandes y sesgaria al agente hacia operar chico por un motivo contable, no
economico. El riesgo conocido de usar la marca -que el agente infle el reward
manteniendo una posicion ilíquida que nunca podria cerrar- se mitiga exponiendo
la brecha entre ambas series como feature de la observacion, para que el agente
la vea en vez de ignorarla.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Por debajo de este denominador relativo, el Differential Sharpe no esta
# definido y devolver el cociente daria un numero enorme sin significado. Mismo
# criterio que `eval.metrics.DISPERSION_NULA_REL`, y por la misma razon: un
# reward de 1e15 domina cualquier entrenamiento sin querer decir nada.
VARIANZA_NULA_REL = 1e-12


@runtime_checkable
class RewardFn(Protocol):
    """Interfaz de recompensa. ``reset`` la llama el entorno en cada episodio."""

    @property
    def name(self) -> str: ...

    def reset(self) -> None: ...

    def compute(self, equity_prev: float, equity_now: float) -> float: ...

    def describe(self) -> dict[str, object]: ...


def _retorno(equity_prev: float, equity_now: float) -> float:
    """Retorno simple del periodo. Con equity previo no positivo, cero.

    El equity puede tocar cero: el proyecto ya lo contempla en ``eval``. Cuando
    ocurre no hay retorno definido, y devolver cero es preferible a propagar un
    ``inf`` que envenenaria el resto del episodio.
    """
    if equity_prev <= 0.0:
        return 0.0
    return (equity_now - equity_prev) / equity_prev


class NetReturnReward:
    """Retorno del periodo, neto de costos por construccion.

    El baseline honesto. Un agente que lo maximiza maximiza el retorno total, no
    el ajustado por riesgo: dos estrategias con el mismo retorno y volatilidades
    muy distintas reciben la misma recompensa. Para el estudio importa, y por eso
    existe tambien el Differential Sharpe.
    """

    def __init__(self, *, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("scale debe ser positivo")
        self.scale = scale

    @property
    def name(self) -> str:
        return "net_return"

    def reset(self) -> None:
        return None

    def compute(self, equity_prev: float, equity_now: float) -> float:
        return _retorno(equity_prev, equity_now) * self.scale

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "scale": self.scale}


class DifferentialSharpeReward:
    """Differential Sharpe Ratio (Moody & Saffell, 1998).

    Recompensa la contribucion **marginal** del retorno de esta barra al Sharpe
    acumulado, usando medias exponenciales de primer y segundo momento::

        dA = R_t - A_{t-1}
        dB = R_t^2 - B_{t-1}
        D_t = (B_{t-1} * dA - A_{t-1} * dB / 2) / (B_{t-1} - A_{t-1}^2)^{3/2}

    y despues ``A_t = A_{t-1} + eta*dA``, ``B_t = B_{t-1} + eta*dB``.

    Sirve para RL online donde el Sharpe clasico no: el Sharpe necesita toda la
    serie y solo se puede computar al final del episodio, asi que como reward
    llega demasiado tarde para asignar credito a una barra concreta.

    Los dos primeros pasos devuelven 0.0. Con ``A = B = 0`` el denominador es
    cero y el ratio no esta definido; inventar un numero ahi le daria al agente
    una senal enorme y arbitraria en el arranque de cada episodio.

    ASSUMPTION: ``eta`` fija la memoria efectiva en ~``1/eta`` barras. Con el
    valor por defecto son unas 100. Es un hiperparametro del reward, se serializa
    con la corrida y **no se toca contra el test**.
    """

    def __init__(self, *, eta: float = 0.01, scale: float = 1.0) -> None:
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta debe estar en (0, 1]")
        if scale <= 0:
            raise ValueError("scale debe ser positivo")
        self.eta = eta
        self.scale = scale
        self._a = 0.0
        self._b = 0.0

    @property
    def name(self) -> str:
        return "differential_sharpe"

    def reset(self) -> None:
        self._a = 0.0
        self._b = 0.0

    @property
    def state(self) -> tuple[float, float]:
        """``(A, B)``: primer y segundo momento suavizados. Para inspeccion."""
        return self._a, self._b

    def compute(self, equity_prev: float, equity_now: float) -> float:
        retorno = _retorno(equity_prev, equity_now)
        a_prev, b_prev = self._a, self._b
        delta_a = retorno - a_prev
        delta_b = retorno * retorno - b_prev

        varianza = b_prev - a_prev * a_prev
        escala = max(abs(b_prev), a_prev * a_prev)
        if varianza <= VARIANZA_NULA_REL * max(escala, 1.0):
            # Todavia no hay dispersion estimada: el ratio no esta definido.
            recompensa = 0.0
        else:
            numerador = b_prev * delta_a - 0.5 * a_prev * delta_b
            recompensa = numerador / (varianza**1.5)

        self._a = a_prev + self.eta * delta_a
        self._b = b_prev + self.eta * delta_b
        return recompensa * self.scale

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "eta": self.eta, "scale": self.scale}


REWARDS: dict[str, type[NetReturnReward] | type[DifferentialSharpeReward]] = {
    "net_return": NetReturnReward,
    "differential_sharpe": DifferentialSharpeReward,
}


def make_reward(name: str, **kwargs: float) -> RewardFn:
    """Fabrica por nombre, para que el reward entre por configuracion."""
    if name not in REWARDS:
        raise ValueError(f"reward desconocido {name!r}; opciones: {sorted(REWARDS)}")
    return REWARDS[name](**kwargs)
