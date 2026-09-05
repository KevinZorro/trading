"""La interfaz de un agente sobre el entorno, y las politicas que no aprenden.

``Strategy`` (en ``sim.engine``) decide con ``MarketView`` y ``AccountSnapshot``;
un agente de RL decide con el vector de observacion normalizado que le entrega
el entorno. Son dos firmas distintas porque son dos contratos distintos, y
forzar una sola haria que el agente recibiera la serie -que es justo lo que la
Etapa 2 se ocupo de impedir-.

``Policy`` es la firma del lado del entorno. Los baselines siguen siendo
``Strategy`` y corren contra el motor directamente: no se los envuelve en un
entorno para compararlos, porque eso agregaria el escalador y el sizer a un
baseline que no los necesita y la comparacion dejaria de ser contra el baseline.
Lo que si se iguala es el punto de arranque, con ``WarmupDelay``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from data.schema import FloatArray
from sim.engine import LONG_ONLY_ACTION_RANGE, Strategy
from sim.orders import MarketOrder
from sim.view import AccountSnapshot, MarketView


@runtime_checkable
class Policy(Protocol):
    """Decide un peso objetivo a partir de la observacion del entorno.

    Devuelve un escalar en ``LONG_ONLY_ACTION_RANGE``. Devolver algo fuera de
    rango no es un error de la politica -el entorno lo recorta y lo registra en
    ``ClipEvent``- pero un agente que recorta sistematicamente esta aprendiendo
    sobre un rango que no existe, y por eso el recorte queda contado.
    """

    @property
    def name(self) -> str: ...

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: FloatArray) -> float: ...

    def describe(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ConstantWeightPolicy:
    """Peso fijo en cada barra. ``weight=1.0`` es la referencia ``always_long``.

    No es ``agents.baselines.BuyAndHold``: esta rebalancea a peso constante en
    cada barra, que es lo que produce un agente cuya accion es siempre 1.0.
    Comprar una vez y no rebalancear es otra cosa, y las dos aparecen en el
    reporte para que la diferencia se vea en vez de discutirse.
    """

    weight: float = 1.0

    def __post_init__(self) -> None:
        bajo, alto = LONG_ONLY_ACTION_RANGE
        if not bajo <= self.weight <= alto:
            raise ValueError(f"weight fuera de {LONG_ONLY_ACTION_RANGE}")

    @property
    def name(self) -> str:
        return f"constant_weight_{self.weight:g}"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: FloatArray) -> float:
        return self.weight

    def describe(self) -> dict[str, object]:
        return {"policy": "constant_weight", "weight": self.weight}


class RandomWeightPolicy:
    """Peso uniforme en ``[0, 1]`` con semilla propia.

    Separa "el agente aprendio algo" de "el entorno tiene un sesgo": una
    politica aleatoria sobre un activo con drift positivo gana plata, y sin este
    punto de comparacion ese retorno se le atribuiria al aprendizaje.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "random_weight"

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def act(self, observation: FloatArray) -> float:
        return float(self._rng.uniform(*LONG_ONLY_ACTION_RANGE))

    def describe(self) -> dict[str, object]:
        return {"policy": "random_weight", "seed": self.seed}


class WarmupDelay:
    """Envuelve una ``Strategy`` para que no opere antes del calentamiento.

    Sin esto la comparacion esta sesgada y no de forma menor: el entorno empieza
    a decidir en la barra ``warmup`` -antes, los indicadores no estan definidos-
    mientras que ``BuyAndHold`` compra en la barra 0. Sobre una serie con drift,
    esas barras de ventaja son retorno regalado al baseline; sobre una serie que
    empieza cayendo, son retorno regalado al agente. En los dos casos la
    diferencia medida deja de ser atribuible a la estrategia.

    El techo se pide con el mismo ``first_decision`` por la misma razon.
    """

    def __init__(self, inner: Strategy, warmup: int) -> None:
        if warmup < 0:
            raise ValueError("warmup no puede ser negativo")
        if not isinstance(inner, Strategy):
            raise TypeError(
                "WarmupDelay envuelve una Strategy (on_bar con vista y cuenta), "
                "no una Policy"
            )
        self._inner = inner
        self.warmup = warmup
        self.name = inner.name

    @property
    def inner(self) -> Strategy:
        return self._inner

    def reset(self, seed: int | None = None) -> None:
        self._inner.reset(seed)

    def on_bar(self, view: MarketView, account: AccountSnapshot) -> MarketOrder | None:
        if view.t < self.warmup:
            return None
        return self._inner.on_bar(view, account)
