"""Generacion de ``client_order_id``: el identificador lo pone el emisor.

Por que el emisor y no el venue: en vivo, si la respuesta a un envio se pierde
por timeout de red, el emisor no sabe si la orden llego. Reintentar sin un
identificador propio duplica la posicion. Con un ``client_order_id`` estable, el
reintento lleva el mismo identificador y el venue lo reconoce como duplicado en
vez de ejecutarlo de nuevo.

La tension de diseno esta entre dos requisitos que no se pueden satisfacer con
un solo generador:

- **Reproducibilidad total** (principio 7): dos corridas del mismo backtest con
  la misma semilla deben producir exactamente los mismos identificadores, o el
  log de una corrida no es comparable con el de otra. Eso descarta ``uuid4()``.
- **Unicidad entre reinicios**: en vivo, si el proceso se reinicia y el contador
  vuelve a cero, la primera orden despues del reinicio reusa un identificador ya
  visto y el venue la descarta como duplicada. La proteccion contra duplicados
  se vuelve una perdida silenciosa de ordenes legitimas.

Se resuelve con un generador **inyectable**: ``SequentialIds`` en simulacion,
determinista; ``PrefixedSequentialIds`` con un prefijo de instancia derivado del
``Clock`` para el adaptador de broker de la Etapa 7. La interfaz es la misma, y
por eso el motor no cambia entre uno y otro.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sim.clock import Clock


@runtime_checkable
class OrderIdGenerator(Protocol):
    """Produce identificadores unicos dentro de su ambito de unicidad."""

    def next_id(self) -> str: ...


class SequentialIds:
    """Contador determinista. El generador de simulacion.

    Mismo ``run_id`` y misma secuencia de ordenes producen exactamente los
    mismos identificadores, corrida tras corrida. **No usar en vivo**: un
    reinicio del proceso reinicia el contador.
    """

    __slots__ = ("_n", "run_id")

    def __init__(self, run_id: str = "sim") -> None:
        if not run_id:
            raise ValueError("run_id no puede ser vacio")
        self.run_id = run_id
        self._n = 0

    def next_id(self) -> str:
        identificador = f"{self.run_id}-{self._n:08d}"
        self._n += 1
        return identificador

    def reset(self) -> None:
        """Vuelve a cero. Lo llama el motor al empezar una corrida."""
        self._n = 0


class PrefixedSequentialIds:
    """Contador con prefijo de instancia tomado del reloj al arrancar.

    Pensado para el adaptador de broker de la Etapa 7. El prefijo se calcula una
    sola vez, en la construccion, y no vuelve a consultar el reloj: dentro de una
    misma instancia la secuencia sigue siendo determinista y auditable, pero dos
    arranques distintos del proceso no pueden colisionar.

    ASSUMPTION: dos instancias del runner no arrancan dentro del mismo
    nanosegundo. Si en algun momento se corren varios runners contra la misma
    cuenta, hay que agregar al prefijo un identificador de instancia explicito,
    no confiar en la resolucion del reloj.
    """

    __slots__ = ("_n", "_prefijo")

    def __init__(self, clock: Clock, run_id: str = "live") -> None:
        if not run_id:
            raise ValueError("run_id no puede ser vacio")
        self._prefijo = f"{run_id}-{clock.now().astype('datetime64[ns]').astype(int)}"
        self._n = 0

    def next_id(self) -> str:
        identificador = f"{self._prefijo}-{self._n:08d}"
        self._n += 1
        return identificador
