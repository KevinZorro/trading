"""Registro de experimentos: la configuracion viaja junto al resultado.

Un numero sin la configuracion que lo produjo no es un resultado, es una
anecdota. Este modulo guarda las dos cosas en el mismo archivo, y lo hace con el
reloj **inyectado**: ``src/`` no consulta el reloj de pared en ningun lado salvo
``sim.clock.SystemClock``, y hay un test que recorre el arbol para verificarlo.

Lo que se guarda incluye la procedencia (version de las dependencias, commit) y
no solo los hiperparametros. Un resultado reproducible necesita las dos: los
mismos hiperparametros con otra version de torch dan otro numero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim.clock import Clock, SystemClock


def jsonable(valor: Any) -> Any:
    """Convierte a algo serializable sin perder informacion en silencio.

    Los ``float`` de numpy, los ``datetime64`` y las tuplas atraviesan
    ``json.dumps`` mal o directamente fallan. Convertirlos aca -y fallar con un
    ``TypeError`` explicito cuando no se puede- es preferible a un
    ``default=str`` global, que convertiria un array entero en la cadena
    ``"[1. 2. 3.]"`` y lo haria irrecuperable.
    """
    if isinstance(valor, dict):
        return {str(k): jsonable(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [jsonable(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, np.ndarray):
        return [jsonable(v) for v in valor.tolist()]
    if isinstance(valor, np.datetime64):
        return str(valor)
    if isinstance(valor, Path):
        return str(valor)
    if valor is None or isinstance(valor, str | int | float | bool):
        return valor
    if hasattr(valor, "describe"):
        return jsonable(valor.describe())
    raise TypeError(
        f"no se puede serializar {type(valor).__name__}. Agregalo a jsonable() "
        "en vez de convertirlo a texto: un array serializado como cadena no se "
        "puede volver a leer."
    )


@dataclass(frozen=True)
class ExperimentRecord:
    """Un experimento completo: que se corrio, con que, y que salio."""

    name: str
    created_at: str
    config: dict[str, Any]
    provenance: dict[str, Any]
    results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "config": jsonable(self.config),
            "provenance": jsonable(self.provenance),
            "results": jsonable(self.results),
        }

    def save(self, directory: Path | str, *, filename: str | None = None) -> Path:
        """Escribe el JSON y devuelve la ruta.

        El nombre lleva el timestamp del reloj inyectado, no el del sistema, asi
        que dos corridas del mismo experimento con el mismo reloj simulado
        producen el mismo archivo: eso es lo que permite testear esta funcion.
        """
        carpeta = Path(directory)
        carpeta.mkdir(parents=True, exist_ok=True)
        nombre = filename or f"{self.name}_{self.created_at.replace(':', '-')}.json"
        ruta = carpeta / nombre
        ruta.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True))
        return ruta


@dataclass
class ExperimentLog:
    """Acumula resultados y los cierra en un :class:`ExperimentRecord`."""

    name: str
    config: dict[str, Any]
    clock: Clock = field(default_factory=SystemClock)
    provenance: dict[str, Any] = field(default_factory=dict)
    _results: dict[str, Any] = field(default_factory=dict, repr=False)

    def add(self, key: str, payload: Any) -> None:
        if key in self._results:
            raise KeyError(
                f"ya hay un resultado bajo {key!r}. Sobreescribirlo perderia el "
                "anterior sin dejar rastro"
            )
        self._results[key] = payload

    def record(self) -> ExperimentRecord:
        return ExperimentRecord(
            name=self.name,
            created_at=str(self.clock.now()),
            config=dict(self.config),
            provenance=dict(self.provenance),
            results=dict(self._results),
        )
