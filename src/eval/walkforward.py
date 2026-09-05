"""Ventanas rodantes para validacion fuera de muestra.

El proyecto prohibe el split simple: entrenar en el 80% inicial y medir en el
20% final da **una** observacion de desempeno fuera de muestra, y esa
observacion depende por completo de que regimen tocaba en ese tramo. Con
ventanas rodantes hay tantas observaciones como ventanas, y la dispersion entre
ellas es informacion, no ruido a promediar.

Tres cosas que esta implementacion garantiza por construccion, no por
disciplina:

1. **Los tres tramos de una ventana son contiguos y en orden**: train,
   validacion, test. Ningun indice de test aparece en el train de su propia
   ventana.
2. **El test de una ventana nunca precede a su train.** Entrenar con datos
   posteriores al periodo evaluado es la forma mas costosa de lookahead porque
   no rompe nada: simplemente da resultados buenisimos.
3. **Los tramos de test no se solapan entre ventanas** cuando el paso es al
   menos del tamano del test, asi que las observaciones fuera de muestra no se
   cuentan dos veces.

Sobre el conjunto de test: se toca **una vez**, al final. Los hiperparametros se
ajustan contra el tramo de validacion de cada ventana. Esta clase no puede
impedirlo -nada puede-, pero mantiene los dos tramos separados y nombrados para
que usar el equivocado sea una decision visible y no un descuido.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import BarSeries
from eval.metrics import MetricError


class WalkForwardError(MetricError):
    """La particion pedida no se puede construir."""


@dataclass(frozen=True)
class Window:
    """Una ventana: tres tramos contiguos ``[inicio, fin)`` sobre la serie."""

    index: int
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]

    def __post_init__(self) -> None:
        tramos = (self.train, self.validation, self.test)
        for nombre, (a, b) in zip(("train", "validation", "test"), tramos, strict=True):
            if b <= a:
                raise WalkForwardError(f"el tramo {nombre} esta vacio: [{a}, {b})")
        if not (
            self.train[1] == self.validation[0] and self.validation[1] == self.test[0]
        ):
            raise WalkForwardError(
                f"los tramos no son contiguos: {self.train}, {self.validation}, "
                f"{self.test}"
            )

    @property
    def start(self) -> int:
        return self.train[0]

    @property
    def stop(self) -> int:
        return self.test[1]

    def __len__(self) -> int:
        return self.stop - self.start

    def slice(self, series: BarSeries, segment: str) -> BarSeries:
        """Sub-serie de un tramo. ``segment`` es el nombre, no un indice.

        Que haya que nombrar el tramo es deliberado: ``ventana.slice(serie,
        "test")`` se lee y se audita; ``serie.slice(*ventana[2])`` no.
        """
        if segment not in ("train", "validation", "test"):
            raise WalkForwardError(f"tramo desconocido: {segment!r}")
        inicio, fin = getattr(self, segment)
        return series.slice(inicio, fin)

    def describe(self) -> dict[str, object]:
        return {
            "index": self.index,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def rolling_windows(
    n_bars: int,
    *,
    train: int,
    validation: int,
    test: int,
    step: int | None = None,
    anchored: bool = False,
    min_bars_per_segment: int = 1,
) -> tuple[Window, ...]:
    """Ventanas rodantes sobre una serie de ``n_bars`` barras.

    ``step`` por defecto es ``test``, que hace que los tramos de test
    particionen la serie sin solaparse: cada barra fuera de muestra se evalua
    exactamente una vez. Un paso menor solapa los tests y **cuenta dos veces**
    las mismas barras, lo cual infla artificialmente el numero de observaciones
    y hace que cualquier test de significancia sobreestime.

    ``anchored=True`` da ventanas expansivas: el train arranca siempre en 0 y
    crece. Es lo correcto cuando el modelo mejora con mas historia y no hay
    sospecha de cambio de regimen; ``anchored=False`` (rodante) es lo correcto
    cuando la relacion cambia con el tiempo, que es la hipotesis del nivel 3 de
    los fixtures y de cualquier mercado real. No hay un default universal, asi
    que el parametro es explicito y viaja serializado.

    ``min_bars_per_segment`` sirve para exigir que cada tramo cubra al menos el
    calentamiento de los indicadores: un tramo mas corto que el calentamiento no
    produce ni una observacion utilizable.
    """
    paso = test if step is None else step
    for nombre, valor in (
        ("train", train),
        ("validation", validation),
        ("test", test),
        ("step", paso),
    ):
        if valor < 1:
            raise WalkForwardError(f"{nombre} debe ser positivo, se paso {valor}")
    for nombre, valor in (("train", train), ("validation", validation), ("test", test)):
        if valor < min_bars_per_segment:
            raise WalkForwardError(
                f"el tramo {nombre} tiene {valor} barras y el minimo es "
                f"{min_bars_per_segment}: no alcanza ni para el calentamiento"
            )
    total = train + validation + test
    if total > n_bars:
        raise WalkForwardError(
            f"una ventana necesita {total} barras y la serie tiene {n_bars}"
        )

    ventanas: list[Window] = []
    desplazamiento = 0
    while True:
        fin = desplazamiento + total
        if fin > n_bars:
            break
        inicio_train = 0 if anchored else desplazamiento
        corte_train = desplazamiento + train
        corte_val = corte_train + validation
        ventanas.append(
            Window(
                index=len(ventanas),
                train=(inicio_train, corte_train),
                validation=(corte_train, corte_val),
                test=(corte_val, fin),
            )
        )
        desplazamiento += paso
    if not ventanas:  # pragma: no cover - lo cubre la comprobacion de arriba
        raise WalkForwardError("no entra ninguna ventana en la serie")
    return tuple(ventanas)


def coverage(windows: tuple[Window, ...], n_bars: int) -> dict[str, object]:
    """Cuanto de la serie queda evaluado fuera de muestra, y si se repite.

    ``overlapping_test_bars`` distinto de cero significa que hay barras contadas
    mas de una vez: cualquier estadistico calculado sobre esas observaciones
    tiene menos informacion independiente de la que su tamano sugiere, y hay que
    decirlo antes de calcular un p-value con ellas.
    """
    vistas: dict[int, int] = {}
    for ventana in windows:
        for t in range(*ventana.test):
            vistas[t] = vistas.get(t, 0) + 1
    repetidas = sum(1 for c in vistas.values() if c > 1)
    return {
        "n_windows": len(windows),
        "test_bars": len(vistas),
        "test_coverage": len(vistas) / n_bars if n_bars else 0.0,
        "overlapping_test_bars": repetidas,
        "unused_tail": n_bars - windows[-1].stop,
    }
