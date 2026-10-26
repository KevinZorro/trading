"""Normalizacion ajustada **solo con datos de train**.

El anti-patron que este modulo evita esta escrito en CLAUDE.md: normalizar con
estadisticas calculadas sobre todo el dataset filtra informacion del test al
train. La media y el desvio de la serie completa dependen de barras que en el
momento de entrenar todavia no ocurrieron, y un modelo entrenado con esas
estadisticas ve un test que ya no es out-of-sample.

El escalador se ajusta una vez, queda inmutable, y se serializa junto al entorno
para que exactamente el mismo transformador se aplique en test y, en la Etapa 7,
en vivo. Un escalador que se reajusta en test o en produccion es un modelo
distinto del que se valido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.schema import FloatArray

# Un desvio por debajo de esto se trata como cero: la columna es constante en
# train y dividir por el la convertiria en ruido amplificado. Es el mismo
# criterio de tolerancia relativa que usa `eval.metrics.DISPERSION_NULA_REL`.
DESVIO_MINIMO = 1e-12


class ScalerError(ValueError):
    """El escalador no se puede ajustar o aplicar sobre lo que se le paso."""


def _momentos_sin_nan(matriz: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Media y desvio por columna ignorando NaN. Sin ningun valor: media 0, desvio 0."""
    validos = np.isfinite(matriz)
    n = validos.sum(axis=0)
    sin_datos = n == 0
    suma = np.where(validos, matriz, 0.0).sum(axis=0)
    media = np.where(sin_datos, 0.0, suma / np.maximum(n, 1))
    cuadrados = np.where(validos, (matriz - media) ** 2, 0.0).sum(axis=0)
    desvio = np.where(sin_datos, 0.0, np.sqrt(cuadrados / np.maximum(n, 1)))
    return media, desvio


@dataclass(frozen=True)
class FeatureScaler:
    """Z-score por columna, con mascara de que columnas se escalan.

    No todas las features quieren z-score. La fraccion de equity en posicion ya
    vive en ``[0, 1]``, y un indicador de "la orden anterior fue rechazada" es
    binario: normalizarlos los saca de su rango natural sin ganar nada. La
    mascara deja esas columnas pasar tal cual, y el hecho queda serializado.
    """

    names: tuple[str, ...]
    mean: FloatArray
    std: FloatArray
    scale_mask: tuple[bool, ...]
    n_samples: int
    # Columnas que pueden llegar enmascaradas (NaN): se ajustan sin los NaN y un NaN
    # sale como 0 tras escalar, o sea la media de train (ADR 0012, 3). Vacio = ninguna.
    maskable: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.names)
        if not (len(self.mean) == len(self.std) == len(self.scale_mask) == n):
            raise ScalerError(
                f"dimensiones inconsistentes: {n} nombres, {len(self.mean)} medias, "
                f"{len(self.std)} desvios, {len(self.scale_mask)} banderas"
            )
        if self.maskable:
            if len(self.maskable) != n:
                raise ScalerError("maskable debe tener un valor por columna")
            if any(
                m and not e for m, e in zip(self.maskable, self.scale_mask, strict=True)
            ):
                raise ScalerError("una columna maskable tiene que estar escalada")
        if np.any(self.std <= 0):
            raise ScalerError("todos los desvios deben ser positivos")

    @classmethod
    def fit(
        cls,
        samples: FloatArray,
        names: tuple[str, ...],
        scale_mask: tuple[bool, ...],
        maskable: tuple[bool, ...] = (),
    ) -> FeatureScaler:
        """Ajusta sobre la matriz de train. Nunca se llama con datos de test.

        Los NaN solo se admiten en columnas ``maskable`` y se excluyen del ajuste.
        """
        matriz = np.asarray(samples, dtype=np.float64)
        if matriz.ndim != 2:
            raise ScalerError(f"samples debe ser 2-D, tiene ndim={matriz.ndim}")
        if matriz.shape[0] == 0:
            raise ScalerError("no hay muestras para ajustar el escalador")
        if matriz.shape[1] != len(names):
            raise ScalerError(
                f"la matriz tiene {matriz.shape[1]} columnas y hay {len(names)} nombres"
            )
        if len(scale_mask) != len(names):
            raise ScalerError("scale_mask debe tener un valor por columna")
        if maskable and len(maskable) != len(names):
            raise ScalerError("maskable debe tener un valor por columna")
        enmascarable = np.asarray(maskable or (False,) * len(names), dtype=bool)
        if np.isinf(matriz).any() or np.isnan(matriz[:, ~enmascarable]).any():
            raise ScalerError("la matriz de ajuste tiene valores no finitos")

        if not enmascarable.any():
            media = np.mean(matriz, axis=0)
            desvio = np.std(matriz, axis=0, ddof=0)
        else:
            media, desvio = _momentos_sin_nan(matriz)
        # Columna constante en train: se deja pasar sin escalar en vez de
        # dividir por algo indistinguible de cero.
        desvio = np.where(desvio <= DESVIO_MINIMO, 1.0, desvio)
        mascara = np.asarray(scale_mask, dtype=bool)
        media = np.where(mascara, media, 0.0)
        desvio = np.where(mascara, desvio, 1.0)
        return cls(
            names=tuple(names),
            mean=media,
            std=desvio,
            scale_mask=tuple(scale_mask),
            n_samples=int(matriz.shape[0]),
            maskable=tuple(maskable) if any(maskable) else (),
        )

    @classmethod
    def identity(
        cls, names: tuple[str, ...], scale_mask: tuple[bool, ...] | None = None
    ) -> FeatureScaler:
        """Escalador que no transforma nada. Util para tests y para ablaciones."""
        n = len(names)
        return cls(
            names=tuple(names),
            mean=np.zeros(n),
            std=np.ones(n),
            scale_mask=scale_mask or (False,) * n,
            n_samples=0,
        )

    def transform(self, x: FloatArray) -> FloatArray:
        """Aplica el z-score. **Nunca reajusta.**"""
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape[-1] != len(self.names):
            raise ScalerError(
                f"se esperaban {len(self.names)} features y llegaron {vector.shape[-1]}"
            )
        salida: FloatArray = (vector - self.mean) / self.std
        if any(self.maskable):
            mascara = np.asarray(self.maskable, dtype=bool)
            # Solo las maskable: un NaN en otra columna sigue siendo un error.
            salida = np.where(mascara & np.isnan(salida), 0.0, salida)
        return salida

    # -- serializacion --------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Forma serializable, para guardar junto a la config de la corrida."""
        datos: dict[str, object] = {
            "names": list(self.names),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "scale_mask": [bool(v) for v in self.scale_mask],
            "n_samples": self.n_samples,
        }
        if any(self.maskable):  # sin columnas enmascarables, la forma es la de siempre
            datos["maskable"] = [bool(v) for v in self.maskable]
        return datos

    @classmethod
    def from_dict(cls, datos: dict[str, object]) -> FeatureScaler:
        return cls(
            names=tuple(datos["names"]),  # type: ignore[arg-type]
            mean=np.asarray(datos["mean"], dtype=np.float64),
            std=np.asarray(datos["std"], dtype=np.float64),
            scale_mask=tuple(datos["scale_mask"]),  # type: ignore[arg-type]
            n_samples=int(datos["n_samples"]),  # type: ignore[call-overload]
            maskable=tuple(datos.get("maskable", ())),  # type: ignore[arg-type]
        )
