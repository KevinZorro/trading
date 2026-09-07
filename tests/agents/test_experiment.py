"""El registro de experimentos: la config viaja con el resultado, y el reloj se inyecta.

Un numero sin la configuracion que lo produjo es una anecdota. Y un modulo que
consulta el reloj de pared rompe la paridad backtest-live que el proyecto
sostiene desde la Etapa 1; hay un test que recorre ``src/`` para verificarlo, y
estos verifican que este modulo use el reloj **inyectado** de verdad.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.experiment import ExperimentLog, ExperimentRecord, jsonable
from sim.clock import SimulatedClock


def reloj() -> SimulatedClock:
    return SimulatedClock(np.datetime64("2024-01-15T00:00:00", "ns"))


def test_el_registro_usa_el_reloj_inyectado() -> None:
    """Con reloj simulado, dos corridas producen el mismo registro.

    Es lo que permite testear esta funcion; con ``datetime.now()`` habria que
    parchear el modulo, y parchear el tiempo es la senal de que el tiempo no
    estaba inyectado.
    """
    a = ExperimentLog(name="x", config={}, clock=reloj()).record()
    b = ExperimentLog(name="x", config={}, clock=reloj()).record()
    assert a.created_at == b.created_at == "2024-01-15T00:00:00.000000000"


def test_no_se_puede_pisar_un_resultado_en_silencio() -> None:
    log = ExperimentLog(name="x", config={}, clock=reloj())
    log.add("nivel_0", {"capture": 0.9})
    with pytest.raises(KeyError, match="ya hay un resultado"):
        log.add("nivel_0", {"capture": 0.1})


def test_el_archivo_lleva_config_procedencia_y_resultados() -> None:
    log = ExperimentLog(
        name="protocolo",
        config={"seeds": [1, 2, 3], "timesteps": 60_000},
        clock=reloj(),
        provenance={"torch": "2.14.0", "commit": "abc123"},
    )
    log.add("nivel_0", {"capture": 0.95})
    datos = log.record().to_dict()
    assert datos["config"]["timesteps"] == 60_000
    assert datos["provenance"]["torch"] == "2.14.0"
    assert datos["results"]["nivel_0"]["capture"] == 0.95


def test_se_guarda_como_json_legible(tmp_path: Path) -> None:
    log = ExperimentLog(name="protocolo", config={"a": 1}, clock=reloj())
    log.add("r", {"x": 1.5})
    ruta = log.record().save(tmp_path)
    assert ruta.exists()
    assert json.loads(ruta.read_text())["results"]["r"]["x"] == 1.5


def test_el_nombre_del_archivo_no_lleva_dos_puntos(tmp_path: Path) -> None:
    """Los timestamps ISO llevan ``:`` y hay sistemas de archivos que no."""
    ruta = ExperimentLog(name="p", config={}, clock=reloj()).record().save(tmp_path)
    assert ":" not in ruta.name


# ---------------------------------------------------------------------------
# Serializacion
# ---------------------------------------------------------------------------


def test_los_tipos_de_numpy_se_convierten_sin_perder_informacion() -> None:
    convertido = jsonable(
        {
            "float": np.float64(1.5),
            "int": np.int64(3),
            "array": np.array([1.0, 2.0]),
            "fecha": np.datetime64("2024-01-01", "ns"),
            "tupla": (1, 2),
        }
    )
    assert convertido["float"] == 1.5
    assert convertido["int"] == 3
    assert convertido["array"] == [1.0, 2.0]
    assert convertido["tupla"] == [1, 2]
    assert json.dumps(convertido)


def test_un_tipo_desconocido_falla_en_vez_de_volverse_texto() -> None:
    """``default=str`` convertiria un array en ``"[1. 2. 3.]"``, irrecuperable.

    Fallar obliga a agregar el tipo a ``jsonable``, que es una decision visible
    en el diff; convertirlo a texto es una perdida silenciosa.
    """

    class Raro:
        pass

    with pytest.raises(TypeError, match="Agregalo a jsonable"):
        jsonable({"x": Raro()})


def test_los_objetos_con_describe_se_serializan_solos() -> None:
    class ConDescribe:
        def describe(self) -> dict[str, object]:
            return {"a": np.float64(2.0)}

    assert jsonable({"x": ConDescribe()}) == {"x": {"a": 2.0}}


def test_el_record_es_json_puro() -> None:
    registro = ExperimentRecord(
        name="x",
        created_at="2024-01-15T00:00:00",
        config={"v": np.float64(1.0)},
        provenance={},
        results={"a": np.array([1, 2])},
    )
    texto = json.dumps(registro.to_dict())
    assert json.loads(texto)["results"]["a"] == [1, 2]
