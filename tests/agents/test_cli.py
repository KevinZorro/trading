"""El corredor del protocolo. Lo que se testea es el ensamblado, no el entrenamiento.

``assemble`` aplica los criterios sobre resultados guardados, asi que se puede
verificar entero sin torch: se fabrican los JSON de los brazos y se comprueba
que el protocolo los encuentre, los juzgue y pare donde tiene que parar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.cli import FIXTURE_SEED, STUDY_SEEDS, build_fixture, main
from tests.agents.test_protocol import brazo

SEED = FIXTURE_SEED


def escribir(directorio: Path, arm: object) -> None:
    ruta = directorio / f"arm_{arm.label}.json"  # type: ignore[attr-defined]
    ruta.write_text(json.dumps(arm.describe()))  # type: ignore[attr-defined]


def poblar(directorio: Path, *, capture_nivel_0: float = 0.95) -> None:
    escribir(directorio, brazo("level_0", capture=capture_nivel_0))
    barrido = (
        (0.25, 0.9, -0.5),
        (0.09, 0.8, -0.3),
        (0.04, 0.7, -0.2),
        (0.01, 0.6, -0.1),
    )
    for r2, cap, beta in barrido:
        escribir(directorio, brazo(f"level_1_beta{beta}", r_squared=r2, capture=cap))
    escribir(directorio, brazo("level_2_beta-0.3", turnover_annualized=10.0))
    escribir(directorio, brazo("level_3_beta-0.3"))
    escribir(directorio, brazo("level_3_beta-0.3_lstm"))
    escribir(
        directorio,
        brazo(
            "level_4",
            time_invested=0.95,
            excess_log_growth_vs_always_long=0.0,
            turnover_annualized=0.08,
        ),
    )


def test_las_semillas_del_estudio_son_diez_y_estan_fijas() -> None:
    """Un rango generado al vuelo hace que "10 semillas" signifique cosas
    distintas en dos corridas."""
    assert len(STUDY_SEEDS) == 10
    assert len(set(STUDY_SEEDS)) == 10


def test_la_semilla_del_fixture_es_distinta_de_las_del_agente() -> None:
    """Las 10 semillas miden la varianza del **entrenamiento** sobre una serie
    fija. Mezclarla con la varianza de la serie daria un solo numero para dos
    fuentes distintas."""
    assert FIXTURE_SEED not in STUDY_SEEDS


@pytest.mark.parametrize(
    ("nivel", "esperado"),
    [
        ("level_0", 0),
        ("level_1", 1),
        ("level_2", 2),
        ("level_3", 3),
        ("level_4", 4),
    ],
)
def test_cada_nivel_construye_su_fixture(nivel: str, esperado: int) -> None:
    fixture = build_fixture(nivel, beta=-0.3, n_bars=600)
    assert fixture.level == esperado
    assert fixture.series.source.startswith("fixture:")


def test_un_nivel_desconocido_falla_con_su_motivo() -> None:
    with pytest.raises(SystemExit, match="nivel desconocido"):
        build_fixture("level_9", beta=None, n_bars=600)


def test_assemble_aplica_los_criterios_y_escribe_el_reporte(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    poblar(tmp_path)
    codigo = main(["assemble", "--out", str(tmp_path)])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "PROTOCOLO DE VALIDACION" in salida
    assert "Protocolo completo sin fallos" in salida
    guardado = json.loads((tmp_path / "protocol.json").read_text())
    assert guardado["results"]["protocol"]["stopped_at"] is None
    niveles = guardado["results"]["protocol"]["levels"]
    assert [n["level"] for n in niveles] == [0, 1, 2, 3, 4]


def test_assemble_devuelve_codigo_distinto_de_cero_si_el_protocolo_para(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El codigo de salida es lo que hace que un fallo se note en un pipeline."""
    poblar(tmp_path, capture_nivel_0=0.2)
    assert main(["assemble", "--out", str(tmp_path)]) == 1
    assert "PROTOCOLO DETENIDO en el nivel 0" in capsys.readouterr().out


def test_assemble_falla_sin_brazos(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no hay brazos"):
        main(["assemble", "--out", str(tmp_path)])


def test_assemble_falla_sin_el_nivel_0(tmp_path: Path) -> None:
    """El protocolo arranca en el nivel 0; sin el no hay nada que ordenar."""
    escribir(tmp_path, brazo("level_4"))
    with pytest.raises(SystemExit, match="falta el brazo del nivel 0"):
        main(["assemble", "--out", str(tmp_path)])


def test_la_procedencia_queda_guardada(tmp_path: Path) -> None:
    """Los mismos hiperparametros con otra version de torch dan otro numero."""
    poblar(tmp_path)
    main(
        [
            "assemble",
            "--out",
            str(tmp_path),
            "--provenance",
            json.dumps({"torch": "2.14.0", "commit": "abc"}),
        ]
    )
    guardado = json.loads((tmp_path / "protocol.json").read_text())
    assert guardado["provenance"]["torch"] == "2.14.0"


def test_el_brazo_lstm_no_entra_en_el_barrido_de_snr(tmp_path: Path) -> None:
    """El nivel 1 compara arquitecturas iguales con SNR distinto.

    Colar el brazo recurrente ahi mezclaria dos variables y la degradacion
    dejaria de ser atribuible al SNR.
    """
    poblar(tmp_path)
    escribir(tmp_path, brazo("level_1_beta-0.3_lstm", r_squared=0.09, capture=0.99))
    main(["assemble", "--out", str(tmp_path)])
    guardado = json.loads((tmp_path / "protocol.json").read_text())
    nivel_1 = guardado["results"]["protocol"]["levels"][1]
    assert all(not a["label"].endswith("_lstm") for a in nivel_1["arms"])


def test_los_brazos_se_reordenan_por_snr(tmp_path: Path) -> None:
    poblar(tmp_path)
    main(["assemble", "--out", str(tmp_path)])
    guardado = json.loads((tmp_path / "protocol.json").read_text())
    nivel_1 = guardado["results"]["protocol"]["levels"][1]
    etiquetas = [a["label"] for a in nivel_1["arms"]]
    assert etiquetas == [
        "level_1_beta-0.5",
        "level_1_beta-0.3",
        "level_1_beta-0.2",
        "level_1_beta-0.1",
    ]
