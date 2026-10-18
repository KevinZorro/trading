"""Invariantes del despliegue, leidas del texto de los archivos (sin Docker)."""

from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
COMPOSE = (RAIZ / "docker-compose.yml").read_text()
DOCKERFILE = (RAIZ / "Dockerfile").read_text()


def test_imagen_copia_docs_porque_la_sombra_exige_el_adr() -> None:
    assert re.search(r"^COPY docs \./docs$", DOCKERFILE, re.MULTILINE)
    assert "uv sync --frozen --no-dev" in DOCKERFILE


def test_imagen_no_corre_como_root() -> None:
    assert re.search(r"^USER 10001$", DOCKERFILE, re.MULTILINE)


def test_ambos_servicios_reinician_solos() -> None:
    assert "restart: unless-stopped" in COMPOSE  # en el ancla comun
    assert COMPOSE.count("<<: *comun") == 2


def test_watchdog_monta_los_datos_en_solo_lectura_y_sin_socket_de_docker() -> None:
    assert COMPOSE.count(":/data:ro") == 1
    assert "docker.sock" not in COMPOSE


def test_ningun_secreto_en_los_archivos_versionados() -> None:
    for texto in (COMPOSE, DOCKERFILE):
        assert not re.search(r"TELEGRAM_TOKEN\s*[:=]\s*\S", texto)


def test_dockerignore_deja_los_secretos_fuera_de_la_imagen() -> None:
    lineas = (RAIZ / ".dockerignore").read_text().splitlines()
    assert ".env" in lineas
    assert ".git" in lineas
