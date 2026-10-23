"""Grabador en vivo de noticias GDELT (Etapa 4, pista L, PR L1).

Cada 15 minutos GDELT publica un lote del GKG. Este proceso lo baja, lo filtra con
:func:`data.gdelt.parse_gkg` y agrega una linea por lote a ``news-AAAA-MM-DD.jsonl``
(append con ``fsync``, un archivo por dia UTC del rotulo). Guarda lo que el
historico no puede dar: la **hora de recepcion propia**, y el ultimo 404 antes del
exito, que acota la latencia real entre el rotulo y la disponibilidad.

Medido el 2026-10-20: GDELT publica el archivo hasta ~11 min *antes* de su rotulo
(el lote 01:15 tiene ``Last-Modified`` 01:03:41), y hay lotes que nunca existen
(el 01:30 del mismo dia da 404). Por eso se empieza a sondear ``lead_minutes``
antes del rotulo, y un lote que no aparece en ``give_up_minutes`` queda como
``status: "missing"``: el hueco es explicito, nunca un reintento infinito ni un
lote inventado.

No decide ``pub_time``. La regla anti-lookahead (rotulo + latencia de 30 min) la
aplica la alineacion de la Etapa 4; aqui solo se registran los hechos.

ASSUMPTION: un lote ``missing`` por errores transitorios (no por 404) se puede
recuperar despues con la descarga historica; el registro dice cuantos intentos
hubo y el ultimo error.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import signal
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.gdelt import (
    GKG_BASE_URL,
    LOTE_MINUTOS,
    THEME_BITCOIN,
    GkgBatch,
    floor_label,
    gkg_url,
    parse_gkg,
)
from live.config import ConfigError
from live.health import Health, parse_ts, read_heartbeat
from live.logs import configure_json_logging
from live.shadow import dia, iso
from live.stall import StallGuard, exit_process
from live.store import SnapshotStore, StoreError, _linea
from sim.clock import Clock, SystemClock

log = logging.getLogger(__name__)
PREFIJO = "NEWS_"
PASO = np.timedelta64(LOTE_MINUTOS, "m")
SCHEMA = 1
# ASSUMPTION: un error que no es 404 (red, 5xx, zip roto) se reintenta al menos 5
# min aunque el rotulo sea viejo; un 404 de un rotulo viejo es un hueco de GDELT.
VENTANA_ERRORES = np.timedelta64(5, "m")
IDLE_MAX_S = 300.0  # el heartbeat se refresca al menos cada 5 min aun sin trabajo


@dataclass(frozen=True)
class NewsConfig:
    data_dir: Path
    theme: str = THEME_BITCOIN
    base_url: str = GKG_BASE_URL
    poll_seconds: float = 60.0
    lead_minutes: int = 15
    give_up_minutes: int = 120
    max_catchup_batches: int = 672  # 7 dias tras una caida larga
    timeout_s: float = 60.0
    heartbeat_max_age_minutes: int = 45

    def __post_init__(self) -> None:
        if not self.base_url.startswith("https://"):
            raise ConfigError("NEWS_BASE_URL debe ser https")
        if self.poll_seconds <= 0 or self.timeout_s <= 0:
            raise ConfigError("poll_seconds y timeout_s deben ser positivos")
        if self.lead_minutes < 0 or self.give_up_minutes <= 0:
            raise ConfigError("lead_minutes >= 0 y give_up_minutes > 0")
        if self.max_catchup_batches < 0:
            raise ConfigError("max_catchup_batches no puede ser negativo")
        if self.heartbeat_max_age_minutes <= LOTE_MINUTOS:
            raise ConfigError(
                "heartbeat_max_age_minutes debe superar el paso de 15 min: si no, la "
                "alerta salta entre dos lotes sanos"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> NewsConfig:
        """Lee ``NEWS_*``. Solo ``NEWS_DATA_DIR`` es obligatoria."""

        def leer(nombre: str) -> str | None:
            valor = environ.get(PREFIJO + nombre, "").strip()
            return valor or None

        directorio = leer("DATA_DIR")
        if directorio is None:
            raise ConfigError("falta NEWS_DATA_DIR: el grabador persiste a disco")
        opciones: dict[str, Any] = {"data_dir": Path(directorio)}
        try:
            for nombre, clave, tipo in (
                ("THEME", "theme", str),
                ("BASE_URL", "base_url", str),
                ("POLL_SECONDS", "poll_seconds", float),
                ("TIMEOUT_S", "timeout_s", float),
                ("LEAD_MINUTES", "lead_minutes", int),
                ("GIVE_UP_MINUTES", "give_up_minutes", int),
                ("MAX_CATCHUP_BATCHES", "max_catchup_batches", int),
                ("HEARTBEAT_MAX_AGE_MINUTES", "heartbeat_max_age_minutes", int),
            ):
                if (valor := leer(nombre)) is not None:
                    opciones[clave] = tipo(valor)
        except ValueError as exc:
            raise ConfigError(f"valor numerico invalido en el entorno: {exc}") from exc
        return cls(**opciones)

    def describe(self) -> dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "theme": self.theme,
            "base_url": self.base_url,
            "poll_seconds": self.poll_seconds,
            "lead_minutes": self.lead_minutes,
            "give_up_minutes": self.give_up_minutes,
            "max_catchup_batches": self.max_catchup_batches,
        }


@dataclass(frozen=True)
class Fetched:
    status: int  # codigo HTTP; el cuerpo, si es 200, quedo en ``dest``
    last_modified: str | None = None


Fetcher = Callable[[str, Path], Fetched]


def fetch_to_file(
    url: str, dest: Path, timeout_s: float
) -> Fetched:  # pragma: no cover
    """Descarga en streaming a disco: el lote nunca se carga entero en memoria."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r, open(dest, "wb") as f:
            while bloque := r.read(1 << 20):
                f.write(bloque)
            return Fetched(r.status, r.headers.get("Last-Modified"))
    except urllib.error.HTTPError as exc:
        return Fetched(exc.code)


class NewsStore(SnapshotStore):
    """Un archivo por dia UTC del rotulo; la cabecera lleva esquema y config."""

    def append_batch(self, label: np.datetime64, registro: Mapping[str, Any]) -> None:
        ruta = self._dir / f"news-{dia(label)}.jsonl"
        cabecera = _linea(self._header) if not ruta.exists() else b""
        self._append(ruta, _linea(registro), cabecera=cabecera)

    def labels(self, desde: np.datetime64) -> set[np.datetime64]:
        """Rotulos ya registrados (``ok`` o ``missing``) desde el dia de ``desde``."""
        primer_dia = f"news-{dia(desde)}.jsonl"
        salida: set[np.datetime64] = set()
        for ruta in sorted(self._dir.glob("news-*.jsonl")):
            if ruta.name < primer_dia:
                continue
            for linea in ruta.read_text().splitlines():
                try:
                    dato = json.loads(linea)
                except ValueError:
                    continue  # ultima linea rota por una caida
                if isinstance(dato, dict) and dato.get("type") == "batch":
                    salida.add(parse_ts(str(dato["label"])))
        return salida


def _md5(ruta: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(ruta, "rb") as f:
        while bloque := f.read(1 << 20):
            h.update(bloque)
    return h.hexdigest()


def _leer_lote(ruta: Path, theme: str) -> GkgBatch:
    with zipfile.ZipFile(ruta) as z:
        miembros = [n for n in z.namelist() if n.endswith(".gkg.csv")]
        if len(miembros) != 1:
            raise zipfile.BadZipFile(f"se esperaba un .gkg.csv, hay {z.namelist()}")
        with z.open(miembros[0]) as crudo:
            return parse_gkg(
                io.TextIOWrapper(crudo, encoding="utf-8", errors="replace"), theme
            )


def _clave(rotulo: np.datetime64) -> int:
    return int(rotulo.astype("int64"))


@dataclass
class _Estado:
    """Seguimiento de un rotulo en sondeo."""

    primero: np.datetime64
    ultimo_404: np.datetime64 | None = None
    intentos: int = 0
    error: Mapping[str, Any] = field(default_factory=dict)


class NewsRecorder:
    def __init__(
        self,
        config: NewsConfig,
        clock: Clock,
        store: NewsStore,
        fetch: Fetcher,
        *,
        sleep: Callable[[float], object],
        on_progress: Callable[[], object] = lambda: None,
    ) -> None:
        self._cfg = config
        self._clock = clock
        self._store = store
        self._fetch = fetch
        self._sleep = sleep
        self._on_progress = on_progress
        self._tmp = config.data_dir / "descarga.tmp"
        self._missing_streak = 0
        self._last: Mapping[str, Any] = {}
        self._grabados: set[np.datetime64] = set()

    def _intento(self, label: np.datetime64) -> tuple[str, dict[str, Any]]:
        """``("ok", campos)``, ``("404", {})`` o ``("error", {"last_error": ...})``."""
        url = gkg_url(label, self._cfg.base_url)
        try:
            r = self._fetch(url, self._tmp)
            if r.status == 404:
                return "404", {}
            if r.status != 200:
                return "error", {"last_error": f"http {r.status}"}
            recibido = self._clock.now()
            lote = _leer_lote(self._tmp, self._cfg.theme)
            return "ok", {
                "url": url,
                "bytes": self._tmp.stat().st_size,
                "md5": _md5(self._tmp),
                "last_modified": r.last_modified,
                "received_at": iso(recibido),
                "rows_total": lote.rows_total,
                "rows_malformed": lote.rows_malformed,
                "rows_theme": len(lote.articles),
                "articles": [
                    {
                        "id": a.record_id,
                        "domain": a.domain,
                        "url": a.url,
                        "tone": list(a.tone),
                        "title": a.title,
                    }
                    for a in lote.articles
                ],
            }
        except (OSError, zipfile.BadZipFile, EOFError) as exc:  # red, disco, zip roto
            return "error", {"last_error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._tmp.unlink(missing_ok=True)

    def _paso(self, label: np.datetime64, est: _Estado) -> Mapping[str, Any] | None:
        """Un intento. Devuelve el registro grabado si el lote quedo resuelto."""
        antes = self._clock.now()
        est.intentos += 1
        resultado, campos = self._intento(label)
        self._on_progress()
        if resultado == "ok":
            return self._cerrar(label, est, "ok", campos)
        if resultado == "404":
            est.ultimo_404 = antes  # cota inferior valida: no existia al pedirlo
        else:
            est.error = campos
        limite = label + np.timedelta64(self._cfg.give_up_minutes, "m")
        if resultado == "error":
            limite = np.maximum(limite, est.primero + VENTANA_ERRORES)
        if self._clock.now() >= limite:
            return self._cerrar(label, est, "missing", est.error)
        return None

    def collect(self, label: np.datetime64) -> Mapping[str, Any]:
        """Sondea UN rotulo hasta resolverlo, bloqueando. ``run`` sondea varios."""
        est = _Estado(self._clock.now())
        while (registro := self._paso(label, est)) is None:
            self._sleep(self._cfg.poll_seconds)
        return registro

    def _cerrar(
        self, label: np.datetime64, est: _Estado, estado: str, extra: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        registro = {
            "type": "batch",
            "schema": SCHEMA,
            "label": iso(label),
            "status": estado,
            "first_attempt_at": iso(est.primero),
            "last_404_at": iso(est.ultimo_404) if est.ultimo_404 is not None else None,
            "attempts": est.intentos,
            **extra,
        }
        self._store.append_batch(label, registro)
        self._grabados.add(label)
        self._missing_streak = self._missing_streak + 1 if estado == "missing" else 0
        self._last = registro
        self._heartbeat()
        log.log(
            logging.INFO if estado == "ok" else logging.WARNING,
            "lote",
            extra={
                "label": registro["label"],
                "status": estado,
                "attempts": est.intentos,
                "rows_theme": registro.get("rows_theme"),
            },
        )
        return registro

    def _heartbeat(self) -> None:
        self._on_progress()
        self._store.write_heartbeat(
            {
                "at": iso(self._clock.now()),
                "last_label": self._last.get("label"),
                "last_status": self._last.get("status"),
                "consecutive_missing": self._missing_streak,
            }
        )

    def first_label(self) -> np.datetime64:
        """Primer rotulo por resolver: el hueco mas viejo sin registrar, o el siguiente
        al ultimo. Con tope de recuperacion tras una caida larga."""
        actual = floor_label(self._clock.now())
        piso = actual - self._cfg.max_catchup_batches * PASO
        self._grabados = self._store.labels(piso)
        if not self._grabados:
            return actual
        ultimo = max(self._grabados, key=_clave)
        label = np.maximum(min(self._grabados, key=_clave), piso)
        while label <= ultimo:
            if label not in self._grabados:
                return np.datetime64(label, "ns")
            label = label + PASO
        return np.datetime64(np.maximum(ultimo + PASO, piso), "ns")

    def _esperar(self, hasta: np.datetime64, stop: Callable[[], bool]) -> None:
        while not stop() and (resto := hasta - self._clock.now()) > np.timedelta64(
            0, "s"
        ):
            self._sleep(min(float(resto / np.timedelta64(1, "s")), IDLE_MAX_S))
            self._heartbeat()

    def run(self, stop: Callable[[], bool]) -> None:
        """Sondea en paralelo todos los rotulos pendientes.

        Un lote que GDELT publica tarde (medido: ~45 min el 2026-10-20) no retiene a
        los siguientes: cada uno conserva su propio ``last_404_at``.
        """
        self._heartbeat()  # el healthcheck ve al proceso vivo desde el arranque
        siguiente = self.first_label()
        lead = np.timedelta64(self._cfg.lead_minutes, "m")
        pendientes: dict[np.datetime64, _Estado] = {}
        while not stop():
            ahora = self._clock.now()
            while siguiente - lead <= ahora:
                if siguiente not in self._grabados:
                    pendientes[siguiente] = _Estado(ahora)
                siguiente = siguiente + PASO
            if not pendientes:
                self._esperar(siguiente - lead, stop)
                continue
            for label in sorted(pendientes, key=_clave):
                if self._paso(label, pendientes[label]) is not None:
                    del pendientes[label]
            self._sleep(self._cfg.poll_seconds)


def evaluate(
    heartbeat: Mapping[str, Any] | None, ahora: np.datetime64, maximo_min: int
) -> Health:
    if heartbeat is None:
        return Health(False, ("heartbeat.json falta o es ilegible",))
    try:
        edad = np.datetime64(ahora, "ns") - parse_ts(str(heartbeat["at"]))
    except (KeyError, ValueError):
        return Health(False, ("heartbeat.json sin campo 'at' valido",))
    if edad > np.timedelta64(maximo_min, "m"):
        return Health(False, (f"heartbeat atrasado mas de {maximo_min} min",))
    return Health(True)


def build_recorder(
    config: NewsConfig,
    clock: Clock,
    *,
    sleep: Callable[[float], object],
    fetch: Fetcher | None = None,
    git_sha: str = "unknown",
    on_progress: Callable[[], object] = lambda: None,
) -> NewsRecorder:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    cabecera = {
        "type": "header",
        "schema": SCHEMA,
        "source": "gdelt-gkg-2.0",
        "git_sha": git_sha,
        "config": config.describe(),
    }
    descarga = fetch or (lambda url, dest: fetch_to_file(url, dest, config.timeout_s))
    return NewsRecorder(
        config,
        clock,
        NewsStore(config.data_dir, cabecera),
        descarga,
        sleep=sleep,
        on_progress=on_progress,
    )


def main(environ: Mapping[str, str], argv: list[str]) -> int:  # pragma: no cover
    if "--health" in argv:  # HEALTHCHECK de Docker: solo el heartbeat
        config = NewsConfig.from_env(environ)
        salud = evaluate(
            read_heartbeat(config.data_dir),
            SystemClock().now(),
            config.heartbeat_max_age_minutes,
        )
        for motivo in salud.reasons:
            print(motivo, file=sys.stderr)
        return 0 if salud.ok else 1
    configure_json_logging(sys.stdout)
    config = NewsConfig.from_env(environ)
    parar = threading.Event()
    for senal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(senal, lambda *_: parar.set())
    reloj = SystemClock()
    guarda = StallGuard(
        reloj, np.timedelta64(config.heartbeat_max_age_minutes, "m"), exit_process
    )
    guarda.start(parar)
    try:
        recorder = build_recorder(
            config,
            reloj,
            sleep=parar.wait,
            git_sha=environ.get("NEWS_GIT_SHA", "unknown"),
            on_progress=guarda.touch,
        )
    except StoreError as exc:
        print(f"no se puede iniciar: {exc}", file=sys.stderr)
        return 1
    recorder.run(parar.is_set)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import os

    raise SystemExit(main(os.environ, sys.argv[1:]))
