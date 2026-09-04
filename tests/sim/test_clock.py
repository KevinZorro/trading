"""El reloj inyectable y la garantia de que es el unico acceso al tiempo."""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from sim.clock import Clock, SimulatedClock, SystemClock

T0 = np.datetime64("2020-01-01T00:00:00", "ns")


class TestSimulatedClock:
    def test_arranca_donde_se_le_dice(self) -> None:
        assert SimulatedClock(T0).now() == T0

    def test_avanza_a_donde_se_le_dice(self) -> None:
        reloj = SimulatedClock(T0)
        destino = T0 + np.timedelta64(3, "D")
        reloj.advance_to(destino)
        assert reloj.now() == destino

    def test_quedarse_en_la_misma_barra_es_valido(self) -> None:
        reloj = SimulatedClock(T0)
        reloj.advance_to(T0)
        assert reloj.now() == T0

    def test_el_tiempo_no_retrocede_dentro_de_una_corrida(self) -> None:
        reloj = SimulatedClock(T0)
        reloj.advance_to(T0 + np.timedelta64(1, "D"))
        with pytest.raises(ValueError, match="no puede retroceder"):
            reloj.advance_to(T0)

    def test_reset_to_es_la_unica_forma_de_rebobinar(self) -> None:
        """Una instancia de Simulator se corre mas de una vez; la segunda
        corrida arranca en la primera barra."""
        reloj = SimulatedClock(T0)
        reloj.advance_to(T0 + np.timedelta64(5, "D"))
        reloj.reset_to(T0)
        assert reloj.now() == T0

    def test_normaliza_la_resolucion_a_nanosegundos(self) -> None:
        """Comparar contra un timestamp de BarSeries no debe requerir conversion."""
        reloj = SimulatedClock(np.datetime64("2020-01-01", "D"))
        assert reloj.now().dtype == np.dtype("<M8[ns]")

    def test_satisface_el_protocolo(self) -> None:
        assert isinstance(SimulatedClock(T0), Clock)


class TestSystemClock:
    def test_satisface_el_protocolo(self) -> None:
        assert isinstance(SystemClock(), Clock)

    def test_devuelve_nanosegundos_utc(self) -> None:
        ahora = SystemClock().now()
        assert ahora.dtype == np.dtype("<M8[ns]")
        # Cota grosera: el test no debe depender de la fecha exacta, solo de que
        # esto sea una hora de pared plausible y no un valor por defecto.
        assert ahora > np.datetime64("2020-01-01", "ns")


class TestNadieMasMiraElRelojDelSistema:
    """Hace mecanica la regla de CLAUDE.md, en vez de dejarla a la disciplina.

    Si alguien agrega un ``datetime.now()`` en cualquier otro modulo de ``src/``,
    este test falla. Es la unica forma de que la paridad backtest-live no se
    erosione de a un llamado por vez.
    """

    PATRON = re.compile(
        r"datetime\.now|datetime\.utcnow|time\.time\(|Timestamp\.now|date\.today"
    )
    PERMITIDO = "clock.py"

    def test_solo_sim_clock_consulta_el_reloj_de_pared(self) -> None:
        raiz = pathlib.Path(__file__).resolve().parents[2] / "src"
        assert raiz.is_dir(), f"no encontre src/ en {raiz}"

        infractores = [
            ruta.relative_to(raiz).as_posix()
            for ruta in sorted(raiz.rglob("*.py"))
            if ruta.name != self.PERMITIDO
            and self.PATRON.search(ruta.read_text(encoding="utf-8"))
        ]
        assert infractores == [], (
            f"estos modulos consultan el reloj del sistema: {infractores}. "
            "El tiempo entra por el Clock inyectable; ver src/sim/clock.py."
        )

    def test_el_test_detecta_de_verdad(self, tmp_path: pathlib.Path) -> None:
        """Verifica el patron contra un caso conocido: sin esto, el test de
        arriba podria estar en verde por buscar algo que nunca coincide."""
        assert self.PATRON.search("x = datetime.now(UTC)")
        assert self.PATRON.search("t = time.time()")
        assert not self.PATRON.search("clock.now()")
