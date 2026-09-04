"""Los guardas de la suite tambien se testean.

Un guard que dejo de funcionar en silencio es peor que no tenerlo: da la
sensacion de aislamiento sin darlo.
"""

from __future__ import annotations

import random
import socket

import numpy as np
import pytest

from .conftest import GLOBAL_SEED, NetworkAccessError


class TestSinRed:
    def test_connect_tcp_esta_bloqueado(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkAccessError, match="sin red"):
            sock.connect(("example.com", 80))

    def test_connect_ex_tcp_esta_bloqueado(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(NetworkAccessError, match="sin red"):
            sock.connect_ex(("example.com", 80))

    def test_create_connection_esta_bloqueado(self) -> None:
        with pytest.raises(NetworkAccessError, match="sin red"):
            socket.create_connection(("example.com", 80))

    def test_urllib_no_puede_salir(self) -> None:
        """La ruta que usaria de verdad un loader mal escrito."""
        urllib_request = pytest.importorskip("urllib.request")
        with pytest.raises(NetworkAccessError):
            urllib_request.urlopen("http://example.com", timeout=1)


class TestSemillas:
    """El oraculo es un RNG independiente sembrado con la misma semilla.

    No una constante copiada de una corrida: eso solo verificaria que el
    resultado no cambio, no que la siembra ocurrio.
    """

    def test_el_rng_de_stdlib_queda_sembrado(self) -> None:
        esperado = random.Random(GLOBAL_SEED).random()
        assert random.random() == esperado

    def test_el_rng_global_de_numpy_queda_sembrado(self) -> None:
        esperado = np.random.RandomState(GLOBAL_SEED).random()
        assert np.random.random() == esperado  # noqa: NPY002

    def test_la_siembra_se_repite_en_cada_test(self) -> None:
        """Mismo valor que el test anterior: la fixture resiembra entre tests."""
        esperado = random.Random(GLOBAL_SEED).random()
        assert random.random() == esperado
