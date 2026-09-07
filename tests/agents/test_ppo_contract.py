"""Test de contrato del adaptador de SB3. **Sin torch.**

El grupo `rl` no se instala en CI -la rueda de torch de PyPI arrastra el stack de
CUDA-, asi que ``tests/agents/test_ppo_smoke.py`` se saltea ahi. Eso dejaba la
costura entre el entorno y ``stable-baselines3`` sin cubrir en el unico lugar
donde la cobertura importa: el que corre en cada PR.

Este archivo la cubre. ``SB3Policy`` no necesita torch para nada: solo llama a
``model.predict(...)``. Se le pasa un modelo falso que **implementa la misma
firma** que el de SB3 y se verifica el contrato de los dos lados:

- **Hacia el modelo**: que la observacion llegue tal como la produjo el entorno,
  con la forma correcta y dentro del espacio declarado, y que la evaluacion sea
  siempre determinista.
- **Hacia el entorno**: que la accion del modelo llegue sin que el adaptador la
  altere -incluso fuera de rango, para que el recorte quede registrado donde
  corresponde- y que el reward que el agente maximiza sea el que produce el
  equity del motor.

El riesgo de un doble es que se aleje del original. Lo cubre
``test_la_firma_del_doble_coincide_con_la_de_sb3``, marcado ``rl``: compara la
firma del falso contra la real. Si SB3 cambia su API, ese test lo dice y estos
dejan de ser una ficcion consistente consigo misma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from agents.policy import Policy
from agents.ppo import PPOConfig, SB3Policy, action_saturation
from agents.runner import build_env, run_policy, scaler_for
from data.fixtures import level_1_noisy
from envs.rewards import NetReturnReward
from sim.engine import SimConfig

SEED = 20240115
CAPITAL = 100_000.0


@dataclass
class ModeloSB3Falso:
    """Doble de un modelo de SB3. Misma firma de ``predict``, sin torch.

    Devuelve acciones de un guion y registra cada llamada, que es lo que permite
    verificar el contrato **hacia el modelo**: con que observacion se lo llamo,
    con que estado, y si se pidio evaluacion determinista.
    """

    acciones: list[float]
    llamadas: list[dict[str, Any]] = field(default_factory=list)
    estado_devuelto: tuple[np.ndarray, ...] | None = None
    indice: int = 0

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.llamadas.append(
            {
                "observation": np.array(observation, copy=True),
                "state": state,
                "episode_start": (
                    None
                    if episode_start is None
                    else np.array(episode_start, copy=True)
                ),
                "deterministic": deterministic,
            }
        )
        accion = self.acciones[min(self.indice, len(self.acciones) - 1)]
        self.indice += 1
        return np.array([accion], dtype=np.float32), self.estado_devuelto


def entorno(serie_fixture: Any = None) -> Any:
    fixture = serie_fixture or level_1_noisy(400, seed=SEED)
    train, _, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)
    return build_env(
        fixture.series,
        config,
        scaler=escalador,
        reward=NetReturnReward(scale=100.0),
    )


# ---------------------------------------------------------------------------
# Contrato hacia el modelo: que recibe
# ---------------------------------------------------------------------------


def test_la_observacion_llega_con_la_forma_que_declara_el_entorno() -> None:
    """El adaptador no reordena, no recorta y no cambia el tipo.

    Si lo hiciera, el agente entrenaria sobre un vector distinto del que el
    entorno declara en ``observation_space``, y nada lo detectaria: PPO acepta
    cualquier vector del tamano correcto.
    """
    env = entorno()
    modelo = ModeloSB3Falso(acciones=[0.5])
    salida = run_policy(env, SB3Policy(modelo, recurrent=False))

    assert modelo.llamadas, "el adaptador no llamo al modelo ni una vez"
    for llamada in modelo.llamadas:
        obs = llamada["observation"]
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float64
        assert env.observation_space.contains(obs)
    assert len(modelo.llamadas) == salida.steps


def test_la_observacion_es_exactamente_la_que_produjo_el_entorno() -> None:
    """Bit a bit, no "parecida".

    Se compara contra la observacion que el propio entorno devuelve al resetear:
    cualquier normalizacion extra dentro del adaptador seria una transformacion
    que el escalador serializado no conoce, y en vivo no se reproduciria.
    """
    env = entorno()
    esperada, _ = env.reset(seed=1)
    modelo = ModeloSB3Falso(acciones=[0.5])
    politica = SB3Policy(modelo, recurrent=False)
    politica.act(esperada)
    np.testing.assert_array_equal(modelo.llamadas[0]["observation"], esperada)


def test_la_evaluacion_siempre_pide_determinismo() -> None:
    """Evaluar muestreando reporta el promedio de las dudas del agente.

    Lo que se pone a operar es la politica, no su distribucion, y por eso el
    adaptador fija ``deterministic=True`` y no lo expone como opcion.
    """
    modelo = ModeloSB3Falso(acciones=[0.3])
    politica = SB3Policy(modelo, recurrent=False)
    politica.act(np.zeros(19))
    assert all(ll["deterministic"] is True for ll in modelo.llamadas)


# ---------------------------------------------------------------------------
# Contrato hacia el entorno: que devuelve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("devuelto", "esperado"),
    [
        (np.array([0.25], dtype=np.float32), 0.25),
        (np.array([[0.75]], dtype=np.float64), 0.75),
        (np.float32(0.5), 0.5),
    ],
)
def test_la_accion_se_convierte_a_escalar_sin_alterarla(
    devuelto: Any, esperado: float
) -> None:
    """SB3 devuelve arrays de formas distintas segun el envoltorio del entorno."""

    class Modelo:
        def predict(self, observation: Any, **kwargs: Any) -> tuple[Any, None]:
            return devuelto, None

    politica = SB3Policy(Modelo(), recurrent=False)
    assert politica.act(np.zeros(19)) == pytest.approx(esperado)


def test_el_adaptador_no_recorta_y_el_recorte_queda_registrado() -> None:
    """**El adaptador no censura al modelo.** El entorno recorta y lo registra.

    Es el invariante de la Etapa 2: un agente entrenado sobre un rango que
    alguien recorta en silencio aprende sobre un mundo que no existe. Si el
    adaptador recortara, ``ClipEvent`` quedaria vacio y el sintoma seria
    invisible.

    En produccion SB3 recorta contra el espacio antes de entregar la accion, asi
    que este caso no aparece; el test existe para fijar de quien es la
    responsabilidad, no para describir lo que suele pasar.
    """
    env = entorno()
    modelo = ModeloSB3Falso(acciones=[1.8])
    salida = run_policy(env, SB3Policy(modelo, recurrent=False))
    assert salida.clipped_actions == salida.steps
    assert env.clip_events[0].raw_action == pytest.approx(1.8)
    assert env.clip_events[0].clipped_action == pytest.approx(1.0)


def test_una_accion_no_finita_es_un_error_y_no_un_recorte() -> None:
    """Un NaN del modelo no se convierte en 0 ni en 1: revienta."""
    env = entorno()
    modelo = ModeloSB3Falso(acciones=[float("nan")])
    with pytest.raises(ValueError, match="accion no finita"):
        run_policy(env, SB3Policy(modelo, recurrent=False))


def test_una_politica_saturada_se_mide_como_saturacion() -> None:
    """El optimo de estos fixtures es bang-bang: saturar es lo esperado."""
    env = entorno()
    modelo = ModeloSB3Falso(acciones=[1.0])
    salida = run_policy(env, SB3Policy(modelo, recurrent=False))
    assert action_saturation(salida.actions) == pytest.approx(1.0)
    assert salida.clipped_actions == 0


# ---------------------------------------------------------------------------
# Propagacion del reward
# ---------------------------------------------------------------------------


def test_el_reward_es_el_retorno_neto_del_equity_del_motor() -> None:
    """El reward que el agente maximiza sale del equity, no de un calculo aparte.

    Es el principio 3 del proyecto -los costos van dentro de la recompensa- y
    aca se verifica de la unica forma que vale: reconstruyendo la suma de
    rewards desde la serie de equity del ``SimResult``.
    """
    env = entorno()
    modelo = ModeloSB3Falso(acciones=[0.6])
    salida = run_policy(env, SB3Policy(modelo, recurrent=False))

    equity = np.asarray(salida.result.equity, dtype=np.float64)
    # El episodio decide desde `warmup` y el ultimo paso expira con reward 0.
    tramo = equity[env.warmup : env.warmup + salida.steps]
    retornos = np.diff(tramo) / tramo[:-1]
    assert float(np.sum(retornos)) * 100.0 == pytest.approx(
        salida.total_reward, rel=1e-9
    )


def test_la_escala_del_reward_es_afin_y_no_cambia_la_politica() -> None:
    """``reward_scale`` multiplica gradientes, no cambia el problema.

    Si cambiara el equity o las acciones seria un parametro del mercado
    disfrazado de hiperparametro de optimizacion.
    """
    fixture = level_1_noisy(400, seed=SEED)
    train, _, _ = fixture.split()
    escalador = scaler_for(train.series)
    config = SimConfig(initial_cash=CAPITAL, max_participation=1.0)

    resultados = []
    for escala in (1.0, 100.0):
        env = build_env(
            fixture.series,
            config,
            scaler=escalador,
            reward=NetReturnReward(scale=escala),
        )
        resultados.append(
            run_policy(env, SB3Policy(ModeloSB3Falso([0.6]), recurrent=False))
        )

    uno, cien = resultados
    assert uno.actions == cien.actions
    np.testing.assert_array_equal(uno.result.equity, cien.result.equity)
    assert cien.total_reward == pytest.approx(uno.total_reward * 100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Politica recurrente: el estado de la LSTM
# ---------------------------------------------------------------------------


def test_la_politica_recurrente_hila_el_estado_entre_pasos() -> None:
    """Primer paso: ``state=None`` y ``episode_start=True``. Despues, al reves.

    Sin eso la LSTM arranca cada barra sin memoria -y entonces no es recurrente-
    o arrastra la del episodio anterior, y el resultado depende del orden en que
    se evaluaron los episodios.
    """
    estado = (np.zeros((1, 1, 4), dtype=np.float32),)
    modelo = ModeloSB3Falso(acciones=[0.5], estado_devuelto=estado)
    politica = SB3Policy(modelo, recurrent=True)

    politica.act(np.zeros(19))
    politica.act(np.zeros(19))
    politica.act(np.zeros(19))

    primera, segunda, tercera = modelo.llamadas
    assert primera["state"] is None
    assert bool(primera["episode_start"][0]) is True
    assert segunda["state"] is estado
    assert bool(segunda["episode_start"][0]) is False
    assert tercera["state"] is estado
    assert bool(tercera["episode_start"][0]) is False


def test_reset_borra_la_memoria_del_episodio_anterior() -> None:
    estado = (np.ones((1, 1, 4), dtype=np.float32),)
    modelo = ModeloSB3Falso(acciones=[0.5], estado_devuelto=estado)
    politica = SB3Policy(modelo, recurrent=True)
    politica.act(np.zeros(19))
    politica.act(np.zeros(19))
    politica.reset()
    politica.act(np.zeros(19))

    assert modelo.llamadas[-1]["state"] is None
    assert bool(modelo.llamadas[-1]["episode_start"][0]) is True


def test_la_politica_no_recurrente_no_manda_estado() -> None:
    """Mandar estado a una politica sin memoria seria un contrato distinto."""
    modelo = ModeloSB3Falso(acciones=[0.5])
    SB3Policy(modelo, recurrent=False).act(np.zeros(19))
    assert modelo.llamadas[0]["state"] is None
    assert modelo.llamadas[0]["episode_start"] is None


# ---------------------------------------------------------------------------
# Identidad del adaptador
# ---------------------------------------------------------------------------


def test_el_adaptador_satisface_el_protocolo_Policy() -> None:
    politica = SB3Policy(ModeloSB3Falso([0.5]), recurrent=False, label="prueba")
    assert isinstance(politica, Policy)
    assert politica.name == "prueba"
    assert politica.describe()["recurrent"] is False


def test_la_config_serializa_todo_lo_que_cambia_el_entrenamiento() -> None:
    """Los mismos hiperparametros con otra config dan otro numero."""
    descripcion = PPOConfig().describe()
    for clave in (
        "total_timesteps",
        "learning_rate",
        "gamma",
        "net_arch",
        "reward_scale",
        "recurrent",
        "policy_name",
    ):
        assert clave in descripcion


# ---------------------------------------------------------------------------
# El puente entre el doble y el original
# ---------------------------------------------------------------------------


@pytest.mark.rl
def test_la_firma_del_doble_coincide_con_la_de_sb3() -> None:
    """El riesgo de un doble es que se aleje del original.

    Si SB3 cambia la firma de ``predict``, los tests de arriba seguirian pasando
    contra una ficcion consistente consigo misma. Este test es el unico que
    necesita el grupo ``rl``, y es el que los ata a la realidad.
    """
    import inspect

    sb3 = pytest.importorskip("stable_baselines3")

    real = inspect.signature(sb3.PPO.predict)
    falso = inspect.signature(ModeloSB3Falso.predict)
    assert list(real.parameters) == list(falso.parameters)
    for nombre, parametro in real.parameters.items():
        if nombre == "self":
            continue
        assert falso.parameters[nombre].default == parametro.default, nombre


@pytest.mark.rl
def test_el_modelo_real_devuelve_lo_que_el_doble_promete() -> None:
    """Un modelo de SB3 sin entrenar ya cumple el contrato de ``predict``.

    Verifica la forma del valor de retorno -tupla ``(accion, estado)``, accion
    como array- contra el original, que es lo que el doble imita.
    """
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    env = entorno()
    modelo = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, device="cpu")
    observacion, _ = env.reset(seed=0)
    accion, estado = modelo.predict(observacion, deterministic=True)
    assert isinstance(accion, np.ndarray)
    assert estado is None
    assert env.action_space.contains(np.asarray(accion, dtype=np.float64))
