# trading

Estudio comparativo de agentes de aprendizaje por refuerzo aplicados a trading: dos
agentes idénticos salvo por su espacio de estado, uno con señales de noticias y otro sin
ellas, evaluados en tres regímenes de riesgo y con capital variable.

Las reglas del proyecto (principios no negociables, invariantes consolidados y flujo de
trabajo) están en [`CLAUDE.md`](CLAUDE.md). Manda ese archivo.

## Puesta en marcha

```bash
uv sync                     # dependencias pinneadas desde uv.lock
uv run pytest               # suite completa
uv run pre-commit install   # hooks locales (ruff, mypy, anti-leakage)
```

## Las cuatro puertas

Las mismas en local y en CI. Si una falla localmente, va a fallar en el PR.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src                                    # modo strict
uv run pytest -m "not slow" --cov --cov-fail-under=85
```

## Anti-leakage

Los tests que verifican que no hay lookahead bias son la garantía central del estudio y
corren en un job propio del CI:

```bash
uv run pytest -m leakage -v
```

La suite entera corre **sin acceso a red**: `tests/conftest.py` bloquea los sockets y
falla con `NetworkAccessError`. Los datos se cargan de archivos locales o se generan con
`data.synthetic` a partir de una semilla fija.

## El agente PPO y su grupo opcional

`torch` y `stable-baselines3` viven en el grupo `rl`, que **`uv sync --frozen` no
instala**: la rueda de torch de PyPI arrastra el stack de CUDA (más de 3 GB) y sacaría
al job principal de CI de su presupuesto de 5 minutos en cada PR. El núcleo de
`agents/` —el protocolo de validación, el reporte por semillas, el walk-forward— no
depende de torch y se testea con políticas deterministas inyectadas.

```bash
uv sync --frozen --group rl          # torch + stable-baselines3 + sb3-contrib
```

### Qué cubre CI y qué no

La costura entre el entorno y SB3 **sí** está cubierta en cada PR, sin torch:
`tests/agents/test_ppo_contract.py` prueba `SB3Policy` contra un doble que implementa
la misma firma de `predict`. Verifica que la observación llegue al modelo tal como la
produjo el entorno, que la evaluación sea siempre determinista, que el adaptador no
altere la acción —el recorte lo hace el entorno y queda registrado— y que el reward
que el agente maximiza sea el retorno neto del equity del motor.

El riesgo de un doble es que se aleje del original, así que hay dos tests marcados
`rl` que comparan su firma contra la de SB3 y verifican que un modelo real devuelva
lo que el doble promete. Esos, y el smoke test que entrena de verdad, **no corren en
CI**: se saltean donde el grupo no está instalado.

Localmente:

```bash
uv sync --frozen --group rl
uv run pytest -m rl -v                       # contrato contra SB3 real + smoke de PPO
uv run pytest tests/agents/test_ppo_smoke.py -v
```

No se agregó un job de CI que instale torch de CPU porque el índice
`download.pytorch.org` no era alcanzable desde el entorno donde se desarrolló esto, y
un job que no se pudo verificar es peor que ninguno. Queda declarado en
`docs/adr/0003` en vez de disimulado.

### Correr el protocolo de validación

```bash
python -m agents.cli arm --level level_0 --bars 4000 --out results
python -m agents.cli assemble --out results          # aplica los criterios
```

`assemble` devuelve código distinto de cero si el protocolo se detuvo en algún nivel.

## Estado

Etapas 1 (simulador, baselines y métricas) y 2 (entorno Gymnasium) cerradas.
Ver la sección "Etapas" de `CLAUDE.md`.
