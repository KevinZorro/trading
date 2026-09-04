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

## Estado

Etapa 1 (simulador, baselines y métricas) cerrada. Ver la sección "Etapas" de
`CLAUDE.md`.
