# Resultados del protocolo de validación — Agente A

Estos JSON son el **resultado reportado** de la Etapa 3, no artefactos
regenerables. Se versionan a propósito, y el motivo es concreto: `torch` no
garantiza reproducibilidad bit a bit entre arquitecturas, así que volver a
correr el barrido en otra máquina no produce exactamente estos números. La
`.gitignore` excluye `outputs/` con el argumento de que "se regeneran desde
config + seed"; para un entrenamiento con torch ese argumento no se sostiene, y
por eso estos archivos sí entran al repositorio.

## Qué hay en cada archivo

- `multipath_<nivel>.json` — un nivel sobre **N caminos independientes × M
  semillas**, con las dos varianzas separadas: `between_path_std` (mercado) y
  `within_path_std` (entrenamiento). Es la forma que exige el principio 5 de
  `CLAUDE.md` sobre fixtures sintéticos. Lleva además el `t` del drift por camino.

  El nivel 4 tiene **dos**: `multipath_level_4.json` (4a, control negativo puro,
  μ=0.08) y `multipath_level_4b.json` (drift detectable, μ=0.42). Son el mismo
  proceso salvo por `mu`, y con la misma semilla comparten la realización del
  ruido exactamente, así que el contraste entre ambos es **pareado**. Leerlos
  por separado no alcanza: ver `docs/adr/0005`.
- `arm_<etiqueta>.json` — un brazo: un fixture concreto entrenado y evaluado
  sobre las 10 semillas del estudio. Lleva la configuración completa del fixture
  y del agente, las distribuciones de cada métrica, los tres baselines, el techo
  del fixture y **una entrada por semilla**. Nada está agregado de forma que se
  pierda el detalle: la mediana se puede recalcular desde `runs`.
- `protocol.json` — el veredicto: los criterios declarados (`thresholds`),
  el resultado de cada nivel y en cuál se detuvo, si se detuvo.

## Cómo se reproduce

```bash
uv sync --frozen --group rl          # torch y stable-baselines3 son opcionales

# Un brazo por comando; se pueden correr en paralelo.
python -m agents.cli arm --level level_0 --bars 4000 --out results
python -m agents.cli arm --level level_1 --beta -0.3 --out results
# ... el resto de los brazos

# El nivel 4 necesita caminos, no solo semillas: 10 x 10.
python -m agents.cli multipath --level level_4  --paths 10 --seeds 10 --out results
python -m agents.cli multipath --level level_4b --paths 10 --seeds 10 --out results

# Los criterios se aplican sobre lo guardado, sin reentrenar.
python -m agents.cli assemble --out results
```

`assemble` devuelve código de salida distinto de cero si el protocolo se detuvo
en algún nivel.

## Lo que estos números **no** dicen

Están medidos sobre fixtures sintéticos, que son infraestructura de validación y
no datos de investigación (ver `docs/adr/0002`). En los niveles 0 y 1 la señal
es una feature cruda de la observación: que el agente la aprenda valida el
pipeline, no su capacidad de descubrir señal en un mercado.

Los niveles **3, 4a y 4b** están reportados con N=10 caminos × M=10 semillas. Los
niveles **1 y 2 siguen con N=1 camino** y quedan como deuda documentada: sus
conclusiones son robustas a la ruta y el cómputo de re-correrlos rinde más en la
Etapa 6 (ver `docs/adr/0004`). El nivel 0 es determinista y su varianza de mercado
es exactamente cero, así que N=1 ahí es completo.

El paso 6 del protocolo —datos reales con walk-forward— no está corrido: no hay
ningún dataset real versionado en el repositorio.
