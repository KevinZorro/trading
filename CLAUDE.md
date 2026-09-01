# CLAUDE.md — Proyecto: RL Trading con Señales de Noticias

## Qué es esto

Estudio comparativo de agentes de aprendizaje por refuerzo aplicados a trading.
Se comparan **dos agentes idénticos** salvo por su espacio de estado:

- **Agente A (baseline):** solo features de precio/volumen/técnicos.
- **Agente B (news-aware):** los mismos features + señales derivadas de noticias y macro.

Se evalúan en **3 regímenes de riesgo** (bajo / medio / alto) y con **capital variable**
para determinar a partir de qué monto los costos de transacción destruyen cualquier ventaja.

**Este es un proyecto de investigación, no un bot de producción.** El objetivo es un
resultado honesto sobre si las noticias aportan alfa, no maximizar una métrica de backtest.
Un resultado negativo bien medido es un resultado válido y esperado.

## Principios no negociables

Estos son los que más se violan por accidente. Trátalos como invariantes del proyecto.

1. **Sin lookahead bias.** Ningún dato disponible en el instante `t` puede depender de
   información publicada después de `t`. Aplica a precios, noticias, revisiones de datos
   fundamentales y a los modelos de NLP usados para extraer sentimiento.
2. **Point-in-time timestamps.** Las noticias usan hora de *publicación original*, nunca
   fecha de última edición. Si un dataset no distingue ambas, se documenta como limitación.
3. **Costos dentro de la recompensa.** Comisiones, spread y slippage se restan en la función
   de reward, no en un post-procesamiento. Restarlos después produce agentes que sobre-operan.
4. **La ejecución nunca ocurre en la misma barra que generó la señal.** Decisión con datos
   de cierre de `t`, ejecución en la apertura de `t+1` (o con latencia explícita en intradía).
5. **Nunca reportar el mejor seed.** Toda métrica se reporta como distribución sobre >= 10
   semillas: mediana, p25, p75, min, max. El mejor run de un agente RL en trading es ruido.
6. **Todo experimento incluye baselines.** buy-and-hold, agente aleatorio y cruce de medias.
   Si el agente no supera buy-and-hold con costos, el resultado es "no supera buy-and-hold".
7. **Reproducibilidad total.** Seeds fijos, versiones de dependencias pinneadas, config
   serializada junto a cada resultado.

## Anti-patrones prohibidos

- Normalizar features usando estadísticas calculadas sobre todo el dataset (incluye el test).
  La normalización se ajusta solo con datos de entrenamiento.
- Usar precios crudos como feature. No son estacionarios. Usar retornos logarítmicos.
- Train/test split simple. Se usa **walk-forward** con ventanas rodantes.
- Rellenar NaN con `ffill` a través de la frontera train/test.
- Ejecutar al precio medio ignorando el spread.
- Iterar hiperparámetros contra el conjunto de test. El test se toca una vez, al final.
- Universos de activos construidos con la lista actual de constituyentes (survivorship bias).

## Stack

- Python 3.11+
- `uv` para gestión de dependencias
- `gymnasium` para la interfaz de entorno
- `stable-baselines3` (PPO, SAC) para los agentes
- `polars` o `pandas` para datos
- `pytest` para tests
- `hydra` o dataclasses + YAML para configuración
- MLflow o wandb para tracking de experimentos

## Estructura del repositorio

```
src/
  data/          # ingesta, alineación temporal, validación point-in-time
  sim/           # motor de simulación: order book, costos, slippage, latencia
  envs/          # entornos Gymnasium que envuelven el simulador
  features/      # técnicos y de noticias; transformadores fit-en-train
  agents/        # wrappers de PPO/SAC + baselines heurísticos
  eval/          # métricas, walk-forward, tests estadísticos
  configs/       # YAML por experimento
tests/
notebooks/       # solo exploración; nada de lógica de producción aquí
```

## Etapas del proyecto

Se construye en orden. **No se avanza a la siguiente etapa sin cerrar la anterior.**

1. **Simulador + baselines.** Motor de ejecución con costos realistas. Baselines corriendo
   sobre él. Sin ML.
2. **Entorno Gymnasium** sobre el simulador, con tests que verifican ausencia de leakage.
3. **Agente A** (solo precio) en un activo, un régimen. Pregunta a responder:
   ¿supera buy-and-hold neto de costos?
4. **Pipeline de noticias** con validación point-in-time estricta.
5. **Agente B** y comparación controlada contra A.
6. **Barrido** de regímenes y capital.
7. **Paper trading** en vivo antes de cualquier consideración de capital real.

## Métricas de evaluación

Retorno total, CAGR, Sharpe, Sortino, max drawdown, Calmar, turnover, win rate,
profit factor, y **costos totales pagados** como métrica de primer nivel.

Para significancia estadística ante comparaciones múltiples, usar **Deflated Sharpe Ratio**
(Bailey & López de Prado), no p-values ingenuos.

## Estilo de trabajo esperado

- Escribe tests antes o junto con la lógica, especialmente en `sim/` y `data/`.
- Los tests de leakage son tan importantes como los tests funcionales.
- Prefiere código explícito y aburrido sobre abstracciones ingeniosas.
- Si una decisión de diseño implica un supuesto sobre el mercado, documéntalo en un
  comentario con la palabra `ASSUMPTION:` para poder auditarlos todos después.
- Si detectas que una instrucción mía introduce leakage o un backtest optimista, dilo
  antes de implementarla.
