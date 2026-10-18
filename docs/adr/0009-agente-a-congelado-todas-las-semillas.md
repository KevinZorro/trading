# ADR 0009 — Agente A congelado: una política por semilla del estudio

Estado: propuesto en el PR 5b · Fecha: 2026-10-18 · Etapa 3.5

**Adenda al ADR 0007 §5.4. No lo edita**: el 0007 es un pre-registro y queda como
está. Donde este ADR y el 0007 §5.4 difieren, rige este ADR **solo** para la
selección de semillas y el corte de datos; todo lo demás del 0007 (métrica, cadencia,
predicción del spread, ledger de 100 USDT) sigue igual. Este ADR se escribe **antes**
de entrenar nada: no existe ningún peso congelado a la fecha.

## Contexto

El 0007 §5.4 fija una sola semilla, la 11 (`STUDY_SEEDS[0]`), con el argumento de que
entrenar varias y elegir una sería reportar el mejor seed. El argumento es correcto, pero
la elección de la 11 no es neutral si no se puede demostrar que se hizo sin mirar
resultados: `STUDY_SEEDS` ya se usó en C3 y en el protocolo, y esa historia no se
puede borrar. Kevin declaró (2026-10-18) que no está seguro de que la 11 se haya
elegido sin mirar resultados. Una semilla elegida con información posterior contamina
justamente lo que el 0007 quería evitar.

Alternativas consideradas:

| Opción | Efecto |
|---|---|
| Mantener la 11 | Respeta el pre-registro; arrastra una duda que no se puede cerrar. |
| Semilla derivada del hash del ADR 0007 | Neutral hacia adelante, pero arbitraria y todavía una sola política. |
| **Todas las semillas de `STUDY_SEEDS`** | No hay elección que contaminar; se obtiene una distribución. |

## Decisión

1. **Se congela un agente por cada semilla de `STUDY_SEEDS`** (las 10), sin excepción y
   sin selección posterior. Ninguna se descarta por su comportamiento (política
   degenerada que siempre vale 0, saturada en 1, etc.): esos casos se reportan como
   parte de la distribución. Solo una falla técnica del entrenamiento (excepción o
   pesos no finitos) permite reintentar, con **la misma semilla**, y queda escrita en el
   manifest.
2. **Cada agente opera su propio ledger de 100 USDT y su propio `FillStore`**
   (directorio por semilla). Comparten el libro y las barras de cada instante, no el
   estado de la cuenta: una orden de un agente no puede consumir liquidez ni cash del
   otro.
3. **Qué mide y qué no mide la distribución.** Diez semillas sobre un solo camino de
   mercado miden varianza de *entrenamiento*, no de *mercado* (CLAUDE.md, principio 5).
   La sombra es un único camino real, N = 1, y se declara así en cada reporte. Los
   fills de los diez agentes **no son diez observaciones independientes del costo**:
   miden la misma realidad con tamaños de orden distintos. El veredicto de costos del
   0007 §6 sale de los snapshots de `live.shadow`, no de estos agentes, y no cambia.
4. **Mismo código y mismos parámetros para las diez.** Lo que cambia entre agentes es
   `seed`, y nada más.

## Entrenamiento (fijado ahora, antes de correrlo)

- **Configuración de PPO**: la de C3, opción (i): `WalkForwardConfig().ppo_config()`
  (60 000 timesteps, MLP 64×64, `reward_scale = 100`). Declarado como en el 0007:
  política **subentrenada**; su función es producir decisiones reales para medir
  mecánica y costos, no rendimiento. Se reemplaza cuando se calibre el presupuesto
  sobre el fixture 4b, antes de la Etapa 5.
- **Datos**: las **1000 últimas barras** de `datasets/binance/BTCUSDT-1d.csv`
  (SHA256 `0940f089…9cc8`, 3328 filas, última barra con `timestamp`
  2026-09-27T00:00Z). El corte queda fijo por el dataset versionado, no por "el momento de
  congelarlo" del 0007: así el entrenamiento es reproducible desde un clon y no
  depende de red. Los pesos nunca vieron una barra posterior a esa fecha, y la sombra
  decide solo sobre barras posteriores.
- **Costos del entorno de entrenamiento**: escenario `principal` de
  `WalkForwardConfig().sim_config()` (Corwin-Schultz tope 200 bps, `SqrtSlippage`
  k = 0.1), con el `InstrumentSpec` del `exchangeInfo` grabado
  (`tests/live/respuestas/exchange_info.json`, `min_notional = 5`), según el ADR 0008.
- **Escalador**: se ajusta solo con esas 1000 barras (`fit_scaler_on_train`) y se
  serializa; no se reajusta nunca en vivo.
- **Sizer**: `TargetWeightSizer` con la regla del cero numérico (posterior a #20), igual
  que el entorno. No es comparable con C3.
- **Sin validación ni selección.** No hay ventana de validación sobre estas barras
  porque no hay nada que elegir: no se ajustan hiperparámetros ni se escoge entre
  semillas.

## Artefactos y trazabilidad

Cada agente deja `model.zip`, `scaler.json` y `config.json`, y un `manifest.json` común
con el SHA256 de cada archivo, el commit, el SHA256 del dataset, del
`exchangeInfo` y de este ADR, las versiones de `torch` y `stable-baselines3`, y las
semillas. `FrozenPolicy` verifica los SHA256 al cargar y falla si no coinciden. La
reproducibilidad es la de SB3 con semilla fija; no es bit a bit entre máquinas, y por eso
el SHA256 de los pesos se registra como lo que se desplegó, no como lo que cualquiera
reproducirá exactamente.

## Qué no concluye

- No concluye nada sobre rentabilidad: política subentrenada, un solo camino, 100 USDT.
- Diez agentes no son diez caminos. Una dispersión entre ellos es varianza de
  entrenamiento.
- No es comparable con C3 (otro sizer, otra ventana, otro `min_notional`).
