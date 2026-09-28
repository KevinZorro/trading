# Tarea C3 — Walk-forward del Agente A sobre BTCUSDT diario

Corrida sellada del ADR 0006, resellado en `e040707` (SHA256 `eea45aa6…d72`), con
`run_commit` `a861623`. Son 110 pares (fold, semilla): 11 folds × 10 semillas. **El
test se tocó una sola vez**, sin reanudaciones. `SEAL.json` se commiteó antes de que
se conociera ningún resultado (`8d41a6d`). El artefacto completo (`records.jsonl.gz`)
está en el Release `c3-results`; ver `MANIFEST.json`.

**La política está subentrenada** (60 000 timesteps, opción i). Este resultado valida
el pipeline y no concluye sobre el mercado. N = 1 camino por construcción.

## Veredicto principal (sección 2): **NO SUPERA**

| Mediana sobre 10 semillas | Principal (Corwin-Schultz) | Solo comisión (sensibilidad) |
|---|---|---|
| `X` mark: exceso log sobre B&H | **−2.42** (p25 −2.68, p75 −2.21, min −2.97, max −1.93) | −1.69 (min −2.18, max −0.98) |
| `X` liquidation | −2.37 (min −2.93, max −1.89) | −1.68 |
| DSR, `n_trials` = 10 (decide) | **0.001** (máx 0.003); umbral 0.95 | 0.006 |
| DSR, `n_trials` = 1 (informativo) | 0.004 | 0.019 |
| Veredicto | **NO SUPERA** | NO SUPERA |

- Ninguna semilla supera a buy-and-hold en ninguna de las dos series.
- Sobre los tests concatenados (2020-05 a 2026-05), buy-and-hold crece **+2.08 log (×8)**.
  El agente queda en −0.34 en el escenario principal y en +0.41 con solo comisión.
- Exposición mediana del agente: **0.20**. Dispersión de la acción: 0.37.
- Contra buy-and-hold, en el escenario principal: aleatorio −1.11, cruce de medias −1.89,
  agente −2.42.

## Secundario: exposición igualada (sección 4.2)

| | Principal | Solo comisión |
|---|---|---|
| `X_exp` | −0.85 (IQR −1.05 a −0.70; las 10 semillas negativas) | **−0.11** (IQR −0.24 a +0.14; cruza el cero) |

La brecha contra buy-and-hold se explica por **poca exposición** en un mercado que
subió ×8 y por **costos**. Con solo comisión, el timing es prácticamente neutro.

## Sensibilidad de costos (sección 1.1)

Quitar spread y slippage mejora al agente en unos +0.7 log, y no cambia el veredicto. La
calibración de costos llega con el modo sombra de la Etapa 3.5.

## Predicción del ADR 0004 (sección 4.3): **NO CONCLUYENTE**

- `G = −0.020`, `F = 0.091`. Las ventanas tras un cambio de régimen rinden algo peor,
  en el sentido predicho, pero la diferencia es cinco veces menor que el ruido de
  reentrenar. CONSISTENTE era el máximo posible.
- Subsidiaria 3: **no sostenida**. Se cumple en 1 de 3 folds tras cambio (el 7).

## Por fold (medianas sobre semillas, escenario principal)

| Fold | Test | Clase | `E` mark | `T` | Exposición | B&H log |
|---|---|---|---|---|---|---|
| 0 | 2020-05..2020-11 | ambigua | −0.37 | +0.03 | 0.33 | +0.63 |
| 1 | 2020-11..2021-06 | tras cambio | −0.71 | −0.10 | 0.17 | +0.73 |
| 2 | 2021-06..2022-01 | sin cambio | −0.06 | +0.09 | 0.31 | +0.21 |
| 3 | 2022-01..2022-07 | ambigua | **+0.45** | −0.10 | 0.28 | −0.67 |
| 4 | 2022-07..2023-02 | tras cambio | −0.08 | −0.09 | 0.20 | −0.01 |
| 5 | 2023-02..2023-08 | ambigua | −0.15 | −0.06 | 0.11 | +0.11 |
| 6 | 2023-08..2024-03 | ambigua | −0.94 | −0.08 | 0.08 | +1.00 |
| 7 | 2024-03..2024-09 | tras cambio | −0.06 | −0.16 | 0.06 | −0.10 |
| 8 | 2024-09..2025-04 | ambigua | −0.25 | −0.03 | 0.07 | +0.24 |
| 9 | 2025-04..2025-11 | ambigua | −0.25 | −0.05 | 0.16 | +0.25 |
| 10 | 2025-11..2026-05 | sin cambio | −0.02 | −0.25 | 0.30 | −0.31 |

El único fold en que el agente supera a buy-and-hold es el 3, la caída de 2022. Su
timing ahí es negativo (`T` = −0.10): el resultado viene de la baja exposición, no de
haber anticipado nada.

## Outliers (sección 5)

**0** fills outlier, como anticipaba el ADR: en los tests del diario no hay barras
marcadas ni huecos de calendario.

## Órdenes y rechazos (sección 1.1)

| | Conteo |
|---|---|
| Órdenes enviadas por el agente | 16 601 |
| Rechazadas por `MIN_NOTIONAL` | **2 915** (17,6 % de las enviadas) |
| Rechazadas por otro motivo | 9 520 (57 %) |
| Baselines, rechazos por `MIN_NOTIONAL` | B&H 0 · aleatorio 475 · cruce de medias 0 |

**Defecto encontrado después de la corrida.** Los registros de C3 guardan el conteo
de "otro motivo", pero no el motivo. No se desglosó reentrenando porque eso sería
tocar el test dos veces. Un diagnóstico **sin tocar el test** apunta a la causa:

- Se entrenó un agente igual sobre el entrenamiento del fold 0 y se lo evaluó dentro
  de esa misma muestra.
- Los rechazos por otro motivo fueron todos `ZERO_AFTER_ROUNDING`.
- De 70, **61 eran residuo de punto flotante** del ledger (≤ 3,9e-18 BTC): tras
  cerrar una posición, el ledger queda con ~1e-18 en vez de 0, y con la acción en 0
  el sizer mandaba una orden por ese residuo en cada barra.
- Los otros **9 eran deltas reales menores a un lote** (1,4e-6 a 9,9e-6 BTC): esos sí
  son rechazos legítimos.
- Un primer reporte de este diagnóstico decía "cantidad exactamente 0". Era incorrecto:
  los valores se habían impreso redondeados a 6 decimales.

Eso no afecta al P&L. Sí infla el denominador de las órdenes enviadas y marca
`last_order_rejected` ("no pude") donde correspondía "no quise". Se corrige después
de C3 con la regla del cero numérico del sizer (PR #20), y eso altera la observación
del agente: **los resultados posteriores no son comparables con este**.

**Estimaciones declaradas, no mediciones.** La tasa de rechazo por `min_notional`
sobre órdenes reales (sin residuo) cae entre dos extremos:

| Supuesto | Órdenes reales | Tasa `MIN_NOTIONAL` |
|---|---|---|
| Ninguna de las 9 520 es residuo | 16 601 | 17,6 % (piso) |
| La proporción in-sample (61/70 residuo) vale para C3 | ≈ 8 305 | **≈ 35 %** |
| Las 9 520 son todas residuo | 7 081 | **≈ 41 %** (cota superior) |

El 35 % extrapola una sola muestra in-sample (un fold, una semilla) a 110 pares fuera
de muestra. **Es una cota a verificar, no una medición.** La primera corrida con el
conteo por motivo lo mide.
