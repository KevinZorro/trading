# Reporte final del estudio: ¿las noticias aportan alfa a un agente RL en BTCUSDT?

Estado: **completo salvo la sección 8 (sombra), que se escribe después del
~2026-11-18**. Este documento reúne y ordena lo ya medido; no introduce cifras
nuevas. Cada número tiene su fuente en `results/` o en un ADR, citada al lado.

## 1. Respuesta corta

1. **¿Las noticias aportan alfa?** En este estudio, **no se pudo mostrar**. Veredicto
   pre-registrado (ADR 0012 §6, `results/estudio/REPORTE.md`): **NO APORTA (plano)**.
   El Agente B rinde en mediana algo más que A y que el placebo, pero la diferencia
   no se distingue del ruido (DSR 0,146 y 0,200 contra un umbral de 0,95).
2. **Por qué, según el re-diagnóstico (N8e):** B **no usa las noticias**. Fijarlas en
   su media cambia la acción 0,0106 en mediana (umbral del ADR: 0,02); los técnicos
   mueven la acción diez veces más (0,105). Por eso la lectura correcta no es "las
   noticias de GDELT no tienen alfa" sino "**este agente, con estas siete columnas,
   no aprendió a extraerlo**".
3. **¿Desde qué capital destruyen los costos la ventaja?** **No se pudo responder, y
   no por falta de corrida sino porque no hay ventaja que destruir** (sección 6). El
   barrido de capital (Etapa 6) no se corrió.
4. **Ninguno de los tres brazos supera a buy-and-hold**, y A tampoco. Esto es
   independiente de las noticias y es el resultado más sólido del estudio.

Un resultado negativo bien medido es el resultado que el diseño declaraba como el más
probable (ADR 0012 §0), y lo es.

## 2. Qué se comparó

Dos agentes PPO idénticos salvo la observación (`STUDY_PPO` y `STUDY_OBSERVATION` de
`agents.study`, configuración **R3 a 120 000 timesteps**: `lookback` 1, lr 1e-4, 5
épocas, red 32×32, `gamma` 0,9; ADR 0011). Tres brazos, 11 folds walk-forward de
1 000 barras de entrenamiento y 200 de test (2020-05 a 2026-05), 10 semillas, BTCUSDT
diario, 100 USDT, costos (Corwin-Schultz con tope de 200 bps, comisión 10 bps, impacto
raíz cuadrada) dentro del reward, ejecución al open de `t+1` (ADR 0006 y 0012 §1):

| Brazo | Observación |
|---|---|
| A | 15 columnas de precio, volumen y técnicos |
| B | A + 7 columnas de noticias de GDELT (atención, tono, FinBERT, indicadores de cobertura) |
| Placebo | A + las mismas 7 columnas de `t − 364` días (conserva el día de la semana) |

Noticias: GDELT GKG 2.0, tema `ECON_BITCOIN`, 2016-08 a 2026-09, 2,74 M artículos.
`pub_time` = rótulo del lote + **60 min** (el p99 medido en vivo fue 48,4 min).
FinBERT (`ProsusAI/finbert`, pinneado por SHA256) solo donde hay título (desde
~2019-09-23). Todo el diseño, el criterio y el sello se fijaron **antes** de entrenar
(ADR 0012; `SEAL.json`). El test se tocó **una vez**: 330 entrenamientos
(3 × 11 × 10).

## 3. Cómo se llegó hasta aquí (la cadena de validación)

| Paso | Resultado | Fuente |
|---|---|---|
| Protocolo del Agente A sobre fixtures con óptimo conocido | Niveles 0 a 2 PASS; nivel 3 MEASURED (memoriza en 10 de 10 caminos); 4a PASS; 4b y el par fallaron con 60 000 timesteps | ADR 0003 a 0005 |
| C3: A sobre BTCUSDT con 60 000 timesteps | NO SUPERA a buy-and-hold (`X` mediana −2,42; DSR 0,001); política subentrenada | `results/c3/REPORTE.md` |
| Calibración del presupuesto sobre 4b | G1 a G4 (120 k a 960 k timesteps) fallan: el presupuesto no era la palanca | ADR 0010, `results/calibracion/` |
| Escalera de regularización | R1 y R2 fallan, **R3 pasa 4b** | ADR 0011, `results/regularizacion/` |
| Confirmación de R3 con el protocolo completo | Todo en verde: 4b con dispersión 0,141 contra 0,15 (poco margen), par con `t` = 18,9 | `results/confirmacion/` |
| Puerta de B sobre fixtures (N6) | 5a PASS (capture 0,925; `t` pareado 14,84 contra 2,262): el pipeline sí hace llegar una señal exógena. 5b MEASURED: con ruido, sin ventaja (`t` −0,77) | `results/puerta/REPORTE.md` |

La puerta importa para la lectura: **cuando una señal de noticias existe y es
fuerte, B la aprende** (5a). Que en datos reales no lo haga es información sobre los
datos y sobre el agente, no sobre una tubería rota.

## 4. Resultado principal (N9)

Mediana sobre 10 semillas (`results/estudio/REPORTE.md`):

| Condición K3 | Valor | Cumple |
|---|---|---|
| `D_A` = X(B) − X(A) > 0 (mark / liquidación) | +0,185 / +0,197 (7 de 10 semillas > 0) | sí |
| `D_P` = X(B) − X(placebo) > 0 (mark / liquidación) | +0,326 / +0,329 (7 de 10) | sí |
| DSR(B − A), `n_trials` = 10, ≥ 0,95 | **0,146** (máx 0,585) | no |
| DSR(B − placebo), `n_trials` = 10, ≥ 0,95 | **0,200** (máx 0,904) | no |

Ni con `n_trials` = 1 (informativo) cruza 0,95 la mediana (0,573 y 0,804).
`D_A` varía de −0,55 a +0,67 entre semillas; tres de diez dan negativo.

Contra buy-and-hold (`X`, exceso en log, mediana):

| Brazo | `X` mark | `X` exposición igualada |
|---|---|---|
| A | −1,53 | −0,26 |
| B | −1,37 | **+0,22** (8 de 10 semillas > 0) |
| Placebo | −1,76 | −0,05 |

Los tres pierden por mucho porque el mercado subió fuerte en el período y los agentes
están poco invertidos (exposición mediana de A en C3: 0,20). Con exposición igualada,
B es el único con mediana positiva. **Es el indicio más favorable a las noticias y es
secundario**: no estaba en el criterio de decisión y no se corrigió por pruebas
múltiples. No debe leerse como hallazgo.

Veredicto secundario de A (K1): **NO SUPERA** (DSR mediana 0,017). El número honesto de
configuraciones de A evaluadas sobre el test real es 2 (C3 y R3).

## 5. Por qué B no mejora: el re-diagnóstico (N8e)

Sobre las 22 políticas de B guardadas (`results/estudio/REDIAGNOSTICO.md`):

| Grupo de entradas fijado en su media | Cambio de la acción (mediana) |
|---|---|
| técnicos | 0,105 |
| posición | 0,031 |
| **noticias** | **0,011** (p75 0,017; máx 0,049) |
| retornos | 0,004 |
| resultado de órdenes | 0,003 |

- 5 de 22 políticas son **constantes** (ignoran todo). Sin ellas la mediana es 0,0125.
- Los folds 0 a 3 reaccionan más (hasta 0,048) y 0, 1 y 3 superan 0,02: "indicio
  débil, no concluyente" frente al piso de ruido de la puerta 5b (0,0252). No se probó
  por qué; una hipótesis sin verificar es peor cobertura de noticias en esos años.
- Reemplazar las noticias por las del placebo cambia la acción 0,017 en mediana, con
  cola larga (máx 0,181): unas pocas políticas reaccionan a la entrada, sin que eso
  demuestre utilidad.

Por tanto: **el "casi sin efecto" de la sección 4 es coherente con un agente que
ignora la entrada**. Las diferencias entre brazos (concentradas en el fold 1 y en los
folds de mucho movimiento) son compatibles con ruido de entrenamiento más pequeñas
diferencias de nivel de exposición.

## 6. La pregunta del capital umbral

El objetivo original era hallar el monto a partir del cual los costos destruyen
cualquier ventaja. Con este resultado:

- **No hay ventaja medida que los costos puedan destruir.** B no supera a A ni al
  placebo de forma distinguible, y los tres pierden contra buy-and-hold. La pregunta
  "¿desde qué capital?" presupone una ventaja a costo cero, y no se observó una.
- **No se corrió el barrido.** Todo el estudio es a **100 USDT** y un solo régimen
  (el barrido de capital y de regímenes de riesgo es la Etapa 6, sin empezar). Afirmar
  un umbral numérico sería inventarlo. Lo único medible aquí es la dirección:
  a mayor capital, el impacto de mercado (raíz cuadrada) crece, así que el umbral
  sólo puede ser menor o igual que cualquier ventaja bruta que existiera.
- **A 100 USDT el piso de la plataforma ya interfiere.** De 6 263 órdenes de A en N9,
  2 670 (42,6 %) se rechazaron por `MIN_NOTIONAL` y 2 190 por redondeo a cero (en C3:
  17,6 % por `MIN_NOTIONAL`). A capital más bajo, el piso de la plataforma ya limita
  lo que el agente puede hacer, antes de hablar de spread.
- **La brecha contra buy-and-hold no es sobre todo de costos.** En C3 la comisión
  sola explica parte, y la sensibilidad "solo comisión" con exposición igualada daba
  −0,11 (cruza el cero): con costos mínimos el timing era neutro. Lo que separa a los
  agentes de buy-and-hold es **estar poco invertidos** en un mercado que subió ×8.
- **Recomendación sobre la Etapa 6:** no correrla para este agente. Rinde sólo si
  alguna configuración supera a buy-and-hold primero. El único candidato débil sería B
  con exposición igualada (+0,22), y eso pide primero un estudio nuevo (sección 9).

## 7. Limitaciones (todas aplican)

1. **N = 1 camino real.** La historia es un solo camino. Diez semillas miden varianza
   de entrenamiento, no de mercado (CLAUDE.md, principio 5). El máximo veredicto posible
   era CONSISTENTE y no se alcanzó. Con datos reales, esto sólo se acota con más
   ventanas, no se mide.
2. **Barras diarias.** Se midió si el estado informativo del día ayuda a decidir la
   exposición del día siguiente. **No** se midió operar noticias intradía, que en cripto
   se descuentan en minutos u horas. Un NO APORTA aquí no entierra esa pregunta.
3. **Fuente de prensa general.** GDELT mide medios generales, no conversación cripto.
   "La fuente era mala" es una lectura admisible.
4. **El agente es chico y la política es casi aleatoria en entrenamiento** (σ ≈ 0,93;
   se evalúa de forma determinista). La red 32×32 con 22 entradas pudo quedar corta
   para B; es un sesgo en contra de B declarado en el ADR 0012 §11. B ve 7 columnas más
   con menos datos de entrenamiento (en los folds 0 a 3 hay menos barras con título).
5. **Arquitectura: el LSTM no se probó en el estudio.** En el nivel 3 de la
   calibración el LSTM rindió algo mejor que el MLP (`capture` −1,304 contra −1,350,
   diferencia 0,045), sin bloquear. Las dos memorizan en 10 de 10 caminos. El estudio
   se corrió con MLP; una arquitectura con memoria queda sin evaluar.
6. **N8e no tiene placebo como agente.** La corrida guardó sólo las políticas de B; la
   comparación es el reemplazo de entradas, no un agente entrenado con placebo. Su
   umbral (0,02) y su piso de ruido (0,0252) vienen de fixtures sintéticos, con otra
   distribución de entradas que la real.
7. **Cobertura de GDELT.** 4,4 % de las barras de test de B sin cobertura suficiente
   (27 % en el fold 0); apagón de 17 días en 2025. El fold 9 se analizó con y sin esos
   días (+0,026 en ambos casos): el apagón no explica nada.
8. **A se evaluó dos veces sobre el mismo test** (C3 y R3), y la elección de R3 usó
   fixtures sintéticos, no el test real. El conteo de pruebas se declara en los DSR.
9. **Inferencia no corregida donde no se declaró.** La comparación con exposición
   igualada y el análisis por fold son descriptivos.
10. **Timestamps de fill:** con la convención de cierre de barra, `timestamp_fill`
    llega una barra tarde (CLAUDE.md); la contabilidad por índice no se afecta.

## 8. Sombra sobre mainnet (valida costos) — **PENDIENTE**

> Se completa después de cumplirse los 30 días de snapshots válidos (primer snapshot
> válido 2026-10-20 00:00:30Z; ventana hasta ~2026-11-18). No hay resultados
> en este documento. Se evalúa **una sola vez** (ADR 0007 §6) y no se reabre.

A llenar:

- **Predicción P1 (spread):** mediana de `ratio_spread` ≤ 3,2e-3 sobre snapshots con
  `model.half_spread_rel > 0`. **P2:** `media(real_rel) < media(model_rel)/2` en las 6
  celdas (3 tamaños × 2 lados). Veredicto: SOSTENIDA / PARCIAL / REFUTADA /
  NO EVALUABLE (< 25 de 30 snapshots válidos o < 10 con `model.half_spread_rel` > 0).
  Pendiente: ______
- **Distribución de `costo_real / costo_simulado` por tamaño de orden.** Pendiente.
- **Agente A congelado (10 semillas, 60 000 timesteps, ADR 0009):** desempeño en
  sombra y distribución sobre semillas. Pendiente.
- **Consecuencia para este reporte:** si el costo real difiere materialmente del
  simulado, las secciones 4 a 6 se releen con ese factor (CLAUDE.md, gate de paper
  trading). Pendiente.
- **Limitaciones ya conocidas de la sombra:** no mide la comisión efectiva de una
  cuenta; no hay L2 histórico para verificar el libro de 2020 a 2026.

## 9. Qué sigue

1. **Completar la sección 8** cuando se cumplan los 30 días.
2. **Operar con dinero real: no.** Falta el gate de CLAUDE.md (paper trading de 3 a 6
   meses contrastado contra el simulador), y el estudio no ofrece una estrategia que
   supere a buy-and-hold.
3. **Seguir con noticias es una decisión de Kevin y exige un estudio nuevo**, con un
   ADR escrito antes: otra arquitectura (LSTM, ventana de retornos mayor) o features
   distintas, con las lecciones de aquí (placebo como agente entrenado, piso de ruido
   medido sobre datos reales). No es una extensión de este veredicto: cualquier
   resultado positivo de esa ronda cuenta como una prueba adicional en el DSR.

## Reproducibilidad

Sello y commit de la corrida: `results/estudio/SEAL.json` (commit `d4da04c`, ADR 0012
SHA256 `5bd0b407…`). Registros completos y políticas en el Release `n9-resultados`,
verificables contra `results/estudio/MANIFEST.json`. Reproducir el veredicto:
`python -m agents.cli study assemble --out results/estudio`; el re-diagnóstico:
`python -m agents.cli study diagnose --policies <carpeta> --out results/estudio`.
