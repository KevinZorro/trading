# ADR 0010 — Presupuesto de entrenamiento calibrado sobre el fixture 4b

- **Estado:** propuesto. Se sella al mergearse: el script de la calibración (PR P2)
  no corre si el SHA256 de este archivo no coincide con el sellado, igual que el
  runner del ADR 0006.
- **Fecha:** 2026-10-20
- **Contexto de etapa:** prerrequisito de la Etapa 5, desarrollado en paralelo con el
  pipeline de noticias de la Etapa 4 (`/mnt/project-files/etapa4-noticias/PLAN.md`,
  pista P).
- **A la fecha de este ADR no se corrió ningún punto de la grilla.** Lo único medido
  con más de 60 000 timesteps es el diagnóstico del ADR 0005: **una** corrida, camino
  701 y semilla 11, a 240 000 timesteps (tiempo invertido 0,777, dispersión 0,366,
  σ = 0,651). Se declara porque informó el rango de la grilla. Ese camino no se usa
  para calibrar (sección 3).

## Contexto

`src/agents/CLAUDE.md` lo dejó como deuda: el nivel 4b no converge a 60 000
timesteps por presupuesto, no por el reward ni por `ent_coef` (ADR 0005,
diagnóstico). El `log_std` aprendido queda en σ = 0,91 y la media de la política
alterna entre plano y all-in. Comparar el Agente A contra el B en ese estado mediría
varianza de entrenamiento, no noticias.

La regla ya estaba escrita: **el presupuesto se calibra sobre el fixture 4b, nunca
sobre datos reales; se aumentan los timesteps hasta que 4b pase su criterio
pre-registrado, sin tocar ese criterio; ese presupuesto queda fijo para todo el
estudio; y la grilla se declara en un ADR antes de correrla.** Este es ese ADR.

## 1. Qué se calibra y qué no

**Se calibra un solo número: `PPOConfig.total_timesteps`.** Todo lo demás de
`PPOConfig` queda en sus valores actuales y serializados (`n_steps` 2048,
`batch_size` 256, `n_epochs` 10, `learning_rate` 3e-4, `gamma` 0,99, `ent_coef`
0,0, red 64×64, `reward_scale` 100, MLP).

**`log_std_init` y un `log_std` programado quedan fuera.** El ADR 0005 los sugirió.
Meterlos sería un segundo eje: multiplica el cómputo y abre una elección más que
después hay que contar como intento. Si ningún punto de la grilla pasa (sección 5),
la segunda ronda los considera, en un ADR nuevo y antes de correrla.

El criterio de 4b es **el del ADR 0005, sin cambios**:

1. `t` mediano del drift en la ventana de entrenamiento ≥ 3,0 (calibración del
   fixture, no del agente);
2. media entre caminos del tiempo invertido ≥ 0,80;
3. media entre caminos de la dispersión de la acción dentro del episodio ≤ 0,15.

Umbrales de `ProtocolThresholds()` tal como están en `main` a la fecha de este ADR.

## 2. La grilla

| Punto | `total_timesteps` | Pasadas por las 4.800 barras de train |
|---|---|---|
| G1 | 120 000 | 25 |
| G2 | 240 000 | 50 |
| G3 | 480 000 | 100 |
| G4 | 960 000 | 200 |

Duplicar en cada paso es la forma más simple de cubrir un orden de magnitud con
cuatro puntos. 60 000 no está porque ya se sabe que falla (ADR 0005). El techo
de 960 000 lo pone el cómputo (sección 6), no un resultado.

## 3. Caminos y semillas

**La calibración no usa los caminos del protocolo.** `PATH_SEEDS` (701 a 761) son los
caminos registrados en los que se juzga el protocolo. Elegir el presupuesto con ellos
y después confirmar con ellos mismos sería elegir y juzgar sobre la misma muestra:
un presupuesto que pasa por suerte en esos diez caminos pasaría también la
confirmación.

- **Calibración:** 5 caminos nuevos, `CALIBRATION_PATH_SEEDS = (809, 811, 821, 823,
  827)`, disjuntos de `PATH_SEEDS`, × las 10 `STUDY_SEEDS`. Son 50 entrenamientos por
  punto. Cinco y no diez para que la grilla quepa en el cómputo; la confirmación
  usa los diez.
- **Confirmación:** el protocolo completo re-corrido sobre `PATH_SEEDS` y
  `STUDY_SEEDS` con el presupuesto elegido (sección 4).

Las semillas de caminos se fijan aquí, antes de generar ningún camino con ellas.

## 4. Regla de decisión

Los puntos se corren **en orden ascendente** y se para en el primero que pasa.

1. **Calibración.** Se elige `T*` = el menor punto de la grilla en el que 4b pasa sus
   tres criterios sobre los 5 caminos de calibración.
2. **Confirmación.** Con `T*` se re-corre **el protocolo completo** (niveles 0 a 3,
   4a, 4b y el par) sobre los caminos y semillas registrados, con
   `agents.cli arm`, `multipath` y `assemble` como en la Etapa 3. No solo 4b: más
   presupuesto puede hacer que el agente invente señal en 4a o rote en el nivel 2.
3. **Resultado.**
   - Protocolo completo en verde → **`T*` queda fijo para todo el estudio**: el
     Agente A y el Agente B de la Etapa 5, el placebo y cualquier re-corrida.
   - 4b falla en la confirmación habiendo pasado en la calibración → se prueba el
     punto siguiente de la grilla, **una sola vez**, con la misma confirmación. Se
     reporta que el primero no confirmó.
   - Falla otro nivel (0 a 3, 4a o el par) → el protocolo se detiene como siempre y
     **no se fija presupuesto**. Es un hallazgo y va a un ADR nuevo.
   - Ningún punto pasa la calibración → **no se fija presupuesto** y no se extiende la
     grilla sobre la marcha. Un ADR nuevo decide la segunda ronda
     (`log_std_init`, más timesteps), antes de correrla.

## 5. Qué se reporta

Para cada punto corrido, con las dos varianzas separadas (`decompose_variance`):

- las tres cantidades del criterio de 4b, por camino y semilla, con mediana, p25,
  p75, mínimo y máximo;
- **σ de la política** al final del entrenamiento (`exp(log_std)`), como curva sobre
  la grilla. No decide; es la evidencia de que la exploración se apagó, que el
  ADR 0005 pidió reportar junto a la comparación A contra B;
- el exceso sobre estar siempre invertido, como evidencia;
- tiempo de pared por entrenamiento y la máquina donde corrió.

Al repositorio van un resumen y un manifest con el SHA256 de cada artefacto; los
JSON completos van al Release (regla de tamaño de PR de `CLAUDE.md`).

## 6. Cómputo

Corre en el PC de Kevin (decisión D6 del plan). La medición de referencia sirve
para dimensionar la grilla, no para elegir nada. Se hizo el 2026-10-20 en un
contenedor de la nube (Xeon de 2,1 GHz, un núcleo, `OMP_NUM_THREADS=1`) con
`agents.cli arm --level level_4b --seeds 1`: 20 000 timesteps tardan 23 s y
120 000 tardan 101 s. Da **unos 0,78 ms por timestep, más unos 8 s fijos** de
carga y evaluación. La grilla de la sección 2 ya estaba escrita antes
de medir. La salida de la corrida de 20 000 imprimió sus métricas y se vieron de
pasada (un camino, una semilla, fuera de la grilla); no cambiaron nada y no se
guardan.

| Punto | Entrenamientos | Timesteps totales | Un núcleo | Cuatro procesos en paralelo |
|---|---|---|---|---|
| G1 | 50 | 6,0 M | ~1,4 h | ~20 min |
| G2 | 50 | 12,0 M | ~2,7 h | ~40 min |
| G3 | 50 | 24,0 M | ~5,3 h | ~1,3 h |
| G4 | 50 | 48,0 M | ~10,5 h | ~2,6 h |
| Confirmación con G3 | 480 | 230 M | ~50 h | ~13 h |
| Confirmación con G4 | 480 | 461 M | ~100 h | ~25 h |

El peor caso de la calibración (los cuatro puntos) es de unas 5 h con cuatro
procesos. La confirmación con LSTM es más lenta que lo estimado para MLP, así que
esas filas son un piso. El PC de Kevin probablemente es más rápido por núcleo que
este contenedor; P2 mide el primer entrenamiento allí antes de lanzar el resto y da
la estimación real.

Como se para en el primer punto que pasa, el costo esperado de la calibración es
menor que la suma. **La confirmación es lo caro**: re-corre el protocolo entero, que
son 8 brazos de un camino × 10 semillas (nivel 0, nivel 1 con cuatro betas, nivel 2,
nivel 3 MLP y LSTM) más 4 multicamino de 10 × 10 (nivel 3 MLP y LSTM, 4a y 4b).

## 7. Cómo se traslada el presupuesto a los datos reales

**El presupuesto se traslada en timesteps absolutos, no en pasadas por los datos.**
Lo que el ADR 0005 diagnosticó es un problema de **optimización**: el `log_std` se
encoge con los pasos de gradiente, no con la cantidad de barras distintas. Por eso lo
que se calibra son pasos de optimización.

La consecuencia hay que escribirla antes: en el walk-forward real cada fold tiene
1.000 barras de entrenamiento, no 4.800, así que `T*` implica **4,8 veces más pasadas
por fold** que en el fixture (con G3, unas 480). Más pasadas por una muestra chica
favorecen la memorización. El walk-forward fuera de muestra es lo que lo mide, y el
reporte de la Etapa 5 lo declara al lado del σ de la política.

La alternativa, calibrar sobre un 4b con 1.000 barras de entrenamiento, no es
posible: con μ = 0,42 el `t` poblacional del drift sobre 1.000 barras es 2,49, por
debajo del 3,0 que el criterio 1 exige al fixture. El nivel no podría medir lo que
pretende.

## 8. Lo que este ADR no cambia

- Los criterios de 4a, 4b y el par (ADR 0005), ni ningún otro umbral del protocolo.
- El veredicto histórico de 4b a 60 000 timesteps: sigue en FAIL.
- **Los Agentes A congelados de la sombra** (ADR 0009, 60 000 timesteps). No se
  reemplazan dentro de la ventana de 30 días que cierra hacia el 2026-11-18. Si se
  re-congelan con `T*`, es en una adenda posterior al cierre de esa ventana.
- El ADR 0006 y el resultado de C3, que corrieron con 60 000 timesteps y quedan como
  registro.
