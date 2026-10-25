# CLAUDE.md — `src/agents/`

Se carga al trabajar en `src/agents/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual.

## Invariantes del Agente A y del protocolo (Etapa 3)


Ver `docs/adr/0003-agente-a-y-protocolo-de-validacion.md`.

- **`torch` y `stable-baselines3` son el grupo opcional `rl`**, que
  `uv sync --frozen` no instala: la rueda de torch de PyPI arrastra el stack de
  CUDA (>3 GB) y sacaría al job principal de CI de sus 5 minutos. El núcleo de
  `agents/` —protocolo, reporte por semillas, walk-forward— no depende de torch
  y se testea con políticas deterministas inyectadas. El import de SB3 es
  diferido. **Consecuencia declarada: el adaptador de SB3 no está cubierto por
  el CI**; su test lleva el marcador `rl` y se saltea donde el grupo no está.
- **El agente y los baselines arrancan en la misma barra.** `WarmupDelay`
  envuelve una `Strategy` para que no opere antes del calentamiento. Sin eso, las
  26 barras de ventaja son retorno regalado a uno de los dos y la diferencia
  medida deja de ser atribuible a la estrategia. El techo se pide con el mismo
  `first_decision` y el mismo `safety`.
- **Los criterios de aprobación se declaran antes de correr** (`ProtocolThresholds`)
  y viajan serializados con el resultado. Un umbral elegido después de ver los
  números no es un criterio, es una descripción.
- **El protocolo para en el primer fallo** y deja los niveles no corridos como
  `SKIPPED` con su motivo. Un reporte con tres niveles y sin explicación se lee
  como si el protocolo tuviera tres niveles.
- **Hay tres veredictos, no dos.** El nivel 3 es `MEASURED`: adaptarse y
  memorizar son ambos hallazgos válidos, y forzar un PASS/FAIL sería inventar
  una hipótesis después del hecho.
- **El criterio del nivel 2 está sobre la rotación, no sobre el retorno.**
  Rendir menos con costos es automático —se restan del equity aunque el agente
  los ignore—; lo que prueba que la penalización llegó al reward es el cambio de
  comportamiento.
- **La métrica del barrido de SNR es `capture`, no el retorno crudo.** Al bajar
  el SNR el techo baja también, así que sin normalizar la degradación se mide a
  sí misma.
- **`summarize` falla con menos de 10 semillas.** El escape explícito enciende
  `below_minimum_seeds`, que sale en el JSON y en el render. Se puede producir un
  resultado con pocas semillas; lo que no se puede es que parezca uno con muchas.
- **El criterio del nivel 4 es un contraste, no un umbral.** El exceso sobre estar
  siempre invertido se compara contra la dispersión **entre caminos**, no contra un
  número fijo. El umbral anterior (`max_median_excess = 0.05`) no era demasiado laxo ni
  demasiado estricto: estaba mal planteado, porque medía una cantidad contra una escala
  que no le correspondía. `decompose_variance` exige al menos dos caminos y el nivel 4
  exige diez.
- **El `t` del drift sobre la ventana de entrenamiento va al lado del veredicto del
  nivel 4**, no en una nota al pie. Si el drift no es detectable en la muestra —en el
  régimen `medium`, con μ=8% y 30% de volatilidad sobre 4800 barras, la mediana medida
  sobre 10 caminos es 0.84, con rango −0.91 a +2.21— entonces
  "converger a estar invertido" le pide al agente aprender algo que la muestra no
  contiene, y un fallo del nivel no es un fallo del agente. `drift_t_statistic` lo
  calcula y `MultiPathResult.drift_detectable` lo contrasta.
- **El nivel 4 son dos hipótesis y un par** (ADR 0005). **4a** es control negativo
  puro con un solo criterio, el exceso contra la dispersión entre caminos. **4b** usa
  un drift pre-registrado para ser detectable (μ=0.42, `t` poblacional 5.46 con la
  fórmula de log-drift `(μ−θ/2)·√n/√(θ·bpy)`) y exige converger a invertido sin
  rotar. **El par** —diferencia pareada de tiempo invertido, mismos shocks— es lo
  único que distingue "reconoce el drift" de "compra por defecto".
- **Un agregado no puede dar verde sobre un componente en rojo.** Existe
  `Verdict.PARTIAL`: el par que responde al drift pero con 4b en FAIL es PARTIAL,
  nunca PASS, y el hallazgo lleva las dos cifras con sus dos umbrales.
- **Deuda antes de la Etapa 5: el presupuesto de entrenamiento.** 4b no converge por
  presupuesto, no por el reward (0.8→1.0 vale `t`=23 sobre 60k timesteps) ni por
  `ent_coef` (es 0.0): el `log_std` aprendido queda en σ=0.91 a 60k timesteps, y a
  240k el tiempo invertido sube de 0.67 a 0.78. Es una sola corrida y se declara así.
  Comparar A contra B con políticas a medio entrenar mediría varianza de
  entrenamiento, no noticias: el σ de la política se reporta junto a la comparación.
  **No se aplica retroactivamente a 4b**; cuando cambie, se re-corre el protocolo
  completo.
- **Deuda saldada el 2026-10-25: la configuración del estudio es R3 a 120 000
  timesteps** (ADR 0011): `lookback` 1, `learning_rate` 1e-4, `n_epochs` 5, red 32×32,
  `gamma` 0,9. El presupuesto solo no era la palanca (ADR 0010: G1 a G4 fallan con
  σ = 0,05); la regularización sí. El protocolo completo con R3 está en verde sobre los
  caminos registrados (`results/confirmacion/summary.json`): 4b pasa con dispersión
  0,141 contra 0,15, poco margen, y el par con `t` = 18,9. **La σ de la política queda
  en ~1,0**: la evaluación es determinista y el par descarta "compra por defecto", pero
  la σ se sigue reportando junto a la comparación A contra B. Agente A, Agente B,
  placebo y re-corridas usan esta configuración; los Agentes A congelados de la sombra
  (60k, ADR 0009) no se tocan dentro de su ventana.
- **El presupuesto de entrenamiento se calibra sobre el fixture 4b, nunca sobre datos
  reales.** Se aumentan los timesteps hasta que 4b pase su criterio pre-registrado, sin
  tocar ese criterio, y ese presupuesto queda fijo para todo el estudio. Así no se
  gasta información del mercado en elegir hiperparámetros. La grilla se declara en un
  ADR **antes** de correrla. Hasta entonces, todo resultado sobre datos reales declara
  que la política está subentrenada y que valida el pipeline, no concluye sobre el
  mercado.
- **La calibración del presupuesto (ADR 0010) no usa los caminos del protocolo** y sus
  umbrales son los de `main` salvo `level_4_min_paths = 5`, que es un override local a
  `calibration.thresholds()`: el default de `ProtocolThresholds` sigue siendo 10 y lo fija
  un test. La unidad reanudable es un entrenamiento; `calibrate run` para en el primer
  punto que pasa y se niega a correr si el SHA256 del ADR no es el sellado.
- **La segunda ronda (ADR 0011) es `calibrate ladder`** (`agents/regularization.py`):
  peldaños R1 a R3 acumulativos, a 120 000 timesteps, sobre los mismos caminos y semillas
  que G1 y con contraste pareado contra G1. Tiene su propio SHA256 de ADR y su propio
  sello, así que sus entrenamientos no se mezclan con los del ADR 0010. La observación
  del peldaño (`lookback`) va en cada registro, porque `ArmResult` no la serializa.
- **La confirmación de `C*` es `calibrate confirm`** (`agents/confirmation.py`):
  `C*` = R3 (`CONFIRMED_RUNG`), 480 entrenamientos con unidad (brazo, camino, semilla)
  reanudable, más el re-diagnóstico sobre `C*`. Arma los `arm_*.json` y
  `multipath_*.json` de siempre y los juzga `cli assemble`: los criterios no tienen
  una segunda implementación.
- **Entrenar y juzgar son dos comandos** (`agents.cli arm` / `assemble`). Los
  criterios se aplican siempre sobre resultados guardados: revisar un umbral no
  exige reentrenar, y si alguien lo cambia después de ver los números, se ve en
  el diff. `ArmResult.from_dict` recalcula las distribuciones desde las corridas
  en vez de leerlas.
- **La política recurrente es una opción y se corre donde tiene sentido.** En
  los niveles 0, 1 y 2 el estado es completamente observable (el óptimo depende
  solo de `r_t`, que está en la observación) y la memoria solo agrega parámetros
  que sobreajustar. Donde importa es en el nivel 3.
- **El nivel 3 está re-reportado con N=10 caminos y "memoriza" se sostiene en
  los diez**, MLP y LSTM. La comparación pareada entre arquitecturas da una
  diferencia de `capture` de −0.0001 con `t` = 0.00: la memoria no aporta nada
  sobre ese fixture. **Los niveles 1 y 2 quedan como deuda documentada y no se
  re-corren**: sus conclusiones son robustas a la ruta (una degradación monótona
  de cuatro puntos con separación grande, y un contraste de comportamiento dentro
  del mismo proceso), y las ~500 corridas rinden más en la Etapa 6. Ver ADR 0004.
- **Walk-forward: los tests de ventanas consecutivas no se solapan** con el paso
  por defecto, y `coverage` reporta `overlapping_test_bars` cuando sí. Con tests
  solapados hay más observaciones que información independiente, y hay que
  decirlo antes de calcular cualquier estadístico con ellas.
