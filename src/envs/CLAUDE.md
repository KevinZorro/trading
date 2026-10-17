# CLAUDE.md — `src/envs/`

Se carga al trabajar en `src/envs/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual. La costura `drive()` y el sizer están en
`src/sim/CLAUDE.md`.

## Invariantes del entorno Gymnasium

Consolidados en la Etapa 2. El env es un **wrapper delgado**: rutea, no calcula.

- Espacio de acción del agente en `[0, 1]`. Usa la constante `LONG_ONLY_ACTION_RANGE`
  del motor; no la redeclares en el env.
- **La garantía anti-leakage del env vive en `ObservationBuilder.build`, que recibe solo
  una `Decision`.** El generador no puede impedir que el env tenga la `BarSeries` —la
  necesita para construir el simulador—, así que la garantía es estructural en el
  constructor de la observación. Hay un test que compara dos series idénticas hasta `t`
  y distintas después, y exige observaciones bit-idénticas.
- **El recorte de la acción a `[0,1]` se registra** en `ClipEvent` y en `info`. Un agente
  entrenado sobre un rango que el env recorta en silencio aprende sobre un mundo que no
  existe. Una acción no finita es un error, no un recorte.
- **La observación distingue "no quise" de "no pude"** con tres columnas separadas:
  `last_order_blocked` (veto de riesgo), `last_order_rejected` (rechazo del venue) y
  `last_fill_ratio`. La posición observada es la **realmente llenada**, del ledger.
- **La normalización se ajusta solo con train** (`fit_scaler_on_train` recorre
  `MarketView` sobre la porción de train, así que la serie de test no está en el objeto)
  y el escalador se serializa con el env. No se reajusta nunca, ni en test ni en vivo.
- **El reward se calcula sobre `equity_mark`**, que ya pagó comisión, spread y slippage:
  los costos están dentro por construcción, no restados después. No se usa
  `equity_liquidation` porque cobraría la fricción de salida en cada barra cuando en la
  realidad se paga una vez; la brecha entre ambas series va como feature para que el
  agente la vea.
- **El env no puentea la `RiskLayer`**: la pasa a `drive()` para que se aplique en el
  mismo punto que en vivo. Se recibe como **fábrica**, no como instancia: la capa acumula
  estado por episodio y reusarla arrastraría el kill switch de uno al siguiente.
- **El episodio arranca después del calentamiento** de los indicadores, enviando `None`
  en cada barra previa. Esas barras quedan en el `SimResult` sin fills, que es la verdad.
- **Gymnasium**: `terminated=True` solo por ruina (equity <= 0); agotarse los datos es
  `truncated=True`. La acción del último paso **expira sin ejecutarse** y da reward 0.
