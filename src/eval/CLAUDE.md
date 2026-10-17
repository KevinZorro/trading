# CLAUDE.md — `src/eval/`

Se carga al trabajar en `src/eval/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual.

## Invariantes de `eval/`


Consolidados al cerrar la Etapa 1. Las métricas son funciones puras sobre arrays,
no métodos de `SimResult`: las mismas tienen que poder aplicarse a un benchmark
externo o a la salida de un broker en la Etapa 7.

- **El gap nunca se suma a los costos.** `CostReport.total_costs_paid` es exactamente
  spread + slippage + comisión. El gap tiene línea propia, con signo, y solo aparece
  sumado en `implementation_shortfall`, que se llama por su nombre.
- **`equity_liquidation <= equity_mark` NO implica que su drawdown sea mayor.** Si la
  posición es grande en el pico y chica en el valle, la fricción deprime el pico más
  que el valle y el drawdown de la serie de liquidación resulta *menor*. Es
  contraintuitivo y hay un test que lo fija. No escribas la aserción al revés.
- **Unidad de trade: round-trip flat-to-flat.** De posición cero a cero, con todas las
  compras y ventas intermedias adentro. **La posición abierta al final no entra al win
  rate**: se reporta aparte. Contarla mezcla P&L realizado con no realizado y hace que
  una estrategia que no cierra sus perdedoras se vea mejor de lo que es.
- **Sin trades cerrados, `win_rate` y `profit_factor` son `None`, no `0.0`.** Un cero se
  lee como "perdió todos los trades", que es distinto de "no hizo ninguno". En un estudio
  cuya hipótesis nula es que el agente no opera, esa diferencia es el resultado.
- **La tasa libre de riesgo por defecto es el `cash_rate` de la corrida**, no cero. El
  motor devenga interés sobre el cash ocioso; descontar cero le regala alfa a una
  estrategia que pasa la mitad del tiempo fuera del mercado cobrando esa misma tasa.
  `periodic_rate` usa la misma fórmula que `SimConfig.rate_per_bar` y hay un test que
  las compara.
- **"Volatilidad cero" se evalúa con tolerancia relativa (`DISPERSION_NULA_REL`), no
  contra cero exacto.** Una serie que crece exactamente 10% por barra no produce
  retornos idénticos en float64; sin el umbral el Sharpe sale del orden de 1e15 en vez
  de infinito, y ese número gana cualquier ranking de semillas sin significar nada.
- **Sharpe y Sortino sobre retornos simples**, no logarítmicos. El Sharpe se define sobre
  aritméticos; calcularlo sobre logarítmicos lo subestima de forma creciente con la
  volatilidad.
- **Sortino: segundo momento parcial inferior sobre todas las observaciones**, no solo
  sobre las negativas. Dividir entre la cantidad de negativas infla el ratio de las
  estrategias que rara vez pierden, que son las que hay que mirar con más desconfianza.
- **Una serie que toca cero se trunca en esa barra** y el reporte expone `ruined_at`.
  Después de cero no hay retorno definido y todo lo posterior sale `inf` o `nan`.
- **`annualization_reliable = years >= 1`.** El CAGR y el Sharpe se anualizan siempre
  (en walk-forward todas las ventanas son cortas y hay que anualizarlas igual para
  poder compararlas), pero la extrapolación queda marcada, no disimulada.
- Una serie de `n` observaciones tiene `n - 1` períodos. Un año de barras diarias son
  253 observaciones, no 252.
