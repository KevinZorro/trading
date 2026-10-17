# CLAUDE.md — `src/risk/`

Se carga al trabajar en `src/risk/`. Lo transversal (principios, CI, seguridad,
flujo de trabajo) está en el `CLAUDE.md` de la raíz y rige igual. Ver
`docs/adr/0001-arquitectura-para-ejecucion-real.md`.

## Invariantes de ejecución real

- **La `RiskLayer` rechaza, nunca redimensiona.** Misma regla que `check_tradable`.
  Recortar una orden al límite hace indistinguible "el agente pidió esto" de "el límite
  lo recortó hasta acá". `GateDecision` no tiene dónde devolver una orden modificada.
- **Los vetos de riesgo van en `SimResult.gate_rejections`, lista separada de los
  rechazos del venue.** "El venue no pudo llenarla" y "nuestra capa no la dejó salir"
  son diagnósticos distintos; agregarlos es el mismo error que agregar el gap a los costos.
- **El kill switch permite cerrar, no abrir.** Bloquear todo deja la posición abierta
  justo cuando algo salió mal. Solo pasan las órdenes que reducen exposición. La
  reposición es manual: si se repusiera solo sería una pausa, no un kill switch.
- **El ancla del límite de pérdida diaria es el equity con que cerró el día anterior**,
  no el primero del día actual. Con barras diarias —donde una barra es un día— anclar a
  la propia barra compara el equity contra sí mismo y el límite no muerde nunca.
  Es pérdida diaria, no drawdown intradiario: el ancla es la apertura, no el máximo.
- **`risk` depende de `sim`, nunca al revés.** La costura es el `Protocol`
  `sim.gate.OrderGate`. El simulador no sabe que existe una política de riesgo, igual
  que no lo sabe un broker real.
