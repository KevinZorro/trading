# Textos para CV, LinkedIn y portafolio

Basados en el [reporte final](REPORTE.md). Cada cifra sale de ahí; no las redondees
hacia arriba. Elige 3 a 5 viñetas según el puesto.

## Título del proyecto

- **ES:** Estudio pre-registrado de aprendizaje por refuerzo con señales de noticias
  para trading de BTCUSDT, con paridad backtest-vivo.
- **EN:** Pre-registered reinforcement learning study of news signals for BTCUSDT
  trading, with backtest-live parity.

## Viñetas para el CV (español)

- Diseñé y pre-registré un estudio controlado (agente con noticias vs. sin noticias
  vs. placebo) de aprendizaje por refuerzo (PPO) sobre BTCUSDT: 330 entrenamientos en
  walk-forward de 11 ventanas × 10 semillas, criterio de decisión y número de pruebas
  sellados por hash antes de tocar los datos de test.
- Construí una tubería de noticias *point-in-time* sobre 2,7 M artículos de GDELT y
  1,4 M titulares puntuados con FinBERT, con regla anti-lookahead verificada por tests
  dedicados y latencia calibrada contra el comportamiento medido en vivo.
- Reporté un resultado negativo bien medido: las noticias no aportan (Deflated Sharpe
  Ratio 0,15 y 0,20 contra 0,95), y un diagnóstico por perturbación mostró que el
  agente no usaba esa entrada; un control positivo descartó una tubería rota.
- Validé el agente sobre series sintéticas con óptimo conocido, separando varianza de
  mercado y de entrenamiento; eso detectó un falso hallazgo cinco veces menor que la
  dispersión entre caminos independientes.
- Desarrollé un sistema en Python con paridad backtest-vivo (interfaz de ejecución
  única, reloj inyectable, IDs de orden idempotentes, capa de riesgo independiente) y
  lo desplegué en Azure con Docker: sombra de solo lectura sobre el libro L2 de Binance
  para validar el modelo de costos, y venue de testnet para validar la mecánica.
  1 520 tests sin red, `mypy --strict` y CI con cobertura ≥ 85 %.

## CV bullets (English)

- Designed and pre-registered a controlled reinforcement learning study (news-aware
  vs. price-only vs. placebo PPO agents) on BTCUSDT: 330 trainings across an 11-fold
  walk-forward × 10 seeds, with the decision rule and number of trials sealed by hash
  before touching the test data.
- Built a point-in-time news pipeline over 2.7 M GDELT articles and 1.4 M
  FinBERT-scored headlines, with a dedicated anti-lookahead test suite and a latency
  rule calibrated against live measurements.
- Reported a well-measured negative result: news added no edge (Deflated Sharpe Ratio
  0.15 and 0.20 vs. a 0.95 threshold), and a perturbation analysis showed the agent
  ignored the news inputs; a positive control ruled out a broken pipeline.
- Validated the agent on synthetic markets with a known optimum, separating market
  variance from training variance; this caught a spurious finding five times smaller
  than the spread across independent paths.
- Engineered a Python system with backtest-live parity (single execution interface,
  injectable clock, idempotent client order IDs, independent risk layer) and deployed
  it on Azure with Docker: a read-only Binance L2 order-book shadow to validate the
  cost model, plus a testnet venue to validate execution mechanics. 1,520 offline
  tests, `mypy --strict`, CI with ≥ 85 % coverage.

## LinkedIn / portafolio

**ES (corta, ~60 palabras):**
¿Aportan alfa las noticias a un agente de trading con aprendizaje por refuerzo? Diseñé
un estudio pre-registrado sobre BTCUSDT con costos reales, walk-forward, 10 semillas,
placebo y Deflated Sharpe Ratio. La respuesta honesta fue que no: el agente no aprendió
a usar las noticias. El sistema corre el mismo código en simulación y contra Binance
(sombra en mainnet y testnet).

**EN (short, ~60 words):**
Do news signals add alpha to a reinforcement learning trading agent? I designed a
pre-registered study on BTCUSDT with real transaction costs, walk-forward evaluation,
10 seeds, a placebo arm and the Deflated Sharpe Ratio. The honest answer was no: the
agent did not learn to use the news. The same code runs in simulation and against
Binance (mainnet shadow and testnet).

## Stack (para la sección de habilidades)

Python 3.11/3.12, PyTorch, Stable-Baselines3 (PPO), Gymnasium, NumPy, pandas,
FinBERT (Hugging Face Transformers), GDELT, API REST y WebSocket de Binance, Docker,
Azure, GitHub Actions, pytest, mypy, ruff, uv.

## Cómo hablar del resultado negativo en una entrevista

- Empieza por la pregunta y el diseño, no por el número: "quería saber si las
  noticias ayudan, y diseñé el experimento para que un falso positivo fuera difícil".
- El negativo es el resultado esperado de un diseño exigente, y viene con su
  diagnóstico: el agente ignoró la entrada, y el control positivo muestra que la
  tubería funcionaba.
- No digas que el sistema "gana" ni que "supera al mercado": ningún brazo superó a
  buy-and-hold. Lo que demuestra el proyecto es rigor y capacidad de construir un
  sistema listo para producción.
