# Calibración del presupuesto de entrenamiento (ADR 0010)

Corre en el PC de Kevin (decisión D6 del plan de la Etapa 4). La grilla, los caminos, la
regla de parada y el criterio de 4b están en
[`docs/adr/0010-presupuesto-de-entrenamiento-sobre-4b.md`](adr/0010-presupuesto-de-entrenamiento-sobre-4b.md);
este documento solo dice cómo se ejecuta. El código es `src/agents/calibration.py` y el
subcomando `python -m agents.cli calibrate`.

El script **se niega a correr** si el ADR en disco no tiene el SHA256 sellado en
`calibration.ADR_SHA256`, y si `results/calibracion/` ya tiene un `SEAL.json` de otra
configuración.

## Cómo funciona

- La unidad es **un entrenamiento** (punto, camino, semilla). Cada uno deja
  `results/calibracion/units/G2_p811_s37.json`, escrito de forma atómica. No hay que
  "retomar" nada: volver a lanzar el mismo comando salta lo que ya existe, y un corte
  pierde solo el entrenamiento en curso.
- `run` recorre G1, G2, G3, G4 en orden y **para en el primero que pasa 4b**. Los puntos
  posteriores no se entrenan.
- Cada proceso usa un solo hilo (`OMP_NUM_THREADS=1`); con `--workers N` se lanzan N.
- Al terminar escribe `summary.json` y `MANIFEST.json` (SHA256 de cada entrenamiento) y
  imprime un texto corto para pegar en el hilo.

## Pasos (todo en PowerShell de tu PC)

El proyecto tiene que estar actualizado y con el grupo `rl` instalado (~1 a 3 GB).

**1. Actualizar el repo e instalar (unos 5 a 10 min la primera vez):**

```powershell
cd RUTA_DE_TU_REPO   # la carpeta donde clonaste trading
git fetch origin; git checkout main; git pull
uv sync --frozen --group rl
$env:PYTHONPATH = "src"
```

**2. Medir cuánto tarda en tu PC (menos de 5 min).** Entrena un solo entrenamiento corto,
no guarda resultados y no imprime métricas; solo tiempos y la proyección de horas para
cada punto de la grilla y para la confirmación. Te pregunta cuántos procesos usar:
cuenta tus núcleos físicos y deja uno libre.

```powershell
$w = Read-Host "Cuantos procesos en paralelo (nucleos fisicos menos 1)"
uv run --group rl python -m agents.cli calibrate bench --workers $w
```

Pega la salida en el hilo. El ADR estima unas 5 h para los cuatro puntos con 4 procesos;
tu máquina puede ser más rápida o más lenta.

**3. Lanzar la calibración (horas; se puede cortar y relanzar).** Deja el PC encendido y
sin suspensión. Con `Ctrl+C` o si se apaga, vuelve a correr **este mismo bloque**:

```powershell
$w = Read-Host "Cuantos procesos en paralelo (los mismos del paso 2)"
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate run --workers $w
```

**4. Ver el estado o el resultado sin entrenar nada:**

```powershell
uv run --group rl python -m agents.cli calibrate report
```

`report` falla diciendo cuántos entrenamientos faltan si el primer punto no terminó.

## Qué pegar en el hilo

El texto que imprime `run` o `report` (una línea por punto con el veredicto, el σ de la
política y el tiempo de pared por entrenamiento, y al final `T* = …` o "Ningún punto
pasó"). El resumen y el manifest van al repositorio; los JSON de `units/` van al Release
(regla de tamaño de PR de `CLAUDE.md`).

## Qué viene después

- **`T*` fijado:** sigue la confirmación del ADR 0010 §4 (protocolo completo sobre los
  caminos y semillas registrados, 480 entrenamientos). `bench` ya proyecta sus horas.
  Los subcomandos `arm`, `multipath` y `assemble` no son reanudables por dentro; si el
  tiempo proyectado lo hace impráctico, la partición se propone antes de correr.
- **Ningún punto pasa:** no se fija presupuesto y no se extiende la grilla; un ADR nuevo
  decide la segunda ronda.

## Si torch falla con `WinError 1114` (c10.dll)

En Windows, torch tiene que importarse antes que numpy y pandas. `agents/__init__`
lo hace solo (`agents/_torch_first.py`); si aun así falla, instala o actualiza el
Visual C++ Redistributable x64 (`winget install --id Microsoft.VCRedist.2015+.x64 -e`)
y abre una terminal nueva.

## Resultado de la calibración (2026-10-23)

Los cuatro puntos fallan 4b y **no se fija presupuesto** (ADR 0010 §4). La dispersión de la
acción queda en 0,33 a 0,35 (máximo 0,15) aunque σ baja de 0,81 a 0,05. El resumen está en
`results/calibracion/summary.json`; los JSON de `units/` van a un Release.

## Diagnóstico (`calibrate diagnose`)

Entrena 10 políticas a 120k pasos (5 caminos nuevos × 2 semillas, ~15 a 20 min con 3
procesos) y mide de qué columnas de la observación depende la acción. La regla de lectura
está declarada en el docstring de `src/agents/diagnosis.py`. En PowerShell de tu PC:

```powershell
git pull
$w = Read-Host "Cuantos procesos en paralelo (los mismos de siempre)"
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate diagnose --workers $w
```

Si se corta, relanza el mismo bloque. Pega la salida completa en el hilo.

## Segunda ronda: escalera de regularización (ADR 0011)

La escalera, los caminos y la regla de parada están en
[`docs/adr/0011-segunda-ronda-de-calibracion-regularizacion.md`](adr/0011-segunda-ronda-de-calibracion-regularizacion.md).
Corre R1, R2 y R3 en orden a 120 000 pasos (50 entrenamientos cada uno, ~1 h 10 min con
3 procesos) y para en el primero que pasa 4b. Se niega a correr si el ADR 0011 en disco no
es el sellado. Deja sus entrenamientos en `results/regularizacion/`, aparte de los de la
primera ronda, y compara cada peldaño **pareado contra G1**: para eso lee los
entrenamientos de G1 de `results/calibracion/units/`, que siguen en tu PC desde la primera
ronda. Si no están, corre igual y lo dice al final.

**1. Actualizar el repo (menos de 1 min).** En PowerShell de tu PC:

```powershell
cd RUTA_DE_TU_REPO   # la carpeta donde clonaste trading
git fetch origin; git checkout main; git pull
uv sync --frozen --group rl
```

**2. Lanzar la escalera (de 1 h 10 min a unas 3 h 30 min; se puede cortar y relanzar).**
Deja el PC encendido y sin suspensión. Si se corta, vuelve a correr **este mismo bloque**:

```powershell
$w = Read-Host "Cuantos procesos en paralelo (los mismos de siempre)"
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate ladder --workers $w
```

**3. Ver el resultado sin entrenar nada:**

```powershell
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate ladder-report
```

Pega en el hilo lo que imprime: una línea por peldaño con su veredicto, la diferencia
contra G1 y su `t` entre caminos, y al final `C* = …` o "Ningún peldaño pasó". Si un
peldaño pasa, la confirmación del protocolo completo necesita un PR aparte que la vuelva
reanudable antes de correrla (ADR 0011 §7).

## Resultado de la escalera (2026-10-23)

R1 y R2 fallan y **R3 pasa 4b**: tiempo invertido 0,97, dispersión de la acción 0,111,
rotación ~8 al año. Pareado contra G1, la dispersión baja 0,21 (`t` = −6,5) y el tiempo
invertido sube 0,15 (`t` = +6,7). `C*` = R3 a 120 000 pasos. Resumen condensado en
`results/regularizacion/summary.json`. La σ de la política queda en ~1,0 en los tres
peldaños: la exploración no se apagó, y por eso el par 4a↔4b de la confirmación es el
que decide si el agente reconoce el drift o compra por defecto.

## Confirmación con `C*` (ADR 0011 §5)

Re-corre el protocolo completo con R3 sobre los caminos y semillas registrados: 8 brazos
de un camino × 10 semillas y 4 multicamino de 10 × 10, 480 entrenamientos, más 10 de
re-diagnóstico. Cada entrenamiento deja su JSON en `results/confirmacion/units/`, así que
**un corte solo pierde el entrenamiento en curso**. Al terminar arma los `arm_*.json` y
`multipath_*.json` en `results/confirmacion/protocolo/` y los juzga `assemble`, como en
la Etapa 3. Como R3 es el último peldaño, si 4b falla aquí no hay un peldaño siguiente
que probar (ADR 0011 §5, regla 4).

**Tiempo: unas 13 a 16 h con 3 procesos.** Los 370 entrenamientos MLP tardan ~4 min cada
uno, y los 110 LSTM tardan más. Deja el PC encendido y sin suspensión. Puedes cortarlo
de noche y relanzarlo al día siguiente.

**1. Actualizar el repo (menos de 1 min).** En PowerShell de tu PC:

```powershell
cd RUTA_DE_TU_REPO   # la carpeta donde clonaste trading
git fetch origin; git checkout main; git pull
uv sync --frozen --group rl
```

**2. Lanzar o relanzar la confirmación.** Si se corta, vuelve a correr este mismo bloque:

```powershell
$w = Read-Host "Cuantos procesos en paralelo (los mismos de siempre)"
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate confirm --workers $w
```

**3. Ver el veredicto sin entrenar nada:**

```powershell
$env:PYTHONPATH = "src"
uv run --group rl python -m agents.cli calibrate confirm-report
```

Pega en el hilo lo que imprime: el reporte del protocolo nivel por nivel y la línea del
re-diagnóstico. Si faltan entrenamientos, dice cuántos.

## Resultado de la confirmación (2026-10-25)

**El protocolo completo pasa con R3, y es la primera vez.** Por la regla 3 del ADR 0011
§5, R3 a 120 000 timesteps queda fijo para todo el estudio. Resumen en
`results/confirmacion/summary.json`, con el SHA256 del `protocol.json` completo.

| Nivel | Etapa 3 (60k, config. de `main`) | Confirmación (R3) |
|---|---|---|
| 0 | PASS, capture 1,00 | PASS, capture 1,00 |
| 1 | PASS; capture con β = −0,1: 0,30 | PASS; capture con β = −0,1: 0,77 |
| 2 | PASS, la rotación cae 19,3 % | PASS, la rotación cae 17,0 % (piso 10 %) |
| 3 | MEASURED, memoriza 10/10; MLP = LSTM | MEASURED, memoriza 10/10; LSTM − MLP = +0,045 de capture (`t` = 3,7) |
| 4a | PASS, `t` = +0,69 | PASS, `t` = +0,29 |
| 4b | FAIL: invertido 0,63, dispersión 0,383 | PASS: invertido 0,93, dispersión 0,141 |
| Par | PARTIAL | PASS: +0,69 de tiempo invertido, `t` = 18,9 |

Para leerlo:

- 4b pasa con poco margen (0,141 contra 0,15). Los caminos 761 (0,36) y 701 (0,19) rotan
  por encima del umbral. El criterio es la media entre caminos, y la media pasa.
- La σ de la política queda en ~1,0 en todos los niveles: la exploración no se apaga a
  120 000 pasos con lr 1e-4. El par descarta que el agente compre por defecto (en 4a se
  invierte 0,24, en 4b 0,93, con los mismos shocks). Aun así, la σ va al lado de la
  comparación A contra B.
- En el nivel 3 la ventaja del LSTM es un hallazgo nuevo, y no bloquea nada.

### Re-diagnóstico con R3 (ADR 0011 §6)

Mismo diagnóstico que #57, sobre 10 políticas R3. Mediana de la dispersión de la acción
en lazo cerrado, fijando un grupo de la observación por vez:

| Grupo fijado | Dispersión (mediana) |
|---|---|
| Ninguno (línea base) | 0,109 |
| Retornos | 0,101 |
| Técnicos | 0,022 |
| Posición | 0,111 |
| Resultado de la orden | 0,109 |

El script lee "los cuatro grupos sostienen la rotación", pero esa lectura es trivial: la
línea base ya está por debajo del umbral de 0,15, así que cualquier grupo "alcanza". Lo
que informa es la caída. Con R3, lo que queda moviendo la acción son los técnicos: al
fijarlos la dispersión cae un 80 %. En #57, con la configuración de `main`, eran sobre
todo los retornos (0,345 → 0,19). `lookback` 1 les quitó ese papel. Con el Agente B
conviene repetir este diagnóstico con las noticias como grupo propio.
