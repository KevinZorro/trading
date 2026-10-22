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
