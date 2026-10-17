"""Verifica que partir ``CLAUDE.md`` no perdio ni duplico ninguna regla.

Compara el ``CLAUDE.md`` de una referencia de git (el de antes de partir) contra
el ``CLAUDE.md`` de la raiz mas los ``src/*/CLAUDE.md`` del arbol de trabajo.

La unidad es cada viñeta (``- `` o ``N. ``, con sus lineas de continuacion),
cada parrafo y cada bloque de codigo. Los titulos no cuentan: se pueden
reorganizar. Se compara con los espacios normalizados, porque mover un bloque
puede cambiar su sangria.

- **Perdida:** una unidad del original que no aparece en ningun archivo nuevo.
- **Duplicada:** una unidad que aparece en mas de uno. Una regla con dos copias
  termina con dos versiones.
- **Nueva:** una unidad de los archivos nuevos que no estaba en el original. No
  es un error: se lista para que el revisor lea solo lo que cambio.

Uso::

    python scripts/verificar_particion_claude_md.py --base origin/main

Sale con codigo 1 si hay alguna unidad perdida o duplicada.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_VINETA = re.compile(r"^\s*(?:-|\d+\.)\s")


def unidades(texto: str) -> list[str]:
    """Parte un markdown en viñetas, parrafos y bloques de codigo normalizados."""
    salida: list[str] = []
    actual: list[str] = []
    en_codigo = False

    def cerrar() -> None:
        if actual:
            salida.append(" ".join(" ".join(actual).split()))
            actual.clear()

    for linea in texto.splitlines():
        if linea.strip().startswith("```"):
            if en_codigo:
                actual.append(linea)
                cerrar()
                en_codigo = False
            else:
                cerrar()
                actual.append(linea)
                en_codigo = True
            continue
        if en_codigo:
            actual.append(linea)
        elif not linea.strip() or linea.startswith("#"):
            cerrar()
        elif _VINETA.match(linea):
            cerrar()
            actual.append(linea)
        else:
            actual.append(linea)
    cerrar()
    return salida


def normalizar(texto: str) -> str:
    return " ".join(texto.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--raiz", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    original = subprocess.run(
        ["git", "show", f"{args.base}:CLAUDE.md"],
        cwd=args.raiz,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    archivos = [args.raiz / "CLAUDE.md", *sorted(args.raiz.glob("src/*/CLAUDE.md"))]
    textos = {
        p.relative_to(args.raiz).as_posix(): p.read_text(encoding="utf-8")
        for p in archivos
    }
    destinos = {nombre: normalizar(t) for nombre, t in textos.items()}

    perdidas: list[str] = []
    duplicadas: list[tuple[str, list[str]]] = []
    for unidad in unidades(original):
        donde = [n for n, t in destinos.items() if unidad in t]
        if not donde:
            perdidas.append(unidad)
        elif len(donde) > 1:
            duplicadas.append((unidad, donde))

    original_norm = normalizar(original)
    nuevas = [
        (nombre, u)
        for nombre, t in textos.items()
        for u in unidades(t)
        if u not in original_norm
    ]

    total = len(unidades(original))
    print(f"unidades del original: {total}")
    for nombre, t in textos.items():
        print(f"  {nombre}: {len(t.encode('utf-8'))} bytes")
    print(f"perdidas: {len(perdidas)}")
    for u in perdidas:
        print(f"  - {u[:120]}")
    print(f"duplicadas: {len(duplicadas)}")
    for u, donde in duplicadas:
        print(f"  - {u[:100]}  ->  {donde}")
    print(f"nuevas (para revisar, no son error): {len(nuevas)}")
    for nombre, u in nuevas:
        print(f"  [{nombre}] {u}")
    return 1 if perdidas or duplicadas else 0


if __name__ == "__main__":
    sys.exit(main())
