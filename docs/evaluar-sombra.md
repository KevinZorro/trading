# Evaluar la sombra de costos (ADR 0007 §6)

La predicción sobre el spread se evalúa **una vez**, cuando se cumplen los 30 días UTC
desde el primer snapshot diario válido (2026-10-20 00:00:30Z, así que la ventana
cierra el 2026-11-18 a las 00:10:30Z). Antes de esa hora el script solo muestra el
avance: cuántos diarios válidos hay y si todavía se puede llegar a 25. No calcula P1 ni
P2.

El script lee una **copia** de `/var/lib/shadow`. No necesita torch, red ni tocar
la VM más allá de copiar los archivos, y la sombra sigue corriendo mientras tanto.

## 1. Empaquetar los snapshots (dentro de la VM, después del ssh)

Solo lee `/var/lib/shadow`; no reinicia nada.

```bash
sudo sh -c 'cd /var/lib/shadow && tar -czf /home/azureuser/sombra-snapshots.tgz snapshots-*.jsonl'
sudo chown azureuser: /home/azureuser/sombra-snapshots.tgz
ls -lh /home/azureuser/sombra-snapshots.tgz
```

## 2. Copiar y evaluar (en PowerShell de tu PC)

El comando te pide la IP pública de la VM (portal de Azure, máquina `MVtrading`,
campo "Dirección IP pública"). Tarda menos de un minuto.

```powershell
$ip = Read-Host "IP publica de la VM"
$datos = "$HOME\sombra-datos"
New-Item -ItemType Directory -Force $datos | Out-Null
scp -i $HOME\.ssh\azure_sombra "azureuser@${ip}:/home/azureuser/sombra-snapshots.tgz" "$datos\"
tar -xzf "$datos\sombra-snapshots.tgz" -C $datos
cd C:\Users\kevin\Desktop\Enginer\trading
git pull
$env:PYTHONPATH = "src"
uv run python scripts/evaluar_sombra.py $datos
```

Con la ventana abierta imprime algo así y no escribe nada:

```
Ventana: 2026-10-20 a 2026-11-18 (cierra 2026-11-18T00:10:30Z)
Dias transcurridos: 10 de 30; faltan 20
Diarios validos: 10  invalidos: 0  faltantes: 0
...
Margen: se pueden perder 5 diario(s) mas.
```

Si dice `YA ES NO EVALUABLE`, el veredicto ya está decidido aunque la ventana siga
abierta.

Con la ventana cerrada escribe en `results/sombra/`:

- `REPORTE.md`: veredicto, P1, P2 por tamaño y lado, costos por celda (diarios y de
  15 minutos), diarios inválidos con su motivo, y limitaciones.
- `summary.json`: las mismas cifras en JSON.
- `MANIFEST.json`: SHA256 de cada `snapshots-*.jsonl` leído y de las dos salidas.

Los snapshots completos no se commitean (CLAUDE.md, tamaño de PR): se suben a un
Release y el manifest ata el resumen a ellos.

```powershell
gh release create sombra-30d "$HOME\sombra-datos\sombra-snapshots.tgz" --title "Sombra: 30 dias de snapshots" --notes "Entrada de results/sombra (ADR 0007 seccion 6)"
```

## Qué decide y qué no

Deciden solo los snapshots de las 00:00:30 de los 30 días de la ventana. Los de 15
minutos se reportan con las mismas cifras y no deciden. Los Agentes A en sombra no
entran (ADR 0009 §3). Los días que la sombra siga juntando después de la ventana no
reabren el veredicto.
