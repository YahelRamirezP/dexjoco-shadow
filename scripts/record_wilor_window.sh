#!/usr/bin/env bash
# Records ONLY the "GraphGrasp - WiLoR" camera preview window (not the full
# screen), by finding its exact geometry via xdotool and feeding that into
# ffmpeg's x11grab. No manual xwininfo click needed.
#
# Usage: ./record_wilor_window.sh /path/to/output_dir
#   (waits for the WiLoR window to appear, then records until Ctrl+C)
set -euo pipefail

OUT_DIR="${1:?Usage: $0 /path/to/output_dir}"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
OUT_FILE="${OUT_DIR}/camara_operador_${TS}.mp4"

WIN_TITLE="GraphGrasp - WiLoR"
echo "Esperando ventana '${WIN_TITLE}'..."
WIN_ID=""
for i in $(seq 1 60); do
    WIN_ID=$(xdotool search --name "$WIN_TITLE" | head -1 || true)
    if [ -n "$WIN_ID" ]; then
        break
    fi
    sleep 1
done
if [ -z "$WIN_ID" ]; then
    echo "No aparecio la ventana '${WIN_TITLE}' en 60s -- corre live_retarget.py primero." >&2
    exit 1
fi

# xdotool geometry is relative to the window's own top-left; getwindowgeometry
# --shell gives X,Y,WIDTH,HEIGHT of the window in absolute screen coords.
eval "$(xdotool getwindowgeometry --shell "$WIN_ID")"
echo "Ventana encontrada: ${WIDTH}x${HEIGHT} en (${X},${Y})"
echo "Grabando -> ${OUT_FILE}  (Ctrl+C para terminar)"

ffmpeg -f x11grab -video_size "${WIDTH}x${HEIGHT}" -i ":0.0+${X},${Y}" \
    -framerate 30 -c:v libx264 -crf 15 -preset slow "$OUT_FILE"
