#!/usr/bin/env bash
# Build (if needed) and run the tetration explorer with GPU + X11.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-tetration-zoomer}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; this image needs an X11 session" >&2
  exit 1
fi

xhost +local:docker >/dev/null 2>&1 || true

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

docker run --rm \
  --gpus all \
  -e DISPLAY="${DISPLAY}" \
  -e MPLBACKEND=Qt5Agg \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --network host \
  "${IMAGE_NAME}"
