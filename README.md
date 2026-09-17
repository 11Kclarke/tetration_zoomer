# Infinite tetration zoomer

GPU explorer for the power-tower fractal: for each pixel base `c`, iterate
`z ← exp(z log c)` from `z = 1` and color by escape time.

## Requirements

- NVIDIA GPU with a CUDA 12-capable driver
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker
- An X11 display (the Qt5 matplotlib window is forwarded with `DISPLAY`)

## Run

```bash
chmod +x run.sh
./run.sh
```

That builds `tetration-zoomer` if needed and starts the GUI. First CuPy kernel
compile is slower than later frames.

Without Docker, from a venv matching `requirements.txt`:

```bash
MPLBACKEND=Qt5Agg python tetration_frac.py
```

`cupy-cuda12x[ctk]` pulls the CUDA 12 runtime libraries into the venv (same
extra used locally). Use `cupy-cuda13x` only if your driver/toolkit is CUDA 13.

## Controls

- Matplotlib toolbar: pan, zoom, home, back, forward
- After navigation, the last image is stretched until a full-res GPU recompute
  finishes (~125 ms idle)
- **Smooth / Iteration**: histogram coloring of continuous vs integer escape
- **Colormap**: cycle `turbo`, `magma`, `cividis`, `cubehelix`, `twilight_shifted`,
  `nipy_spectral`
- Hover: integer escape iteration (or `recomputing…` while the view is stale)

Default render is 4096×4096, 120 iterations. Change those in `TetrationExplorer`
if you need a lighter first frame.
