# NVIDIA CUDA 12 runtime + Python 3.12 for the tetration explorer GUI.
# Host needs an NVIDIA driver compatible with CUDA 12, nvidia-container-toolkit,
# and an X11 display (or equivalent) for Qt5.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Qt5Agg \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        libx11-xcb1 \
        libdbus-1-3 \
        libfontconfig1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt tetration_frac.py ./

RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

CMD ["python3", "tetration_frac.py"]
