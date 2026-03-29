#!/usr/bin/env bash
set -e

xhost +local:root

docker run -it \
  -e DISPLAY="${DISPLAY}" \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --mount type=bind,src="${HOME}/.Xauthority",dst=/tmp/.docker.xauth,readonly \
  --privileged \
  --net=host \
  --ipc=host \
  --cap-add=SYS_NICE \
  --name fourier_aurora_server \
  fourier_aurora_sdk_gr2:v1.3.0 bash
