#!/usr/bin/env bash
# ROCm Primus dev container (xiaoming-dev-fix). Run from host with GPU/IB.

set -euo pipefail

docker stop xiaoming-dev-fix
docker rm xiaoming-dev-fix

docker run -d \
  --name=xiaoming-dev-fix \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --cap-add=CAP_SYS_ADMIN \
  --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
  -v "${HOME}:${HOME}" \
  -v "${HOME}/Primus-dev:/workspace/rccl" \
  -w "${HOME}" \
  docker.io/rocm/primus:v26.2 sleep infinity
# docker.io/tasimage/primus:pr-563-ainic sleep infinity
# docker.io/rocm/primus:v26.1 sleep infinity
# docker.io/tasimage/primus:pr-463 sleep infinity
