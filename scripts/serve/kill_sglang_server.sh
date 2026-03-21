#!/usr/bin/env bash
set -euo pipefail

container_name="${1:-sglang-server}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

docker stop "$container_name" >/dev/null 2>&1 || true
docker rm "$container_name" >/dev/null 2>&1 || true
echo "Stopped and removed container: $container_name"
