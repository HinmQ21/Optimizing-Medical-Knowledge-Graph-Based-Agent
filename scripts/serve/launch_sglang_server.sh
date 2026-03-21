#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/serve/launch_sglang_server.sh --model-path PATH [options]

Options:
  --model-path PATH             Required model directory or HF id.
  --image NAME                  Default: lmsysorg/sglang:spark
  --container-name NAME         Default: sglang-server
  --port PORT                   Default: 30000
  --host HOST                   Default: 0.0.0.0
  --tp N                        Default: 1
  --dp N                        Default: 1
  --mem-fraction-static FLOAT   Default: 0.8
  --cuda-devices IDS            Example: 0 or 0,1. Passed to docker --gpus device=...
  --served-model-name NAME      Default: default
  --shm-size SIZE               Default: 16g
  --detach                      Launch container in background.
  --remove-existing             Stop and remove existing container with same name before launch.
  --                           Pass remaining args directly to sglang.launch_server.
EOF
}

model_path=""
image_name="lmsysorg/sglang:spark"
container_name="sglang-server"
port="30000"
host="0.0.0.0"
tp="1"
dp="1"
mem_fraction_static="0.7"
cuda_devices=""
served_model_name="default"
shm_size="16g"
detach_mode="0"
remove_existing="0"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      model_path="$2"
      shift 2
      ;;
    --image)
      image_name="$2"
      shift 2
      ;;
    --container-name)
      container_name="$2"
      shift 2
      ;;
    --port)
      port="$2"
      shift 2
      ;;
    --host)
      host="$2"
      shift 2
      ;;
    --tp)
      tp="$2"
      shift 2
      ;;
    --dp)
      dp="$2"
      shift 2
      ;;
    --mem-fraction-static)
      mem_fraction_static="$2"
      shift 2
      ;;
    --cuda-devices)
      cuda_devices="$2"
      shift 2
      ;;
    --served-model-name)
      served_model_name="$2"
      shift 2
      ;;
    --shm-size)
      shm_size="$2"
      shift 2
      ;;
    --detach)
      detach_mode="1"
      shift
      ;;
    --remove-existing)
      remove_existing="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$model_path" ]]; then
  echo "--model-path is required" >&2
  usage >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

host_model_path="$model_path"
container_model_path="$model_path"
docker_mount_args=()

if [[ -d "$model_path" || -f "$model_path" ]]; then
  host_model_path="$(realpath "$model_path")"
  container_model_path="$host_model_path"
  docker_mount_args=(-v "$host_model_path:$container_model_path:ro")
fi

docker_cmd=(
  docker run
  --name "$container_name"
  --shm-size "$shm_size"
  -p "${port}:${port}"
)

if [[ "$detach_mode" == "1" ]]; then
  docker_cmd+=(-d)
fi

if [[ -n "$cuda_devices" ]]; then
  docker_cmd+=(--gpus "device=${cuda_devices}")
else
  docker_cmd+=(--gpus all)
fi

if [[ ${#docker_mount_args[@]} -gt 0 ]]; then
  docker_cmd+=("${docker_mount_args[@]}")
fi

cmd=(
  python3 -m sglang.launch_server
  --model-path "$container_model_path"
  --host "$host"
  --port "$port"
  --served-model-name "$served_model_name"
  --mem-fraction-static "$mem_fraction_static"
  --dp "$dp"
  --tp "$tp"
)

if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

docker_cmd+=("$image_name")
docker_cmd+=("${cmd[@]}")

if [[ "$remove_existing" == "1" ]]; then
  docker rm -f "$container_name" >/dev/null 2>&1 || true
fi

echo "Launching SGLang server:"
printf '  %q' "${docker_cmd[@]}"
echo

if [[ "$detach_mode" == "1" ]]; then
  container_id="$("${docker_cmd[@]}")"
  echo "SGLang started in Docker container: ${container_id}"
  echo "Container name: $container_name"
  echo "Follow logs with: docker logs -f $container_name"
else
  exec "${docker_cmd[@]}"
fi
