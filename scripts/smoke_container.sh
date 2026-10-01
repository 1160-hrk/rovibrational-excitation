#!/usr/bin/env bash
# Build and exercise the development container on a real Docker daemon.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
image_tag="rve-devcontainer-smoke:${GITHUB_SHA:-local}-$$"
container_name="rve-container-smoke-$$"
temporary_root="$(mktemp -d /tmp/rve-container-smoke.XXXXXX)"
jupyter_token="rve-container-smoke-token"
stage="initialization"

cleanup() {
    status=$?
    trap - EXIT
    if command -v docker >/dev/null 2>&1 &&
        docker inspect "${container_name}" >/dev/null 2>&1; then
        if [[ "${status}" -ne 0 ]]; then
            docker logs "${container_name}" || true
        fi
        docker rm -f "${container_name}" >/dev/null || true
    fi
    if command -v docker >/dev/null 2>&1; then
        docker image rm "${image_tag}" >/dev/null 2>&1 || true
    fi
    rm -rf -- "${temporary_root}"
    if [[ "${status}" -ne 0 ]]; then
        echo "::error title=Container smoke failed::stage=${stage}"
    fi
    exit "${status}"
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
    echo "container smoke requires the Docker CLI" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "container smoke requires a working Docker daemon" >&2
    exit 1
fi

mkdir -p "${temporary_root}/notebooks"
stage="image build"
docker build --pull --tag "${image_tag}" "${project_root}"

image_check="test \"\$(id -u)\" -ne 0"
image_check+=" && test \"\$(id -un)\" = devuser"
image_check+=" && test \"\$(pwd)\" = /workspace"
image_check+=" && cd /tmp"
image_check+=" && python -c \"import rovibrational_excitation as rve; assert rve.__version__\""
image_check+=" && command -v jupyter >/dev/null"
stage="unmounted image validation"
docker run --rm --entrypoint /bin/sh "${image_tag}" -lc "${image_check}"

container_args=(
    --detach
    --name "${container_name}"
    --publish 127.0.0.1::8888
    --env RVE_JUPYTER_HOST=0.0.0.0
    --mount "type=bind,src=${project_root},dst=/workspace,readonly"
    --mount "type=bind,src=${temporary_root}/notebooks,dst=/workspace/notebooks"
    --entrypoint /workspace/scripts/start_jupyter.sh
)
stage="Jupyter launch"
docker run "${container_args[@]}" "${image_tag}" \
    "--IdentityProvider.token=${jupyter_token}" >/dev/null

stage="Jupyter port discovery"
port_mapping="$(docker port "${container_name}" 8888/tcp)"
host_port="${port_mapping##*:}"
if ! [[ "${host_port}" =~ ^[0-9]+$ ]]; then
    echo "container smoke could not resolve the published Jupyter port" >&2
    exit 1
fi

api_url="http://127.0.0.1:${host_port}/api/contents"
stage="authenticated Jupyter readiness"
ready=false
for ((_attempt = 1; _attempt <= 30; _attempt++)); do
    if curl --silent --fail \
        --header "Authorization: token ${jupyter_token}" \
        "${api_url}" >"${temporary_root}/jupyter-api.json"; then
        ready=true
        break
    fi
    running="$(docker inspect --format "{{.State.Running}}" "${container_name}")"
    if [[ "${running}" != "true" ]]; then
        break
    fi
    sleep 1
done
if [[ "${ready}" != "true" ]]; then
    echo "authenticated Jupyter API did not become ready" >&2
    exit 1
fi

stage="Jupyter authentication enforcement"
unauthenticated_http="$(
    curl --silent \
        --output /dev/null \
        --write-out "%{http_code}" \
        "${api_url}" || true
)"
if [[ "${unauthenticated_http}" == "200" ]]; then
    echo "unauthenticated HTTP request unexpectedly succeeded" >&2
    exit 1
fi
stage="runtime user validation"
if [[ "$(docker exec "${container_name}" id -u)" -eq 0 ]]; then
    echo "Jupyter container unexpectedly runs as root" >&2
    exit 1
fi

echo "Container smoke passed: non-root package import and authenticated Jupyter API"
