#!/usr/bin/env bash
# Safe local Jupyter Lab launcher for repository development.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_project_root="$(cd -- "${script_dir}/.." && pwd)"
project_root="${RVE_JUPYTER_ROOT:-${script_project_root}}"
host="${RVE_JUPYTER_HOST:-127.0.0.1}"
port="${RVE_JUPYTER_PORT:-8888}"

if [[ "${project_root}" != /* ]]; then
    echo "RVE_JUPYTER_ROOT must be an absolute path" >&2
    exit 2
fi
if ! [[ "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "RVE_JUPYTER_PORT must be an integer from 1 through 65535" >&2
    exit 2
fi
if ! command -v jupyter >/dev/null 2>&1; then
    echo "jupyter is not installed in the active environment" >&2
    exit 1
fi

mkdir -p "${project_root}/notebooks"
cd "${project_root}"

echo "Rovibrational Excitation Jupyter Lab"
echo "Root: ${project_root}"
echo "URL: http://${host}:${port}"
echo "Jupyter authentication and XSRF protection remain enabled."
if [[ "${host}" != "127.0.0.1" && "${host}" != "localhost" ]]; then
    echo "WARNING: Jupyter is binding beyond localhost; keep authentication enabled." >&2
fi

args=(
    lab
    "--ip=${host}"
    "--port=${port}"
    --no-browser
    "--ServerApp.root_dir=${project_root}"
)
if [[ "$(id -u)" -eq 0 ]]; then
    args+=(--allow-root)
fi

exec jupyter "${args[@]}" "$@"
