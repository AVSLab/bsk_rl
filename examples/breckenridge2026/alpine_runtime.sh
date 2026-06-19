#!/bin/bash

# Configure the C++ runtime required by the Basilisk wheel on Alpine.

configure_breckenridge_alpine_runtime() {
    local venv_python=${1:?Pass the virtual-environment Python executable}

    if ! type module >/dev/null 2>&1; then
        echo "The Alpine module command is unavailable in this shell." >&2
        return 1
    fi

    # The historical GNC jobs used Alpine's default GCC module. Pinning a
    # version is brittle because CURC periodically changes available versions.
    module load gcc

    local gxx
    local libstdcpp
    gxx=$(command -v g++ || true)
    if [[ -z "$gxx" ]]; then
        echo "The loaded GCC module did not provide g++." >&2
        return 1
    fi

    libstdcpp=$("$gxx" -print-file-name=libstdc++.so.6)
    if [[ "$libstdcpp" == "libstdc++.so.6" || ! -f "$libstdcpp" ]]; then
        echo "Could not locate the GCC module's libstdc++.so.6." >&2
        return 1
    fi

    if ! strings "$libstdcpp" |
        awk '$0 == "GLIBCXX_3.4.29" { found = 1 } END { exit !found }'; then
        echo "$libstdcpp does not provide GLIBCXX_3.4.29." >&2
        return 1
    fi

    export LD_LIBRARY_PATH="$(dirname "$libstdcpp"):${LD_LIBRARY_PATH:-}"

    echo "gcc=$("$gxx" -dumpfullversion -dumpversion)"
    echo "libstdc++=$libstdcpp"
    "$venv_python" - <<'PY'
from Basilisk.architecture import sim_model

print(f"Basilisk runtime import succeeded: {sim_model.__file__}")
PY
}
