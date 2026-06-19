#!/bin/bash

# Configure the C++ runtime required by the Basilisk wheel on Alpine.

configure_breckenridge_alpine_runtime() {
    local venv_python=${1:?Pass the virtual-environment Python executable}
    local gcc_root
    local libstdcpp
    local candidate
    local -a gcc_roots=()

    if [[ -n "${BRECK_GCC_ROOT:-}" ]]; then
        gcc_roots+=("$BRECK_GCC_ROOT")
    fi
    gcc_roots+=("/curc/sw/install/gcc/14.2.0")
    if [[ -d /curc/sw/install/gcc ]]; then
        while IFS= read -r candidate; do
            gcc_roots+=("$candidate")
        done < <(
            find /curc/sw/install/gcc -mindepth 1 -maxdepth 1 -type d -print |
                sort -Vr
        )
    fi

    gcc_root=""
    libstdcpp=""
    for candidate in "${gcc_roots[@]}"; do
        [[ -d "$candidate" ]] || continue
        libstdcpp=""
        if [[ -x "$candidate/bin/g++" ]]; then
            libstdcpp=$(
                "$candidate/bin/g++" -print-file-name=libstdc++.so.6 2>/dev/null ||
                    true
            )
        fi
        if [[ -z "$libstdcpp" || "$libstdcpp" == "libstdc++.so.6" || ! -f "$libstdcpp" ]]; then
            libstdcpp=$(find "$candidate" -type f -name 'libstdc++.so.6*' -print -quit)
        fi
        [[ -f "$libstdcpp" ]] || continue
        if strings "$libstdcpp" |
            awk '$0 == "GLIBCXX_3.4.29" { found = 1 } END { exit !found }'; then
            gcc_root="$candidate"
            break
        fi
        libstdcpp=""
    done

    if [[ -z "$gcc_root" || -z "$libstdcpp" ]]; then
        echo "No installed Alpine GCC runtime provides GLIBCXX_3.4.29." >&2
        echo "Checked roots: ${gcc_roots[*]}" >&2
        return 1
    fi

    export PATH="$gcc_root/bin:${PATH}"
    export LD_LIBRARY_PATH="$(dirname "$libstdcpp"):${LD_LIBRARY_PATH:-}"

    echo "gcc_root=$gcc_root"
    if [[ -x "$gcc_root/bin/g++" ]]; then
        echo "gcc=$("$gcc_root/bin/g++" -dumpfullversion -dumpversion)"
    fi
    echo "libstdc++=$libstdcpp"
    "$venv_python" - <<'PY'
from Basilisk.architecture import sim_model

print(f"Basilisk runtime import succeeded: {sim_model.__file__}")
PY
}
