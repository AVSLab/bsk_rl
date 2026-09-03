#!/usr/bin/env bash

# Shared runtime setup for current Alpine AMOS 2026 jobs.  This mirrors the
# runtime used by the successful Research Focus I jobs and avoids relying on
# compiler module aliases that are unavailable on some Alpine login nodes.

gcc_root=${BSK_RL_GCC_ROOT:-/curc/sw/install/gcc/14.2.0}
gcc_bin="$gcc_root/bin/gcc"
venv_root=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}

if [[ ! -x "$gcc_bin" ]]; then
    echo "Required Alpine GCC executable not found: $gcc_bin" >&2
    exit 10
fi
if [[ ! -x "$venv_root/bin/python" ]]; then
    echo "Required Python environment not found: $venv_root" >&2
    exit 11
fi

export PATH="$gcc_root/bin:$PATH"
libstdcxx_path=$($gcc_bin -print-file-name=libstdc++.so.6)
if [[ ! -f "$libstdcxx_path" ]]; then
    echo "GCC did not resolve libstdc++.so.6: $libstdcxx_path" >&2
    exit 12
fi
if ! grep -aFq 'GLIBCXX_3.4.29' "$libstdcxx_path"; then
    echo "Compiler runtime lacks required GLIBCXX_3.4.29: $libstdcxx_path" >&2
    exit 13
fi
runtime_library_path="$(dirname "$libstdcxx_path"):$gcc_root/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$runtime_library_path"

# shellcheck source=/dev/null
source "$venv_root/bin/activate"

echo "gcc_path=$(command -v gcc)"
echo "gcc_version=$(gcc -dumpfullversion -dumpversion)"
echo "libstdcxx_path=$libstdcxx_path"
echo "python_path=$(command -v python)"
