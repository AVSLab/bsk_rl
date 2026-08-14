#!/usr/bin/env bash
# This file is sourced by the Research Focus I Slurm jobs.

GCC_ROOT=${BSK_RL_GCC_ROOT:-/curc/sw/install/gcc/14.2.0}
GCC_BIN="$GCC_ROOT/bin/gcc"
VENV_ROOT=${BSK_RL_VENV_ROOT:-/projects/$USER/.venv}

if [[ ! -x "$GCC_BIN" ]]; then
    echo "Required Alpine GCC executable not found: $GCC_BIN" >&2
    exit 10
fi
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    echo "Required Python environment not found: $VENV_ROOT" >&2
    exit 11
fi

export PATH="$GCC_ROOT/bin:$PATH"
LIBSTDCXX_PATH=$($GCC_BIN -print-file-name=libstdc++.so.6)
if [[ ! -f "$LIBSTDCXX_PATH" ]]; then
    echo "GCC did not resolve libstdc++.so.6: $LIBSTDCXX_PATH" >&2
    exit 12
fi
if ! grep -a -q "GLIBCXX_3.4.29" "$LIBSTDCXX_PATH"; then
    echo "Compiler runtime lacks required GLIBCXX_3.4.29: $LIBSTDCXX_PATH" >&2
    exit 13
fi
export LD_LIBRARY_PATH="$(dirname "$LIBSTDCXX_PATH"):$GCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"

source "$VENV_ROOT/bin/activate"

echo "gcc_path=$(command -v gcc)"
echo "gcc_version=$(gcc -dumpfullversion -dumpversion)"
echo "libstdcxx_path=$LIBSTDCXX_PATH"
echo "python_path=$(command -v python)"
