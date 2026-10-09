#!/usr/bin/env bash
# Unpack only into a NEW checkout; never modifies /projects/dahu1128/bsk_rl.
set -euo pipefail
: "${BSK_PROJECT_ROOT:?Set the NEW completion project destination}"
BSK_RELEASE_DIR="${1:?Pass the directory containing base.bundle/source-overlay.tar.gz}"
BSK_RELEASE_DIR="$(cd "$BSK_RELEASE_DIR" && pwd)"
BSK_DEPLOY_PYTHON="${BSK_DEPLOY_PYTHON:-/projects/dahu1128/.venv-completion-v2/bin/python}"
test ! -e "$BSK_PROJECT_ROOT"
cd "$BSK_RELEASE_DIR"
sha256sum -c SHA256SUMS
git clone --no-checkout "$BSK_RELEASE_DIR/base.bundle" "$BSK_PROJECT_ROOT"
BSK_BASE_COMMIT="$("$BSK_DEPLOY_PYTHON" -c 'import json; print(json.load(open("release.json"))["base_commit"])')"
BSK_RELEASE_BRANCH="$("$BSK_DEPLOY_PYTHON" -c 'import json; print(json.load(open("release.json"))["branch"])')"
git -C "$BSK_PROJECT_ROOT" switch --detach "$BSK_BASE_COMMIT"
git -C "$BSK_PROJECT_ROOT" switch -c "$BSK_RELEASE_BRANCH"
tar -xzf "$BSK_RELEASE_DIR/source-overlay.tar.gz" -C "$BSK_PROJECT_ROOT"
cp "$BSK_RELEASE_DIR/release.json" "$BSK_PROJECT_ROOT/completion-release.json"
"$BSK_DEPLOY_PYTHON" - "$BSK_PROJECT_ROOT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = json.loads((root / 'completion-release.json').read_text())
for name, expected in record['source_sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Deployed source mismatch: {name}')
print('All deployed source hashes match the reviewed release.')
PY
