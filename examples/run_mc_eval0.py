#!/usr/bin/env python3
import os
import sys
import subprocess

# =========================
# USER CONFIG (edit here)
# =========================
EVAL_SCRIPT_NAME = "policy_evaluation_2026.py"  # your copied evaluation script filename

SEED_START = 0          # starting seed for this batch
N_RUNS = 10               # number of sequential seeds to run
QUIET = True            # pass --quiet to reduce prints (only if your eval script supports it)
SAVE_DATA = True        # pass --save_data / --no_save_data (only if your eval script supports it)
TARGET_ENV = "mixed"      # "leo" or "mixed"
MIX_WEIGHTS = {"LEO": 0.5, "MEO": 0.3, "GEO": 0.2}


# Optional: throttle parallel launches from multiple terminals (not needed)
# SLEEP_BETWEEN_RUNS_SEC = 0
# =========================


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(here, EVAL_SCRIPT_NAME)

    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    failures = []
    for k in range(N_RUNS):
        seed = SEED_START + k

        cmd = [sys.executable, eval_script, "--seed", str(seed)]
        cmd += ["--target_env", TARGET_ENV]
        if TARGET_ENV == "mixed":
            import json
            cmd += ["--mix_weights", json.dumps(MIX_WEIGHTS)]


        if QUIET:
            cmd.append("--quiet")

        if SAVE_DATA:
            cmd.append("--save_data")
        else:
            cmd.append("--no_save_data")

        print(f"\n=== Running seed {seed} ===")
        ret = subprocess.run(cmd, cwd=here).returncode  # cwd=examples ensures relative paths behave
        if ret != 0:
            failures.append((seed, ret))
            print(f"!! seed {seed} failed with code {ret}")

        # if SLEEP_BETWEEN_RUNS_SEC:
        #     import time
        #     time.sleep(SLEEP_BETWEEN_RUNS_SEC)

    print("\nDone.")
    if failures:
        print("Failures:", failures)


if __name__ == "__main__":
    main()
