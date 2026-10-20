"""One gate, existing samples only; no training/sampling in this orchestrator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "examples/multiagent_imaging/configs/walker4_mixed_amos_best_conflict.json"
CONFIG_SHA = "06f4b06a9be2c483db50cb109c52b2f53aac3e4fbcbaf67479a8d88949e4e478"


def input_hashes(inputs):
    files = [inputs / "updates.json"]
    files += [f for f in sorted((inputs / "update-journal").glob("*")) if f.is_file()]
    for name in ("checkpoint_0000", "checkpoint_0003", "learner_batch"):
        files += [f for f in sorted((inputs / name).rglob("*")) if f.is_file()]
    from examples.multiagent_imaging.update_evidence import file_sha256
    return {str(f.relative_to(inputs)): file_sha256(f) for f in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "verify"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "stage": args.stage, "started_at_unix": time.time()}
    try:
        if args.stage == "preflight":
            from examples.multiagent_imaging.cluster.audit_runtime import audit
            from examples.multiagent_imaging.cluster.effective_imports import effective_import_record
            runtime = audit(allocation=True)
            (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
            effective = effective_import_record()
            (args.output/"effective-imports-driver.json").write_text(json.dumps(effective,indent=2)+"\n")
            assert runtime["passed"], runtime["errors"]
            assert effective["passed"], effective["errors"]
            expected_native = json.loads((Path(__file__).with_name("gate-parent-runtime.manifest")).read_text())["loaded"]["native_sha256"]
            assert runtime["loaded"]["native_sha256"] == expected_native, "Native binaries changed since parent diagnostic"
            manifest = json.loads((ROOT/"completion-release.json").read_text())
            for name, expected in manifest["source_sha256"].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
            assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA
            checkpoint = args.input_root/"checkpoint_0003"
            saved = json.loads((checkpoint/"manifest.json").read_text())
            assert saved["training_iteration"] == 3 and saved["policy_decisions_lifetime"] == 14938
            assert saved["policy_sha256"] == "442840e6bf5b3effc5b9c7612eeefe5d51316ec3042976a57359ec45f12a39b2"
            assert saved["optimizer_sha256"] == "a65ab8ff1476f048891a6416ac9a08867582c71e52e26b1a819eac761c9c2f0f"
            from examples.multiagent_imaging.readiness import validate_schema
            from examples.multiagent_imaging.config import MultiAgentImagingConfig
            validate_schema(saved["schema"],MultiAgentImagingConfig.from_json(CONFIG))
            identity = json.loads((checkpoint/"wandb.json").read_text())
            assert identity["id"] == "7yrd5jn2"
            import wandb
            wandb.login(key=Path(os.environ["BSK_RL_WANDB_KEY_PATH"]).read_text().strip(),relogin=False)
            live = wandb.Api(timeout=60).run(identity["entity"]+"/"+identity["project"]+"/"+identity["id"])
            assert live.id == identity["id"]
            hashes = input_hashes(args.input_root)
            (args.output/"original-input-hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
            result.update(source_hashes_verified=len(manifest["source_sha256"]), original_input_files=len(hashes), online_wandb=True, wandb_id=live.id)
        else:
            before = json.loads((args.output/"original-input-hashes.json").read_text())
            assert input_hashes(args.input_root) == before, "Original checkpoint/batch bytes changed"
            resume = json.loads((args.output/"resume/zero-update-resume.json").read_text())
            fit = json.loads((args.output/"critic/fixed-batch-critic-fit.json").read_text())
            target = json.loads((args.output/"critic/fixed-batch-target-audit.json").read_text())
            assert resume["passed"] and resume["training_iteration"] == 3 and resume["policy_decisions_lifetime"] == 14938
            assert resume["new_ppo_updates"] == resume["new_sampled_episodes"] == 0
            assert resume["wandb_id"] == resume["wandb"]["id"] == "7yrd5jn2"
            assert target["passed"] and target["true_decisions"] == 4944 and target["target_max_abs_error"] == 0
            assert target["padded_rows"] == target["forced_no_action_count"] == 0
            assert target["acquisition_trace_present"] and target["physical_delivery_trace_present"]
            assert fit["can_fit"] and fit["actor_unchanged"]
            result.update(checkpoint_bytes_unchanged=True,input_files_verified=len(before),new_ppo_updates=0,new_sampled_episodes=0,critic_steps=fit["steps"])
        result["passed"] = True
    except BaseException as exc:
        result["error"] = str(exc)
        result["exception_type"] = type(exc).__name__
        raise
    finally:
        result["ended_at_unix"] = time.time()
        (args.output/(args.stage+"-gate.json")).write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k != "error"}),flush=True)

if __name__ == "__main__":
    main()
