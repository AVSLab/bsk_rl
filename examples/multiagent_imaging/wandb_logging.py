"""Optional W&B logging with explicit run identity, mode, and resume semantics."""

import json
import os
import threading
import time
import uuid
from examples.multiagent_imaging.study_metrics import LifetimeMetrics
from pathlib import Path


class TrainingLogger:
    def __init__(self, output, config, *, mode="disabled", resume=None, required=False):
        self.run = None
        self.driver_pid = os.getpid()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.heartbeat_thread = None
        self.background_error = None
        self.started_at = time.monotonic()
        self.iteration = 0
        self.lifetime = LifetimeMetrics()
        self.accounting_base = {"driver_wall_s": 0., "driver_allocated_cpu_h": 0.}
        if resume and (Path(resume)/"manifest.json").exists():
            saved_manifest = json.loads((Path(resume)/"manifest.json").read_text())
            saved_state = saved_manifest.get("lifetime_metrics")
            if saved_state: self.lifetime = LifetimeMetrics(saved_state)
            self.accounting_base.update(saved_manifest.get("training_accounting") or {})
        self.output = Path(output)
        if mode == "disabled":
            if required:
                raise ValueError("Required W&B logging cannot be disabled.")
            return
        if mode not in {"online", "offline"}:
            raise ValueError("W&B mode must be disabled, offline, or online.")
        try:
            import wandb

            if mode == "online" and os.environ.get("BSK_RL_WANDB_KEY_PATH"):
                key_path = Path(os.environ["BSK_RL_WANDB_KEY_PATH"]).expanduser()
                # The key never enters config, manifests, source bundles or logs.
                wandb.login(key=key_path.read_text().strip(), relogin=False)
            identity = {}
            if resume:
                saved = Path(resume) / "wandb.json"
                if saved.is_file():
                    identity = json.loads(saved.read_text())
            if identity and mode != "online":
                raise ValueError("Continuing the same W&B run requires online mode.")
            fresh_study = (not identity and config.get("mission",{}).get("information_case") in {"catalog_tx","geometry_only"})
            # Explicit fresh IDs override inherited WANDB_RUN_ID/BSK_* variables;
            # resume=never prevents an accidental historical run collision.
            requested_id = uuid.uuid4().hex[:8] if fresh_study else identity.get("id",os.environ.get("BSK_RL_WANDB_RUN_ID"))
            self.run = wandb.init(
                project=identity.get(
                    "project", os.environ.get("BSK_RL_WANDB_PROJECT", "multi-agent-space-imaging")
                ),
                entity=identity.get("entity", os.environ.get("BSK_RL_WANDB_ENTITY")),
                group=identity.get("group", os.environ.get(
                    "BSK_RL_WANDB_GROUP", "multi-agent-space-imaging-learning-recovery"
                )),
                name=identity.get("name", os.environ.get("BSK_RL_WANDB_NAME", self.output.name)),
                id=requested_id,
                resume="must" if identity else "never" if fresh_study else None,
                config=config,
                dir=str(self.output),
                mode=mode,
            )
            self.metadata = {
                "id": self.run.id,
                "project": self.run.project,
                "entity": self.run.entity,
                "url": self.run.url if mode == "online" else None,
                "mode": mode,
                "name": getattr(self.run,"name",os.environ.get("BSK_RL_WANDB_NAME",self.output.name)),
                "group": identity.get("group",os.environ.get("BSK_RL_WANDB_GROUP","multi-agent-space-imaging-learning-recovery")),
            }
            self.save_identity(self.output)
            if hasattr(wandb,"define_metric"):
                for axis in ("lifetime/true_decisions", "training/iteration", "episode/index", "status/elapsed_wall_h", "status/lifetime_driver_wall_h", "progress/sim_time_s"):
                    wandb.define_metric(axis)
                for prefix in ("mission/*", "agent/*", "lifetime/*", "losses/*", "learner_metrics/*", "update/*"):
                    wandb.define_metric(prefix, step_metric="lifetime/true_decisions")
                wandb.define_metric("status/*", step_metric="status/lifetime_driver_wall_h")
                wandb.define_metric("progress/*", step_metric="status/lifetime_driver_wall_h")
        except Exception:
            if required:
                raise
            raise RuntimeError(
                "Requested W&B initialization failed; choose disabled explicitly to run without logging."
            ) from None

    def save_identity(self, folder):
        if self.run is not None:
            (Path(folder) / "wandb.json").write_text(
                json.dumps(self.metadata, indent=2) + "\n"
            )

    def _log(self, metrics):
        if os.getpid() != self.driver_pid: raise RuntimeError("W&B publication is driver only")
        if self.run is not None:
            with self.lock: self.run.log(metrics)

    def ingest_episode(self, episode):
        metrics = episode.get("mission_metrics")
        if metrics is None: return False
        with self.lock:
            if not self.lifetime.add(episode["lifetime_episode_id"], metrics): return False
            from examples.multiagent_imaging.readiness import write_json
            write_json(self.output/"lifetime-state.json",self.lifetime.state)
            state=self.lifetime.state
            row = {"lifetime/true_decisions":state["true_decisions"],
                "lifetime/reward":state["reward"], "lifetime/complete_training_episodes":state["complete_training_episodes"],
                "lifetime/failed_horizon_episodes":state["failed_horizon_episodes"],
                "episode/id":episode["lifetime_episode_id"], "episode/index":state["complete_training_episodes"],
                "training/iteration":self.iteration, "mission/metric_version":metrics["version"],
                "mission/reached_horizon":int(metrics["reached_horizon"]),
                "mission/constellation_reward":metrics["constellation_reward"],
                "mission/useful_delivered_visits":metrics["useful_delivered_visits"],
                "mission/capture_coverage_fraction":metrics["capture_coverage_fraction"],
                "mission/ground_coverage_fraction":metrics["ground_coverage_fraction"],
                "mission/redundant_imaging_sensor_s":metrics["execution_redundancy"]["total_sensor_s"],
                "mission/redundant_execution_fraction":metrics["execution_redundancy"]["fraction_of_constellation_available_time"],
                "mission/available_live_sensor_s":metrics["execution_redundancy"]["available_live_sensor_s"],
                "mission/unaware_redundant_imaging_sensor_s":metrics["execution_redundancy"]["unaware_sensor_s"],
                "mission/pending_duplicate_acquisitions":metrics["pending_duplicates"]["excess_completed_acquisitions"],
                "mission/fresh_overlap_target_s":metrics["pending_duplicates"]["overlap_target_s"],
                "mission/excess_inventory_target_sensor_s":metrics["pending_duplicates"]["excess_inventory_target_sensor_s"]}
            for key,value in metrics["latency"].items():
                if not isinstance(value,list): row[f"mission/latency/{key}"]=value
            for key,value in (episode.get("cooperative_tasking") or {}).items():
                if isinstance(value,(int,float)):
                    row[f"mission/requests/{key}"] = value
            denom=state["available_live_sensor_s"]
            row["lifetime/redundant_execution_fraction"]=state["redundant_sensor_s"]/denom if denom else None
            row["lifetime/useful_delivered_visits"]=state["useful_delivered_visits"]
            row["lifetime/useful_delivery_delay_pooled_median_s"]=__import__('statistics').median(state["useful_product_delays_s"]) if state["useful_product_delays_s"] else None
            row["lifetime/useful_delivery_delay_valid_n"]=len(state["useful_product_delays_s"])
            for sensor,agent in metrics["agents"].items():
                for key,value in agent.items(): row[f"agent/{sensor}/{key}"]=value
                resources=episode["resources"].get(sensor)
                if resources:
                    row[f"agent/{sensor}/storage_fraction"]=resources[1]
                    row[f"agent/{sensor}/battery_fraction"]=resources[0]
                    row[f"agent/{sensor}/maximum_wheel_fraction"]=max(abs(v) for v in resources[2:])
                counts=episode["action_selection_counts"].get(sensor,{})
                for action in episode.get("action_labels",{}).get(sensor,counts): row[f"agent/{sensor}/action/{action}"]=counts.get(action,0)
            row["mission/physical_onboard_overlap_target_s"]=metrics["physical_onboard_overlap"]["overlap_target_s"]
            for outcome in ("accepted","lost","expired","receiver_unavailable"):
                row[f"mission/packets/{outcome}"]=episode["communication_summary"]["packet_outcome_counts"].get(outcome,0)
            row["mission/unique_acknowledged_versions"]=len(episode.get("transport_ack_revision_pairs",[]))
            self._log({k:v for k,v in row.items() if v is not None})
            return True

    def drain_spool(self):
        spool=os.environ.get("BSK_STUDY_EPISODE_SPOOL")
        if spool:
            for path in sorted(Path(spool).glob("*.json")):
                self.ingest_episode(json.loads(path.read_text()))

    def heartbeat(self, status="collecting_or_optimizing"):
        with self.lock:
            if os.getpid() != self.driver_pid:
                raise RuntimeError("Only the driver may log W&B metrics")
            progress={}
            import psutil
            from examples.multiagent_imaging.memory_profile import read_cgroup_memory
            process=psutil.Process(self.driver_pid)
            progress["memory/driver_rss_bytes"]=process.memory_info().rss
            try:
                progress["memory/driver_uss_bytes"]=getattr(process.memory_full_info(),"uss",None)
            except (psutil.AccessDenied,psutil.NoSuchProcess):
                pass
            if Path("/proc/self/cgroup").is_file():
                groups=read_cgroup_memory()
                bounded=[]
                for group in groups:
                    values=group["values"]
                    limit=values.get("memory.max",values.get("memory.limit_in_bytes","max"))
                    if limit.isdigit() and 0<int(limit)<2**60:
                        bounded.append((int(limit),values))
                if bounded:
                    limit,values=min(bounded,key=lambda item:item[0])
                    progress.update({"memory/job_cgroup_limit_bytes":limit,
                        "memory/job_cgroup_current_bytes":int(values.get("memory.current",values.get("memory.usage_in_bytes",0))),
                        "memory/job_cgroup_peak_bytes":int(values.get("memory.peak",values.get("memory.max_usage_in_bytes",0))),
                        "memory/job_cgroup_failcnt":int(values.get("memory.failcnt",0))})
            progress={k:v for k,v in progress.items() if v is not None}
            spool=os.environ.get("BSK_STUDY_EPISODE_SPOOL")
            if spool:
                for path in sorted((Path(spool)/"progress").glob("*.json")):
                    raw=json.loads(path.read_text())
                    progress.update({"progress/sim_time_s":raw["sim_time_s"],"progress/seed":raw["seed"],
                        "progress/episode_index":raw["next_episode_index"],
                        "progress/constellation_reward":sum(raw["reward_by_agent"].values())})
                    for sensor,agent in raw["agents"].items():
                        for key,value in agent.items():
                            if key=="actions":
                                progress.update({f"progress/{sensor}/action/{action}":count for action,count in value.items()})
                            else: progress[f"progress/{sensor}/{key}"]=value
            self._log({"status/elapsed_wall_h":(time.monotonic()-self.started_at)/3600,
                "status/state":status, "status/allocated_cpu_h":float(os.environ.get("BSK_ALLOCATED_CPUS","0"))*(time.monotonic()-self.started_at)/3600,
                "status/lifetime_driver_wall_h":self.accounting_state()["driver_wall_s"]/3600,
                "status/lifetime_driver_allocated_cpu_h":self.accounting_state()["driver_allocated_cpu_h"],
                "lifetime/true_decisions":self.lifetime.state["true_decisions"], "training/iteration":self.iteration, **progress})

    def accounting_state(self):
        """Resume-preserved driver cost, distinct from the allocation's final sacct.

        Readbacks do not accumulate elapsed time twice. Preflight, frozen
        evaluation and other driver processes are covered by job-exit/sacct.
        """
        elapsed = time.monotonic()-self.started_at
        return {"driver_wall_s":self.accounting_base["driver_wall_s"]+elapsed,
            "driver_allocated_cpu_h":self.accounting_base["driver_allocated_cpu_h"]+float(os.environ.get("BSK_ALLOCATED_CPUS","0"))*elapsed/3600}

    def start_heartbeat(self, interval_s=30):
        def work():
            while not self.stop_event.wait(interval_s):
                try: self.drain_spool(); self.heartbeat()
                except BaseException as exc: self.background_error=exc; return
        self.heartbeat_thread=threading.Thread(target=work,daemon=True);self.heartbeat_thread.start()

    def check_background(self):
        if self.background_error: raise RuntimeError("Driver logging/heartbeat failed") from self.background_error

    def checkpoint_state(self):
        self.drain_spool(); self.check_background()
        with self.lock: return json.loads(json.dumps(self.lifetime.state))

    def log_update(self, record):
        self.iteration=record["iteration"]
        for episode in record["episodes"]: self.ingest_episode(episode)
        self.check_background()
        if self.run is None:
            return
        metrics = {"lifetime/true_decisions":record["policy_decisions_lifetime"],
            "update/complete_episodes":record["complete_episode_count"],
            "training/iteration": record["iteration"],
            "training/policy_decisions_total": record["policy_decisions_lifetime"],
            "training/episodes": record["complete_episode_count"],
            "update/new_sampled_true_decisions": record.get('actual_policy_decisions',0),
            "update/condensed_learner_rows": record.get('condensed_learner_rows',0),
            "update/repeated_learner_presentations": record.get('repeated_learner_presentations',0),
            "training/gradient_l2": record["gradient_l2"],
            "training/parameter_delta_l2": record["parameter_change"]["l2"],
        }
        for key in ("actor_l2", "critic_l2"):
            if key in record["parameter_change"]:
                metrics[f"training/parameter_delta_{key}"] = record["parameter_change"][
                    key
                ]
        for section in ("losses", "learner_metrics", "measurement"):
            for key, value in record[section].items():
                if isinstance(value, (int, float)):
                    metrics[f"{section}/{key}"] = value
        episodes = record["episodes"]
        if not episodes:
            self._log(metrics);return
        mission_fields = {
            "constellation_reward": ("constellation_reward",),
            "capture_coverage_fraction": ("coverage", "capture_coverage_fraction"),
            "ground_delivery_coverage_fraction": (
                "coverage",
                "ground_delivery_coverage_fraction",
            ),
            "useful_acquisitions": ("team_summary", "unique_acquisition_count"),
            "useful_delivered_visits": ("team_summary", "unique_service_count"),
            "duplicate_attempts": ("team_summary", "duplicate_attempt_count"),
            "useful_post_cooldown_revisits": ("useful_post_cooldown_revisits", "count"),
            "radio_occupancy_s": ("communication_summary", "radio_occupancy_s"),
        }
        for key, path in mission_fields.items():
            values = []
            for episode in episodes:
                value = episode
                for part in path:
                    value = value.get(part) if isinstance(value, dict) else None
                if isinstance(value, (int, float)):
                    values.append(value)
            if values:
                metrics[f"episode/{key}_mean"] = sum(values) / len(values)
                metrics[f"episode/{key}_valid_n"] = len(values)
        for outcome in sorted(
            {
                k
                for e in episodes
                for k in e.get("communication_summary", {}).get(
                    "packet_outcome_counts", {}
                )
            }
        ):
            metrics[f"packets/{outcome}_mean"] = sum(
                e.get("communication_summary", {})
                .get("packet_outcome_counts", {})
                .get(outcome, 0)
                for e in episodes
            ) / len(episodes)
        metrics["communication/acknowledged_versions_mean"] = sum(
            sum(len(v) for v in e.get("acknowledged_versions_by_link", {}).values())
            for e in episodes
        ) / len(episodes)
        sensors = sorted(
            {sensor for e in episodes for sensor in e["action_selection_counts"]}
        )
        for sensor in sensors:
            labels = sorted(
                {
                    label
                    for e in episodes
                    for label in e["action_selection_counts"].get(sensor, {})
                }
            )
            for label in labels:
                metrics[f"actions/{sensor}/{label}_mean"] = sum(
                    e["action_selection_counts"].get(sensor, {}).get(label, 0)
                    for e in episodes
                ) / len(episodes)
        self._log(metrics)

    def finish(self, failed=False):
        self.stop_event.set()
        if self.heartbeat_thread: self.heartbeat_thread.join(timeout=5)
        if self.run is not None:
            self.run.finish(exit_code=1 if failed else 0)
