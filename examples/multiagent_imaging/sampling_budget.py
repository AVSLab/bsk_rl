"""Conservative job-wide sampling caps, partitioned across synchronous runners.

No policy observation, reward or simulation state is involved. Per-worker quotas
sum to the declared job cap. An exhausted quota fails truthfully; it never drops
unfinished rows or presents a partial mission as a completed episode.
"""


def worker_quota(total, worker_index, num_env_runners):
    if total is None:
        return None
    total, worker_index, num_env_runners = map(int, (total, worker_index, num_env_runners))
    if total < 0 or num_env_runners < 0:
        raise ValueError("Nonnegative total and runner count required")
    if num_env_runners == 0:
        if worker_index != 0:
            raise ValueError("Local sampling requires worker zero")
        return total
    if worker_index == 0:
        return None  # Local driver runner does not sample with remote runners.
    if not 1 <= worker_index <= num_env_runners:
        raise ValueError("Worker outside the declared topology")
    quotient, remainder = divmod(total, num_env_runners)
    return quotient + int(worker_index <= remainder)


def budget_manifest(episodes, decisions, num_env_runners):
    workers = range(1, num_env_runners + 1) if num_env_runners else (0,)
    return dict(
        version="partitioned-job-caps-v1",
        global_episode_cap=episodes,
        global_true_decision_cap=decisions,
        num_env_runners=num_env_runners,
        quotas={str(i): dict(episodes=worker_quota(episodes, i, num_env_runners),
                            true_decisions=worker_quota(decisions, i, num_env_runners))
                for i in workers},
        semantics="New sampling in this invocation; quotas sum to job cap. Exhaustion is a retained failure, not silent redistribution or partial success.",
    )


def configure_environment(episodes, decisions, num_env_runners):
    """Replace invocation-scoped caps; an uncapped restore inherits no old cap."""
    import os
    os.environ['BSK_STUDY_NUM_ENV_RUNNERS'] = str(num_env_runners)
    for name, value in [('BSK_STUDY_MAX_EPISODES', episodes),
                        ('BSK_STUDY_MAX_TRUE_DECISIONS', decisions)]:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    return budget_manifest(episodes, decisions, num_env_runners)
