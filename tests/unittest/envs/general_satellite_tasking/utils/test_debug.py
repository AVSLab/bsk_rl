from bsk_rl.envs.general_satellite_tasking.utils.debug import MEMORY_LEAK_CHECKING


def test_memory_leak_checking():
    # Ensure leak checking is off by default
    assert MEMORY_LEAK_CHECKING is False
