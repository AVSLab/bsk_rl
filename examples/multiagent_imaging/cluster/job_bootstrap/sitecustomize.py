"""Per-job preload; inactive outside the explicitly opted-in repair launcher."""
import os
if os.environ.get("BSK_REQUIRE_EXTERNAL_PSUTIL") == "1":
    import psutil
    if psutil.__version__ != "6.1.0" or "/ray/thirdparty_files/" in psutil.__file__.replace("\\", "/"):
        raise SystemExit("Required external psutil 6.1.0 was not loaded by per-job bootstrap")
