import ast
import os
import pkgutil
import sys
from unittest.mock import MagicMock

"""
Copy this file to the site-packages directory of your virtual environment to mock
Basilisk for generating documentation without Basilisk installed
"""

os.environ["PYTHON_MOCK_BASILISK"] = "1"

bsk_rl_package_path = (
    pkgutil.get_loader("bsk_rl").get_filename().split("__init__.py")[0]
)
print(bsk_rl_package_path)

# Find all imports from Basilisk within bsk_rl package
all_basilisk_imports = set()
for root, _, files in os.walk(bsk_rl_package_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                try:
                    tree = ast.parse(f.read(), filename=file)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name.startswith("Basilisk"):
                                    all_basilisk_imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.module.startswith("Basilisk"):
                                all_basilisk_imports.add(node.module)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

# Mock those imports
for submodule_name in all_basilisk_imports:
    sys.modules[submodule_name] = MagicMock()

# Mock some other imports that might cause issues
sys.modules["Basilisk"].__path__ = "not/a/real/path"
