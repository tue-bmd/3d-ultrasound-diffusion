"""Default locations of data, models and results.

Every default can be overridden with an environment variable, which can also be put in a
``.env`` file in the repository root (see ``.env.example``). Command line flags of the
individual scripts take precedence over both.

- ``US3D_DATA_DIR``: dataset root (local folder or ``hf://<repo_id>``) with ``train/`` and ``test/``.
- ``US3D_RESULTS_DIR``: where inference / evaluation results are written.
- ``US3D_DIFFUSION_MODEL``: local folder or ``hf://<repo>`` of the diffusion model
  (default: where ``train.py`` saves it).
- ``US3D_SUPERVISED_MODEL``: local folder or ``hf://<repo>`` of an optional supervised U-Net.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path = REPO_ROOT / ".env"):
    """Minimal ``.env`` loader: ``KEY=value`` lines; existing env vars are not overwritten."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# Strings rather than Paths, so ``hf://<repo_id>`` datasets work as well.
DATA_DIR = os.environ.get("US3D_DATA_DIR", str(REPO_ROOT / "data")).rstrip("/")
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
RESULTS_DIR = Path(os.environ.get("US3D_RESULTS_DIR", REPO_ROOT / "results"))

# train.py saves the diffusion model here, inference.py loads it from here by default
TRAINED_DIFFUSION_DIR = RESULTS_DIR / "models" / "diffusion"
DIFFUSION_MODEL = os.environ.get("US3D_DIFFUSION_MODEL", str(TRAINED_DIFFUSION_DIR))
SUPERVISED_MODEL = os.environ.get("US3D_SUPERVISED_MODEL")  # optional U-Net baseline
