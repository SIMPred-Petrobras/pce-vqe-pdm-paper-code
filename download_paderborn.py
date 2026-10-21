"""Download the Paderborn University bearing dataset (KAt-DataCenter) into ``PADERBORN/``.

Thin wrapper around :mod:`pdm.download` (``src/pdm/download.py``), which
documents the dataset and the options. Examples (from the repository root)::

    python download_paderborn.py                          # all 32 bearings (5.4 GB)
    python download_paderborn.py --groups healthy real    # healthy + real damage
    python download_paderborn.py --bearings K001 KA01 --extract
    python download_paderborn.py --dry-run                # list what would be downloaded

The experiment (``python src/main.py``) reads the measurements straight from
the archives, so ``--extract`` is optional.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pdm.download import main  # noqa: E402

if __name__ == "__main__":
    main(destination=ROOT / "PADERBORN")
