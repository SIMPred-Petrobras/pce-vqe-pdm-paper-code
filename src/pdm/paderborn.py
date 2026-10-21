"""The Paderborn University bearing dataset (KAt-DataCenter): bearings, files and windows.

The dataset (Lessmeier et al., 2016) has 32 ball bearings of type 6203, each
stored as one RAR archive ``<code>.rar`` with 20 measurements of 4 s for each
of four operating conditions, ``<code>/<condition>_<code>_<n>.mat``
(``n = 1..20``). Every measurement is one MATLAB struct whose field ``Y``
lists the channels; this study uses two of them:

* ``vibration_1`` -- acceleration of the bearing housing, 64 kHz, 256 001-256 004 samples;
* ``speed`` -- shaft speed in rpm, 4 kHz, 16 001 samples (nominally 1500 rpm).

This module holds the bearing table (damage, origin, manufacturer and run-in time,
from Tables 4, 5 and 7 of Lessmeier et al. and from the fact sheet in each
archive), reads one measurement straight from its archive, and cuts it into
non-overlapping windows. The cross-validation protocol built on top of it is
in :mod:`pdm.protocol`.

Reference: C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition
Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor
Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven
Classification", PHM Society European Conference, 2016. The data are licensed
CC BY-NC 4.0 (KAt-DataCenter, Chair of Design and Drive Technology, Paderborn
University).
"""

from __future__ import annotations

import io
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .download import extract, member_bytes, rar_extractor

#: Sampling rate of ``vibration_1`` (Hz).
SAMPLING_RATE = 64_000
#: Sampling rate of ``speed`` (Hz).
SPEED_RATE = 4_000
#: Measurements per bearing and operating condition.
MEASUREMENTS = tuple(range(1, 21))

#: Operating conditions: rotational speed (rpm), load torque (Nm) and radial force (N).
CONDITIONS = {
    "N15_M07_F10": (1500, 0.7, 1000),
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}
#: Condition of the experiment (the basic setup of the dataset).
MAIN_CONDITION = "N15_M07_F10"


@dataclass(frozen=True)
class Bearing:
    """One test bearing of the dataset.

    Attributes:
        code: Bearing code (``K001``, ``KA01``, ...).
        group: ``"healthy"``, ``"artificial"`` (damage made by hand) or
            ``"real"`` (damage from accelerated lifetime tests).
        location: Damaged component: ``"none"``, ``"OR"`` (outer ring),
            ``"IR"`` (inner ring) or ``"IR+OR"`` (both; the ``KB`` bearings).
        method: How the damage was made or what it is: ``"EDM"``,
            ``"engraver"``, ``"drilling"`` (artificial), ``"pitting"`` or
            ``"indentations"`` (real), ``"none"`` (healthy).
        extent: Extent of the damage (level 1-3; 0 for healthy bearings).
        manufacturer: ``"IBU"``, ``"MTK"`` or ``"FAG"``.
        run_in_hours: Run-in time of the healthy bearings (``None`` otherwise;
            K001 ran for more than 50 h, stored as 50).
    """

    code: str
    group: str
    location: str
    method: str
    extent: int
    manufacturer: str
    run_in_hours: float | None = None

    @property
    def faulty(self) -> bool:
        """Whether the bearing is damaged."""
        return self.group != "healthy"


_IBU, _MTK, _FAG = "IBU", "MTK", "FAG"

#: The 32 bearings. Damage: Tables 4 and 5 of Lessmeier et al. (2016); run-in
#: times: Table 7; manufacturer: fact sheet in each archive.
BEARINGS: dict[str, Bearing] = {b.code: b for b in [
    # Healthy (all IBU).
    Bearing("K001", "healthy", "none", "none", 0, _IBU, 50.0),
    Bearing("K002", "healthy", "none", "none", 0, _IBU, 19.0),
    Bearing("K003", "healthy", "none", "none", 0, _IBU, 1.0),
    Bearing("K004", "healthy", "none", "none", 0, _IBU, 5.0),
    Bearing("K005", "healthy", "none", "none", 0, _IBU, 10.0),
    Bearing("K006", "healthy", "none", "none", 0, _IBU, 16.0),
    # Artificial damage.
    Bearing("KA01", "artificial", "OR", "EDM", 1, _MTK),
    Bearing("KA03", "artificial", "OR", "engraver", 2, _MTK),
    Bearing("KA05", "artificial", "OR", "engraver", 1, _IBU),
    Bearing("KA06", "artificial", "OR", "engraver", 2, _IBU),
    Bearing("KA07", "artificial", "OR", "drilling", 1, _IBU),
    Bearing("KA08", "artificial", "OR", "drilling", 2, _IBU),
    Bearing("KA09", "artificial", "OR", "drilling", 2, _IBU),
    Bearing("KI01", "artificial", "IR", "EDM", 1, _MTK),
    Bearing("KI03", "artificial", "IR", "engraver", 1, _MTK),
    Bearing("KI05", "artificial", "IR", "engraver", 1, _IBU),
    Bearing("KI07", "artificial", "IR", "engraver", 2, _IBU),
    Bearing("KI08", "artificial", "IR", "engraver", 2, _IBU),
    # Real damage from accelerated lifetime tests.
    Bearing("KA04", "real", "OR", "pitting", 1, _FAG),
    Bearing("KA15", "real", "OR", "indentations", 1, _FAG),
    Bearing("KA16", "real", "OR", "pitting", 2, _MTK),
    Bearing("KA22", "real", "OR", "pitting", 1, _IBU),
    Bearing("KA30", "real", "OR", "indentations", 1, _MTK),
    Bearing("KB23", "real", "IR+OR", "pitting", 2, _MTK),
    Bearing("KB24", "real", "IR+OR", "pitting", 3, _FAG),
    Bearing("KB27", "real", "IR+OR", "indentations", 1, _FAG),
    Bearing("KI04", "real", "IR", "pitting", 1, _MTK),
    Bearing("KI14", "real", "IR", "pitting", 1, _MTK),
    Bearing("KI16", "real", "IR", "pitting", 3, _FAG),
    Bearing("KI17", "real", "IR", "pitting", 1, _MTK),
    Bearing("KI18", "real", "IR", "pitting", 2, _MTK),
    Bearing("KI21", "real", "IR", "pitting", 1, _FAG),
]}


def codes(group: str | None = None, manufacturer: str | None = None) -> list[str]:
    """Bearing codes, optionally filtered by group and manufacturer, in table order."""
    return [c for c, b in BEARINGS.items()
            if (group is None or b.group == group) and (manufacturer is None or b.manufacturer == manufacturer)]


# --------------------------------------------------------------------------
# Reading the measurements
# --------------------------------------------------------------------------

def member_name(code: str, condition: str, measurement: int) -> str:
    """Path of one measurement inside the archive of its bearing."""
    return f"{code}/{condition}_{code}_{measurement}.mat"


def read_mat(source: str | Path | bytes) -> tuple[np.ndarray, np.ndarray]:
    """Vibration (64 kHz) and shaft speed (rpm, 4 kHz) of one measurement.

    The file holds one struct (named after the file); its field ``Y`` is an
    array of channels with fields ``Name``, ``Raster`` and ``Data``.

    Args:
        source: Path of the ``.mat`` file, or its content.
    """
    mat = loadmat(io.BytesIO(source) if isinstance(source, bytes) else source,
                  squeeze_me=True, struct_as_record=False)
    key = next(k for k in mat if not k.startswith("__"))
    channels = {str(ch.Name): np.asarray(ch.Data, dtype=float).ravel() for ch in np.atleast_1d(mat[key].Y)}
    return channels["vibration_1"], channels["speed"]


class MeasurementReader:
    """Read measurements from extracted files or straight from the RAR archives.

    A measurement already extracted to ``raw_dir/<code>/`` (for example by
    ``download_paderborn.py --extract``) is read from there. Otherwise it is
    streamed from its archive into memory by the RAR tool found on the system
    (:func:`pdm.download.member_bytes`), so nothing is written to disk (the
    80 measurements of a bearing take about 670 MB once extracted). Tools that
    cannot stream (``unar``) extract the file to a temporary folder instead.

    Args:
        archives_dir: Folder with the ``<code>.rar`` archives.
        raw_dir: Folder with already extracted measurements (optional).
    """

    def __init__(self, archives_dir: str | Path, raw_dir: str | Path | None = None):
        self.archives_dir = Path(archives_dir)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self._command: list[str] | None = None

    def _tool(self) -> list[str]:
        if self._command is None:
            self._command = rar_extractor()
            if self._command is None:
                raise RuntimeError("no RAR extractor found; install 7-Zip, unrar or unar, or use bsdtar "
                                   "(tar.exe on Windows 10+), or extract the archives with "
                                   "'python download_paderborn.py --extract'")
        return self._command

    def read(self, code: str, condition: str, measurement: int) -> tuple[np.ndarray, np.ndarray]:
        """Vibration and speed of one measurement (see :func:`read_mat`)."""
        member = member_name(code, condition, measurement)
        if self.raw_dir is not None and (self.raw_dir / member).exists():
            return read_mat(self.raw_dir / member)
        archive = self.archives_dir / f"{code}.rar"
        if not archive.exists():
            raise FileNotFoundError(f"{archive} not found; download it with 'python download_paderborn.py'")
        data = member_bytes(archive, member, self._tool())
        if data is not None:
            return read_mat(data)
        tmp = Path(tempfile.mkdtemp(prefix="paderborn_", dir=self.archives_dir.parent))
        try:
            extract(archive, tmp, self._tool(), members=[member])
            return read_mat(tmp / member)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def split_windows(vibration: np.ndarray, speed: np.ndarray, window_seconds: float = 1.0):
    """Cut one measurement into consecutive, non-overlapping windows.

    The last incomplete window (the 1-4 samples beyond 4 s) is dropped, so a
    4-s measurement gives 4 windows of 1 s. The shaft speed of a window is the
    mean of the ``speed`` channel over the same time span.

    Yields:
        ``(index, samples, shaft_rpm)`` for every window.
    """
    n = int(round(window_seconds * SAMPLING_RATE))
    ns = int(round(window_seconds * SPEED_RATE))
    count = min(len(vibration) // n, len(speed) // ns)
    for j in range(count):
        yield j, vibration[j * n:(j + 1) * n], float(np.mean(speed[j * ns:(j + 1) * ns]))


def extractor_name() -> str | None:
    """Name of the RAR extractor that will be used (``None`` if there is none)."""
    command = rar_extractor()
    return None if command is None else Path(command[0]).name
