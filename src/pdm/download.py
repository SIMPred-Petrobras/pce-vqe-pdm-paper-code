"""Download and extraction of the Paderborn University bearing dataset (KAt-DataCenter).

The dataset (Lessmeier et al., 2016) has 32 ball bearings of type 6203:

* 6 healthy bearings (K001-K006);
* 12 bearings with artificial damage (KA01, KA03, KA05-KA09, KI01, KI03, KI05, KI07, KI08);
* 14 bearings with real damage from accelerated lifetime tests (KA04, KA15, KA16, KA22, KA30,
  KB23, KB24, KB27, KI04, KI14, KI16, KI17, KI18, KI21).

Each bearing is one RAR archive (about 150-180 MB; 5.4 GB in total) with 20 measurements of 4 s
for each of the four operating conditions (N15_M07_F10, N09_M07_F10, N15_M01_F10, N15_M07_F04),
stored as MATLAB files ``<bearing>/<condition>_<bearing>_<measurement>.mat``.

The archives are saved to ``<destination>/archives`` and, with ``--extract``, unpacked to
``<destination>/raw/<bearing>``. Extraction is optional: the experiment reads each measurement
straight from its archive. Archives already downloaded (same size as on the server) are
skipped, and interrupted downloads restart from scratch. Extraction needs an external tool that
reads RAR files: 7-Zip (``7z``/``7za``/``7zz``), ``unrar``, ``unar`` or bsdtar (the ``tar.exe``
shipped with Windows 10 and later).

The data are licensed CC BY-NC 4.0: cite Lessmeier et al. (2016) and the KAt-DataCenter,
Chair of Design and Drive Technology, Paderborn University
(https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/).

Run it through the script at the repository root::

    python download_paderborn.py                          # all 32 bearings
    python download_paderborn.py --groups healthy real    # healthy + real damage
    python download_paderborn.py --bearings K001 KA01 --extract
    python download_paderborn.py --extract --conditions N15_M07_F10
    python download_paderborn.py --dry-run                # list what would be downloaded
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"

GROUPS = {
    "healthy": ["K001", "K002", "K003", "K004", "K005", "K006"],
    "artificial": ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                   "KI01", "KI03", "KI05", "KI07", "KI08"],
    "real": ["KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
             "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"],
}
ALL_BEARINGS = [b for group in GROUPS.values() for b in group]
CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
CHUNK = 1 << 20


def ssl_context() -> ssl.SSLContext:
    """System root certificates plus certifi's (Mozilla) bundle, when installed.

    The server's chain ends in the HARICA root. On Windows, Python only sees
    the roots already in the local certificate store, and Windows installs
    less common roots on demand, so the download can fail with "self-signed
    certificate in certificate chain" until some other program fetches the
    root. Adding certifi's bundle avoids that; verification stays enabled.
    """
    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return context


def urlopen(request: str | urllib.request.Request):
    """``urllib.request.urlopen`` with the certificate store of :func:`ssl_context`."""
    return urllib.request.urlopen(request, timeout=60, context=ssl_context())


def remote_size(url: str) -> int | None:
    """Size in bytes reported by the server, or ``None`` if unknown."""
    with urlopen(urllib.request.Request(url, method="HEAD")) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download(bearing: str, archives: Path) -> Path:
    """Download one bearing archive (skipped if a complete copy already exists)."""
    url = f"{BASE_URL}{bearing}.rar"
    target = archives / f"{bearing}.rar"
    expected = remote_size(url)
    if target.exists() and expected is not None and target.stat().st_size == expected:
        print(f"  {bearing}: already downloaded ({expected / 1e6:.0f} MB)")
        return target

    partial = target.with_suffix(".rar.part")
    received = 0
    with urlopen(url) as response, open(partial, "wb") as out:
        total = expected or int(response.headers.get("Content-Length") or 0)
        while chunk := response.read(CHUNK):
            out.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {bearing}: {received / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if expected is not None and received != expected:
        partial.unlink(missing_ok=True)
        raise IOError(f"{bearing}: received {received} bytes, expected {expected}")
    partial.replace(target)
    return target


def rar_extractor() -> list[str] | None:
    """Command prefix of an available RAR extractor (see :func:`extract`), or ``None``."""
    for name in ("7z", "7za", "7zz"):
        if path := shutil.which(name):
            return [path, "x", "-y"]
    if path := shutil.which("unrar"):
        return [path, "x", "-o+"]
    if path := shutil.which("unar"):
        return [path, "-f"]
    candidates = [shutil.which("bsdtar"), shutil.which("tar")]
    if platform.system() == "Windows":
        candidates.insert(0, os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe"))
    for path in filter(None, candidates):
        try:
            version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout
        except OSError:
            continue
        if "bsdtar" in version:                       # GNU tar cannot read RAR archives
            # -m: do not restore modification times (bsdtar on Windows fails to set
            # them for these archives and would exit with an error after extracting).
            return [path, "-x", "-m", "-f"]
    return None


def extract(archive: Path, output: Path, command: list[str], members: list[str] | None = None) -> None:
    """Unpack an archive (or only the given members) into ``output`` with the detected tool.

    Args:
        archive: RAR archive.
        output: Destination folder (created if missing); members keep their
            path inside the archive (``<bearing>/<file>.mat``).
        command: Prefix returned by :func:`rar_extractor`.
        members: Paths inside the archive to extract, with ``/`` separators
            (wildcards are accepted by every supported tool); ``None`` = all.
    """
    output.mkdir(parents=True, exist_ok=True)
    members = list(members or [])
    tool = Path(command[0]).stem.lower()
    if tool in ("7z", "7za", "7zz"):
        args = command + [str(archive), f"-o{output}"] + members
    elif tool == "unrar":
        args = command + [str(archive)] + members + [str(output) + os.sep]
    elif tool == "unar":
        args = command + ["-o", str(output), str(archive)] + members
    else:                                             # bsdtar
        args = command + [str(archive), "-C", str(output)] + members
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def member_bytes(archive: Path, member: str, command: list[str]) -> bytes | None:
    """Content of one archive member, streamed through the extractor's standard output (no disk use).

    Args:
        archive: RAR archive.
        member: Path inside the archive, with ``/`` separators.
        command: Prefix returned by :func:`rar_extractor`.

    Returns:
        The member's bytes, or ``None`` if the tool cannot write to standard
        output (``unar``); use :func:`extract` then.
    """
    exe, tool = command[0], Path(command[0]).stem.lower()
    if tool in ("7z", "7za", "7zz"):
        args = [exe, "e", "-so", str(archive), member]
    elif tool == "unrar":
        args = [exe, "p", "-inul", str(archive), member]
    elif tool == "unar":
        return None
    else:                                             # bsdtar: -q stops at the first match
        args = [exe, "-x", "-q", "-O", "-f", str(archive), member]
    data = subprocess.run(args, check=True, capture_output=True).stdout
    if not data:
        raise FileNotFoundError(f"{member} not found in {archive}")
    return data


def keep_conditions(folder: Path, conditions: list[str]) -> int:
    """Delete extracted measurements of other operating conditions; returns the number kept."""
    kept = 0
    for mat in folder.rglob("*.mat"):
        if any(mat.name.startswith(c + "_") for c in conditions):
            kept += 1
        else:
            mat.unlink()
    return kept


def main(argv: list[str] | None = None, destination: Path | None = None) -> None:
    """Command-line interface (see the module docstring).

    Args:
        argv: Arguments (``None`` = ``sys.argv[1:]``).
        destination: Dataset folder (``None`` = ``PADERBORN`` at the repository root).
    """
    destination = Path(destination) if destination else Path(__file__).resolve().parents[2] / "PADERBORN"
    parser = argparse.ArgumentParser(description="Download the Paderborn (KAt) bearing dataset.")
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS),
                        help="bearing groups to download (default: all)")
    parser.add_argument("--bearings", nargs="+", metavar="CODE",
                        help="explicit bearing codes (overrides --groups), e.g. K001 KA01")
    parser.add_argument("--extract", action="store_true",
                        help="unpack the archives into PADERBORN/raw/<bearing> (optional: the experiment "
                             "reads the archives directly)")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        help="with --extract, keep only these operating conditions")
    parser.add_argument("--delete-archives", action="store_true",
                        help="with --extract, delete each archive after unpacking")
    parser.add_argument("--dry-run", action="store_true", help="only list the archives that would be downloaded")
    args = parser.parse_args(argv)

    if args.bearings:
        unknown = sorted(set(args.bearings) - set(ALL_BEARINGS))
        if unknown:
            parser.error(f"unknown bearing codes {unknown}; valid codes: {', '.join(ALL_BEARINGS)}")
        bearings = args.bearings
    else:
        bearings = [b for g in args.groups for b in GROUPS[g]]
    if (args.conditions or args.delete_archives) and not args.extract:
        parser.error("--conditions and --delete-archives require --extract")

    print(f"{len(bearings)} bearings: {' '.join(bearings)}")
    if args.dry_run:
        for b in bearings:
            print(f"  {BASE_URL}{b}.rar -> {destination / 'archives' / (b + '.rar')}")
        return

    command = None
    if args.extract:
        command = rar_extractor()
        if command is None:
            sys.exit("No RAR extractor found. Install 7-Zip (7z), unrar or unar, or use bsdtar, and run again.")

    archives = destination / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for b in bearings:
        try:
            archive = download(b, archives)
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLCertVerificationError):
                sys.exit(f"{b}: certificate verification failed ({error.reason.verify_message}). "
                         "Install certifi (pip install certifi) or point SSL_CERT_FILE to a CA bundle "
                         "that includes the HARICA root, and run again.")
            raise
        if command is not None:
            output = destination / "raw"
            members = [f"{b}/{c}_*" for c in args.conditions] if args.conditions else None
            extract(archive, output, command, members)
            note = ""
            if args.conditions:
                note = f", {keep_conditions(output / b, args.conditions)} measurements kept"
            print(f"  {b}: extracted to {output / b}{note}")
            if args.delete_archives:
                archive.unlink()
    print("Done. Please cite Lessmeier et al. (2016) and the KAt-DataCenter, Paderborn University (CC BY-NC 4.0).")
