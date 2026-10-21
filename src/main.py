"""Run the whole experiment: feature selection with a VQE and one-class SVM training with PCE.

Usage (from the repository root)::

    python src/main.py                         # full experiment (hours; resumable)
    python src/main.py --profile smoke         # every stage on a tiny slice of the data (minutes)
    python src/main.py --plan                  # stages, tasks and estimated time, without running
    python src/main.py --stages report         # only regenerate the tables and figures
    python src/main.py --rounds 0              # only the first cross-validation round

The dataset must be in ``PADERBORN/archives`` (``python download_paderborn.py``).
Results go to ``results/<profile>/``: CSV files written as each task
finishes, the loss curves, the trained parameters, the log ``run.log`` and,
at the end, ``report/`` with the tables, the figures and ``summary.md``.
Re-running the same command resumes an interrupted run: finished tasks are
skipped. Every results folder records the signature of its methodology (the
version of the code's methods and a hash of the settings); if the code or the
settings change, the old folder is archived (renamed with its signature and
date) and the run starts over, so results of different methodologies are
never mixed. ``--fresh`` deletes the profile's results first.

The terminal shows the log and, on the last line, a progress bar with the
overall progress, the elapsed time and an estimate of the remaining time.

Stages (see ``experiment/pipeline.py``): data, costs, subsets, selection, ocsvm, report.

"""

import os

# One thread per process: the workers already use every core, and numpy's own
# threads would compete for them. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.config import STAGES, get_config  # noqa: E402
from experiment.storage import read_json, write_json  # noqa: E402
from experiment.pipeline import Pipeline, StageError, fmt_seconds  # noqa: E402
from experiment.progress import Progress, Runner, setup_logging  # noqa: E402
from pdm.paderborn import extractor_name  # noqa: E402

log = logging.getLogger("experiment")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Stages: " + ", ".join(STAGES))
    parser.add_argument("--profile", default="full", choices=["full", "smoke"],
                        help="'full' = the protocol of the paper; 'smoke' = quick check of every stage")
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="run only these stages (default: all)")
    parser.add_argument("--rounds", nargs="+", type=int, choices=[0, 1, 2], help="cross-validation rounds to run")
    parser.add_argument("--n-jobs", type=int, default=0, help="worker processes (default: all logical cores)")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument("--data-dir", type=Path, help="dataset folder (default: PADERBORN/ at the repository root)")
    parser.add_argument("--results-dir", type=Path, help="parent folder of the results (default: results/)")
    parser.add_argument("--fresh", action="store_true", help="delete this profile's results before running")
    parser.add_argument("--plan", action="store_true", help="show the stages and the estimated time, then exit")
    parser.add_argument("--no-bar", action="store_true", help="do not draw the progress bar")
    parser.add_argument("--verbose", action="store_true", help="also show debug messages in the terminal")
    return parser.parse_args(argv)


def prepare_results_dir(cfg, fresh: bool) -> Path | None:
    """Make sure the results folder belongs to this methodology; returns where an older one was archived.

    A folder whose recorded signature differs from the current one (or that
    has none, as the first version did) is renamed to
    ``<profile>_v<version>_<hash>_<date>`` and a new folder is started.
    ``fresh`` deletes the folder instead.
    """
    folder, signature, archived = cfg.results_dir, cfg.signature(), None
    if folder.exists() and fresh:
        shutil.rmtree(folder)
    elif folder.exists() and any(folder.iterdir()):
        recorded = read_json(folder / "signature.json")
        if recorded != signature:
            old = recorded or {"methodology_version": 1, "hash": "unrecorded"}
            stamp = time.strftime("%Y%m%d-%H%M%S")
            archived = folder.with_name(f"{folder.name}_v{old['methodology_version']}_{old['hash']}_{stamp}")
            folder.rename(archived)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "signature.json", signature)
    return archived


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {"seed": args.seed, "n_jobs": args.n_jobs or (os.cpu_count() or 1)}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir.resolve()
    if args.results_dir:
        overrides["results_root"] = args.results_dir.resolve()
    if args.rounds:
        overrides["rounds"] = tuple(sorted(set(args.rounds)))
    cfg = get_config(args.profile, **overrides)
    stages = [s for s in STAGES if s in (args.stages or STAGES)]

    # --plan only reads: it neither creates nor archives the results folder.
    archived = None if args.plan else prepare_results_dir(cfg, args.fresh)

    progress = Progress(disable=args.no_bar or args.plan or not sys.stderr.isatty())
    setup_logging(None if args.plan else cfg.results_dir / "run.log", progress, args.verbose)
    runner = Runner(cfg.n_jobs, progress)
    t0 = time.time()
    try:
        archives = cfg.data_dir / "archives"
        if "data" in stages and not any(archives.glob("*.rar")):
            log.error(f"no archives in {archives}; download the dataset with 'python download_paderborn.py'")
            return 2
        log.info(f"profile '{cfg.profile}': results in {cfg.results_dir}, {cfg.n_jobs} worker processes, "
                 f"RAR tool: {extractor_name() or 'none'}, methodology {cfg.signature()}")
        if archived is not None:
            log.warning(f"the results folder held another methodology; it was archived as {archived}")
        if not args.plan:
            (cfg.results_dir / "config.json").write_text(cfg.to_json())
        pipeline = Pipeline(cfg, runner)
        plan = pipeline.plan(stages)
        for _, row in plan.iterrows():
            progress.plan(row["stage"], row["wall_seconds"])
            log.info(f"  plan: {row['stage']:<16} {row['tasks']:>5} tasks, estimated {fmt_seconds(row['wall_seconds'])}")
        log.info(f"  plan: total estimated {fmt_seconds(plan['wall_seconds'].sum())} "
                 "(an upper bound for stages already partly done; refined while running)")
        if args.plan:
            return 0
        pipeline.run(stages)
    except KeyboardInterrupt:
        log.warning("interrupted: finished tasks are saved; run the same command again to resume")
        return 130
    except (StageError, FileNotFoundError) as error:
        log.error(f"{error}; finished tasks are saved, run the same command again to retry")
        return 1
    finally:
        runner.close()
        progress.close()
    if runner.failures:
        log.error(f"{len(runner.failures)} tasks failed (see run.log): {runner.failures[:10]}; "
                  "run again to retry them")
        return 1
    log.info(f"done in {fmt_seconds(time.time() - t0)}; report in {cfg.results_dir / 'report'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
