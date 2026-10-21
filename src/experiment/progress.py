"""Terminal logging, a progress bar pinned to the bottom of the terminal, and the parallel task runner.

Progress bar
------------
A single ``tqdm`` bar stays on the last line of the terminal for the whole
run; log messages are printed above it (:class:`TqdmHandler`) and also saved
to ``run.log``. The bar shows the overall progress (in percent), the elapsed
time and an estimate of the remaining time.

Tasks take very different times (a VQE run lasts seconds, a 16-qubit PCE run
up to an hour), so progress is measured in *estimated work*: every task has a
weight (its expected run time in CPU-seconds, see
:meth:`experiment.pipeline.Pipeline.plan` and :mod:`experiment.estimates`), and
running tasks report the fraction of their optimizer iterations already done.
The remaining time is ``remaining work / (work done / elapsed time)``, so a
uniform error in the estimates cancels out, and the estimate improves as the
run goes on. Stages whose task lists depend on earlier results (for example,
the one-class SVM runs depend on the selection) start with a planned weight that
is replaced by the actual one when the stage begins. A background thread
redraws the bar every second, so the elapsed time keeps moving during long tasks.

Parallel tasks
--------------
:class:`Runner` executes tasks in a pool of worker processes (one thread
each, see ``main.py``). Workers send their log records and progress fractions
to the main process through a queue; the main process writes every result to
disk as soon as its task finishes, so an interrupted run resumes from the
last finished task.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

LOGGER_NAME = "experiment"
log = logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------
# Progress bar
# --------------------------------------------------------------------------

class Progress:
    """Weighted progress bar pinned to the bottom of the terminal.

    Args:
        total: Planned total work (estimated CPU-seconds).
        disable: Do not draw the bar (e.g. when the output is not a terminal).
    """

    BAR_FORMAT = "{desc} {percentage:5.1f}%|{bar}| elapsed {elapsed} | remaining ~{remaining} {postfix}"

    def __init__(self, total: float = 1.0, disable: bool = False):
        self._lock = threading.RLock()
        self._done = 0.0                                    # work of finished tasks
        self._running: dict[str, tuple[float, float]] = {}  # task -> (weight, fraction done)
        self._finished: set[str] = set()                    # late messages of these tasks are ignored
        self._planned: dict[str, float] = {}                # stage -> planned work
        self._total = max(float(total), 1e-9)
        self._stage = ""
        self._counts = (0, 0)
        self.bar = tqdm(total=self._total, bar_format=self.BAR_FORMAT, file=sys.stderr, dynamic_ncols=True,
                        smoothing=0, mininterval=0.5, leave=True, disable=disable, desc="")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._refresh_loop, name="progress-refresh", daemon=True)
        self._thread.start()

    # -- planning ------------------------------------------------------------
    def plan(self, stage: str, work: float) -> None:
        """Set the planned work of a stage (replaces any previous plan for it)."""
        with self._lock:
            self._planned[stage] = max(0.0, float(work))
            self._recompute_total()

    def _recompute_total(self) -> None:
        self._total = max(self._done + sum(w * (1 - f) for w, f in self._running.values())
                          + sum(self._planned.values()), self._done, 1e-9)

    # -- stages and tasks ----------------------------------------------------
    def stage(self, name: str, n_tasks: int, work: float) -> None:
        """Start a stage: its planned work becomes the actual work of its tasks."""
        with self._lock:
            self._stage, self._counts = name, (0, n_tasks)
            self._planned.pop(name, None)
            self._planned[f"{name}:queued"] = float(work)
            self._recompute_total()
            self._draw()

    def task_started(self, key: str, weight: float) -> None:
        with self._lock:
            if key in self._finished or key in self._running:
                return
            queued = self._planned.get(f"{self._stage}:queued", 0.0)
            self._planned[f"{self._stage}:queued"] = max(0.0, queued - weight)
            self._running[key] = (weight, 0.0)

    def task_progress(self, key: str, fraction: float) -> None:
        with self._lock:
            if key in self._running:
                weight, _ = self._running[key]
                self._running[key] = (weight, min(max(float(fraction), 0.0), 1.0))

    def task_finished(self, key: str, weight: float) -> None:
        with self._lock:
            self._finished.add(key)
            if key in self._running:
                self._running.pop(key)
            else:                                      # finished without a start message
                queued = self._planned.get(f"{self._stage}:queued", 0.0)
                self._planned[f"{self._stage}:queued"] = max(0.0, queued - weight)
            self._done += weight
            done, total = self._counts
            self._counts = (done + 1, total)
            self._recompute_total()
            self._draw()

    def end_stage(self) -> None:
        with self._lock:
            self._planned.pop(f"{self._stage}:queued", None)
            self._recompute_total()
            self._draw()

    # -- drawing -------------------------------------------------------------
    def _draw(self) -> None:
        n = self._done + sum(w * f for w, f in self._running.values())
        self.bar.total = self._total
        self.bar.n = min(n, self._total)        # set directly: tqdm then infers the mean rate n / elapsed
        done, total = self._counts
        self.bar.set_description_str(f"[{self._stage} {done}/{total}]" if self._stage else "", refresh=False)
        self.bar.refresh()

    def _refresh_loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                try:
                    self._draw()
                except Exception:                     # never let the display thread kill the run
                    pass

    def write(self, message: str) -> None:
        """Print a line above the bar."""
        with self._lock:
            self.bar.write(message, file=sys.stderr)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        with self._lock:
            self._draw()
            self.bar.close()


class TqdmHandler(logging.Handler):
    """Logging handler that prints above the progress bar."""

    def __init__(self, progress: Progress | None = None):
        super().__init__()
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if self.progress is not None:
                self.progress.write(message)
            else:
                tqdm.write(message, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path | None, progress: Progress | None, verbose: bool = False) -> logging.Logger:
    """Log INFO messages (DEBUG with ``verbose``) to the terminal and everything to ``log_file`` (if given)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    terminal = TqdmHandler(progress)
    terminal.setLevel(logging.DEBUG if verbose else logging.INFO)
    terminal.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(terminal)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, encoding="utf-8")
        file.setLevel(logging.DEBUG)
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(processName)s %(message)s"))
        logger.addHandler(file)
    return logger


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_QUEUE = None          # queue to the main process (set in each worker)
_CURRENT = None        # key of the task running in this worker


class _QueueLogHandler(logging.Handler):
    """Send log records of a worker to the main process."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _QUEUE.put(("log", record.levelno, f"[{_CURRENT}] {record.getMessage()}" if _CURRENT else
                        record.getMessage()))
        except Exception:
            pass


def _init_worker(queue) -> None:
    """Initializer of every worker process: route logging and progress to the main process."""
    global _QUEUE
    _QUEUE = queue
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(_QueueLogHandler())


class _Reporter:
    """Progress callback of one task, throttled to one message per second."""

    def __init__(self, key: str, sink: Callable[[tuple], None]):
        self.key, self.sink, self.last = key, sink, 0.0

    def __call__(self, fraction: float) -> None:
        now = time.monotonic()
        if now - self.last >= 1.0 or fraction >= 1.0:
            self.last = now
            self.sink(("progress", self.key, float(fraction)))


def _execute(fn: Callable, spec: Any, key: str) -> tuple[str, Any, float]:
    """Run one task in a worker; returns ``(key, result, seconds)``."""
    global _CURRENT
    _CURRENT = key
    t0 = time.time()
    _QUEUE.put(("start", key, 0.0))
    try:
        result = fn(spec, _Reporter(key, _QUEUE.put))
    finally:
        _CURRENT = None
    return key, result, time.time() - t0


# --------------------------------------------------------------------------
# Main side
# --------------------------------------------------------------------------

@dataclass
class Task:
    """One unit of work.

    Attributes:
        key: Unique name (also used in log messages).
        fn: Top-level function ``fn(spec, progress) -> result`` (it must be
            importable by the worker processes).
        spec: Its input (picklable).
        weight: Estimated cost in CPU-seconds (for the progress bar).
        on_done: Called in the main process with the result (saves it to disk).
    """

    key: str
    fn: Callable
    spec: Any
    weight: float = 1.0
    on_done: Callable[[Any], None] | None = None


@dataclass
class Runner:
    """Executes lists of tasks, in parallel worker processes or in the main process (``n_jobs = 1``).

    Args:
        n_jobs: Worker processes.
        progress: The progress bar.
    """

    n_jobs: int
    progress: Progress
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pool = None
        self._queue = None
        self._listener = None
        self._weights: dict[str, float] = {}

    # -- pool ----------------------------------------------------------------
    def _ensure_pool(self) -> None:
        if self._pool is not None or self.n_jobs <= 1:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._pool = ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=ctx, initializer=_init_worker,
                                         initargs=(self._queue,))
        self._listener = threading.Thread(target=self._listen, name="worker-listener", daemon=True)
        self._listener.start()

    def _listen(self) -> None:
        """Forward worker messages (logs, task starts, progress) to the logger and the bar."""
        while True:
            try:
                message = self._queue.get()
            except (EOFError, OSError):
                return
            if message is None:
                return
            kind, a, b = message
            if kind == "log":
                log.log(a, b)
            elif kind == "start":
                self.progress.task_started(a, self._weights.get(a, 0.0))
            elif kind == "progress":
                self.progress.task_progress(a, b)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._queue.put(None)
            self._listener.join(timeout=5)
            self._pool = None

    # -- execution -----------------------------------------------------------
    def run(self, stage: str, tasks: list[Task]) -> int:
        """Run the tasks of one stage; returns the number of failed tasks.

        The heaviest tasks are submitted first, which shortens the tail of the
        stage. A failing task is logged (with its traceback in ``run.log``) and
        does not stop the others; re-running the pipeline retries it.
        """
        tasks = sorted(tasks, key=lambda t: -t.weight)
        self._weights.update({t.key: t.weight for t in tasks})
        self.progress.stage(stage, len(tasks), sum(t.weight for t in tasks))
        failed = 0
        if not tasks:
            self.progress.end_stage()
            return 0
        if self.n_jobs <= 1:
            for t in tasks:
                failed += not self._run_inline(t)
        else:
            self._ensure_pool()
            by_future = {self._pool.submit(_execute, t.fn, t.spec, t.key): t for t in tasks}
            pending = set(by_future)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    t = by_future[future]
                    try:
                        _, result, seconds = future.result()
                    except Exception as error:
                        failed += 1
                        self._failed(t, error, "".join(traceback.format_exception(error)))
                    else:
                        failed += not self._finish(t, result, seconds)
        self.progress.end_stage()
        return failed

    def _run_inline(self, t: Task) -> bool:
        self.progress.task_started(t.key, t.weight)
        t0 = time.time()
        try:
            result = t.fn(t.spec, _Reporter(t.key, lambda m: self.progress.task_progress(m[1], m[2])))
        except Exception as error:
            self._failed(t, error, traceback.format_exc())
            return False
        return self._finish(t, result, time.time() - t0)

    def _finish(self, t: Task, result: Any, seconds: float) -> bool:
        try:
            if t.on_done is not None:
                t.on_done(result)
        except Exception as error:
            self._failed(t, error, traceback.format_exc())
            return False
        log.debug(f"{t.key} finished in {seconds:.1f}s")
        self.progress.task_finished(t.key, t.weight)
        return True

    def _failed(self, t: Task, error: Exception, trace: str) -> None:
        self.failures.append(t.key)
        log.error(f"task {t.key} failed: {error!r} (traceback in run.log)")
        log.debug(trace)
        self.progress.task_finished(t.key, t.weight)
