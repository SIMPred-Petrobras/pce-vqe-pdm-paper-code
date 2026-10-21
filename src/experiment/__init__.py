"""The experiment: configuration, pipeline stages, worker tasks, progress display, report and figures.

Modules:

* :mod:`experiment.config` -- settings and the ``full`` and ``smoke`` profiles.
* :mod:`experiment.pipeline` -- the stages and the files they write.
* :mod:`experiment.tasks` -- units of work executed by the worker processes.
* :mod:`experiment.progress` -- logging, progress bar and parallel task runner.
* :mod:`experiment.estimates` -- run-time estimates (progress bar and ``--plan``).
* :mod:`experiment.storage` -- incremental, resumable CSV files.
* :mod:`experiment.report`, :mod:`experiment.figures`, :mod:`experiment.style` --
  tables, figures and their style.

Run it with ``python src/main.py`` (see that file).
"""
