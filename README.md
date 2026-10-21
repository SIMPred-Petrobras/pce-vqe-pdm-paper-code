# Predictive Maintenance at the Edge: An Embedded Anomaly Detection Pipeline Trained with Quantum Optimization

Code of the article *Predictive Maintenance at the Edge: An Embedded Anomaly Detection Pipeline Trained with Quantum
Optimization* (IEEE Computing in Science & Engineering, special issue on quantum computing), evaluated on the
Paderborn University bearing dataset.

The article presents a predictive-maintenance pipeline that detects anomalies in bearings from vibration signals and
runs on a low-cost microcontroller without a floating-point unit (ESP32-C6). Two steps of its training are combinatorial
problems written as QUBOs (quadratic unconstrained binary optimization) and solved by simulated quantum optimizers:

1. **Feature selection with the VQE.** Choose 5 of 16 vibration features by mRMR (maximum relevance, minimum
   redundancy) under a budget of 18 million clock cycles per window on the ESP32-C6 (4368 possible subsets). The VQE
   uses a **Hamming-weight-preserving ansatz** (a pyramid of RBS gates), so every measured bitstring has exactly 5
   ones and the QUBO needs no penalty for that constraint. The budget, an inequality, enters by **unbalanced
   penalization**, without slack qubits; an auxiliary qubit `y_fft` pays the FFT shared by the spectral features once.
   Trained by Adam (exact simulation), SPSA (1024 shots) and COBYLA.
2. **One-class SVM training with PCE.** With binary multipliers and the multiplier-free kernel
   `2^(-gamma ||u - v||_1)`, inference needs only integer additions, shifts and a small table, but training becomes
   combinatorial: choose `w = 32` support vectors among 320 healthy training windows. **Pauli-correlation encoding
   (PCE)** stores the 320 binary variables in the two-qubit correlators of 16 qubits, with an IBM-native circuit. The
   cardinality constraint is imposed by an **offset relaxation**, because the usual penalty asks the correlators for
   values that quantum mechanics forbids when the choice is sparse. Trained by Adam (exact simulation) and SPSA (1024
   shots).

Every quantum answer is compared with random choices put through the same post-processing and assessed with the
metric of the application. The study does not aim at quantum advantage: the instances are small and all circuits
are simulated.

**[METHODOLOGY.md](METHODOLOGY.md)** specifies the whole experiment and holds what did not fit in the article: the
protocol, the derivations, every hyper-parameter, the detailed results of the reference run and the history of the
study.

## Installation

Python 3.12 or newer (tested with 3.14):

```bash
pip install -r requirements.txt
```

Reading the dataset needs a tool that reads RAR archives: 7-Zip (`7z`), `unrar`, `unar` or bsdtar (the `tar.exe`
shipped with Windows 10 and later works).

## Data

```bash
python download_paderborn.py          # 32 archives, 5.4 GB, into PADERBORN/archives
python download_paderborn.py --help   # options: some bearings only, extraction, dry run
```

The experiment reads each measurement straight from its archive into memory (nothing is extracted to disk) and caches
the features of every bearing in `PADERBORN/features/`. The data (KAt-DataCenter, Chair of Design and Drive
Technology, Paderborn University) are licensed CC BY-NC 4.0; please cite Lessmeier et al. (2016).

## Running

The whole methodology runs from a single entry point:

```bash
python src/main.py --profile smoke    # every stage on a tiny slice of the data, about 1-2 minutes
python src/main.py --plan             # stages, number of tasks and estimated time of the full run
python src/main.py                    # full experiment (about 2.5 hours on a 12-thread desktop)
python src/main.py --stages report    # only regenerate the tables and figures from saved results
```

The terminal shows the log and, on its last line, a progress bar with the overall progress, the elapsed time and the
estimated remaining time. Tasks run in parallel worker processes (`--n-jobs`, default: all logical cores). Every
result is written to disk as soon as its task finishes, so an interrupted run resumes when the same command is run
again. Each results folder records the signature of its methodology (a version number of the methods and a hash of
the settings): when the methods or the settings change, the old folder is archived (renamed to
`<profile>_v<version>_<hash>_<date>`) and the run starts over, so results of different methodologies are never mixed;
`--fresh` deletes the folder instead. Other options: `--rounds 0` (a single cross-validation round), `--seed`,
`--data-dir`, `--results-dir`, `--no-bar`, `--verbose`. The `smoke` profile only checks the code; its numbers mean
nothing.

### Stages

| Stage | What it does | Main outputs (in `results/<profile>/`) |
|---|---|---|
| `data` | windows and 16 features of every bearing | `windows.csv` (feature cache in `PADERBORN/features/`) |
| `costs` | ESP32-C6 clock cycles of every feature and of the FFT stage | `feature_costs.csv`, `fft_stage.csv` |
| `subsets` | scikit-learn one-class SVM with each of the 4368 subsets of 5 features, in every round | `subset_metrics.csv` |
| `selection` | feature-selection QUBO per round (penalty weights tuned by enumeration); exact optima with and without the budget, VQE (Adam, SPSA, COBYLA), random subsets | `selection_problems.csv`, `vqe_runs.csv`, `vqe_history.csv`, `random_selections.csv` |
| `ocsvm` | supervised threshold and best-known solution; PCE (Adam, SPSA) and random support vectors, before and after local search | `ocsvm_prep.json`, `ocsvm_runs.csv`, `ocsvm_results.csv`, `loss_history.csv`, `params/` |
| `report` | tables, figures and a Markdown summary | `report/` |

`run.log` holds the full log (with the tracebacks of failed tasks) and `config.json` the settings of the run.

### Figures of the article

| Article | File in `results/full/report/` |
|---|---|
| Figure 2 (feature selection: mRMR gap and F1 against cycles) | `fig_selection_gap_f1.pdf` |
| Figure 3 (gap of the decoded PCE solutions) | `fig_decoded_gap.pdf` |
| Figure 4 (PCE loss curves) | `fig_loss_curves.pdf` |
| Figure 5 (accuracy, F1 and AUC of the binary detectors) | `fig_ocsvm_violins.pdf` |

`summary.md` in the same folder has the tables and the numbers quoted in the article. Figure 1 (diagrams of the
pipeline and of the circuits) is not generated by the code.

## Code structure

```
src/
  main.py               single entry point (command line, logging, progress bar)
  qopt/                 quantum optimization core, independent of the application
    qubo.py             QUBO container and its Ising (spin) form
    feasible.py         cardinality-constrained blocks, exact enumeration, references, local searches
    vqe.py              VQE with a Hamming-weight-preserving RBS ansatz (exact subspace simulation, adjoint
                        gradient), Adam / SPSA / COBYLA, best-sample readout
    pce.py              PCE solver: encoding, IBM heavy-hex (and the PCE paper's brickwork) ansatz, offset
                        relaxation, Adam / SPSA / SLSQP on exact or shot-based simulation, readout, decoding,
                        correlator bound, PennyLane cross-check
  pdm/                  predictive-maintenance application
    download.py         dataset download and archive access
    paderborn.py        bearing table (damage, manufacturer), measurements and windows
    protocol.py         bearing-wise cross-validation: rounds, calibration and test sets
    features.py         the 16 vibration features
    costs.py            ESP32-C6 cycle model of feature extraction and inference
    selection.py        mRMR feature-selection QUBO under a cycle budget (y_fft, unbalanced penalization)
    ocsvm.py            one-class SVM QUBO, kernel, supervised threshold, integer-only inference, scikit-learn reference
    metrics.py          detection metrics
  experiment/           the experiment
    config.py           settings and the `full` and `smoke` profiles
    pipeline.py         stages and their outputs
    tasks.py            work units executed by the worker processes
    progress.py         logging, progress bar and parallel runner
    estimates.py        run-time estimates (progress bar and --plan)
    storage.py          resumable CSV files
    report.py           tables and summary.md
    figures.py, style.py
download_paderborn.py   dataset download (wrapper around src/pdm/download.py)
```

Both simulators are plain numpy. The VQE simulator works in the weight-`k` subspace (4368 x 2 amplitudes for 5 of 16
plus the free qubit); the PCE simulator holds the full 16-qubit state vector and agrees with PennyLane to machine
precision:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from qopt.pce import verify_against_pennylane; print(verify_against_pennylane(ansatz='heavy_hex'))"
```

### From the article to the code

| Article | Code |
|---|---|
| Eq. (1), OCSVM score `f(u) = sum λ_i K(v_i, u) - ρ` | `pdm.ocsvm.BinaryOCSVM.score` (mean kernel `s(u)`; `ρ = w τ`) |
| Base-2 Laplacian kernel | `pdm.ocsvm.l1_kernel`, `auto_gamma`; integer version `BinaryOCSVM.score_fixed` |
| Eq. (2), QUBO | `qopt.qubo.QUBO` |
| Eqs. (3)-(4), Ising Hamiltonian, `J` and `h` | `qopt.qubo.QUBO.ising` (spin sign convention noted there) |
| Eq. (5), RBS gate and pyramid | `qopt.vqe.pyramid`, `HammingWeightCircuit` |
| VQE training and readout | `qopt.vqe.run_vqe` |
| Eq. (6), PCE encoding and decoding | `qopt.pce.pauli_strings`, `decode` |
| PCE circuit (Figure 1d) | `qopt.pce.heavy_hex`, `heavy_hex_edges`, `StateVectorCircuit` |
| Eqs. (7)-(8), PCE loss and offset | `qopt.pce.PCEProblem.loss`, `PCEProblem.relax` |
| Correlator bound `sum <P_a P_b> >= -n/2` | `qopt.pce.mean_spin_bound` |
| Local search after decoding | `qopt.feasible.swap_search` |
| Eq. (9), mRMR selection with constraints | `pdm.selection.build_problem`, `relevance`, `redundancy`, `tune_penalty` |
| Eq. (10), OCSVM with binary multipliers | `pdm.ocsvm.ocsvm_qubo` |
| Supervised threshold | `pdm.ocsvm.inner_threshold` |
| Table 1, cycles per feature | `pdm.costs.cost_table`, `fft_table` |

## References

* C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition Monitoring of Bearing Damage in Electromechanical
  Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven
  Classification", PHM Society European Conference (2016).
* A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor", *Nature Communications* 5,
  4213 (2014).
* J. Landman et al., "Quantum methods for neural networks and application to medical image classification",
  *Quantum* 6, 881 (2022) (RBS gates and pyramid circuits).
* M. Sciorilli et al., "Towards large-scale quantum optimization solvers with few qubits", *Nature Communications*
  16, 476 (2025).
* J. A. Montanez-Barrera et al., "Unbalanced penalization: a new approach to encode inequality constraints of
  combinatorial problems for quantum optimization algorithms", *Quantum Science and Technology* 9, 025022 (2024).
* B. Schölkopf et al., "Estimating the Support of a High-Dimensional Distribution", *Neural Computation* 13(7),
  1443-1471 (2001).
* D. Anguita, S. Pischiutta, S. Ridella, D. Sterpi, "Feed-Forward Support Vector Machine Without Multipliers",
  *IEEE Transactions on Neural Networks* 17(5), 1328-1331 (2006).
* H. Peng, F. Long, C. Ding, "Feature selection based on mutual information: criteria of max-dependency,
  max-relevance, and min-redundancy", *IEEE TPAMI* 27(8), 1226-1238 (2005).
* A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating mutual information", *Physical Review E* 69, 066138 (2004).
* J. C. Spall, "Implementation of the simultaneous perturbation algorithm for stochastic optimization", *IEEE
  Transactions on Aerospace and Electronic Systems* 34(3), 817-823 (1998).
* D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR (2015).
* Espressif Systems, ESP-DSP library (FFT benchmarks used to calibrate the cycle model).
