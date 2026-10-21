"""Settings of the experiment and its two predefined profiles.

* ``"full"`` -- the protocol of the paper: Paderborn condition N15_M07_F10,
  all 20 measurements of every bearing (1-s windows without overlap), 3
  bearing-wise rounds. Feature selection picks 5 of 16
  features by mRMR (relevance: mutual information with the artificial
  damage; redundancy: correlation on healthy windows) under a budget of 18
  million ESP32-C6 cycles per window, with a VQE (Adam, SPSA, COBYLA); the
  one-class SVM with binary multipliers
  (``w = 32`` of 320 healthy training windows, 16 qubits) is trained by PCE
  (Adam on the exact simulation, SPSA with 1024 shots). Both are compared with
  random choices. The relevance of the features and the decision threshold
  use the artificial-damage bearings; the training of the one-class SVM uses
  healthy windows only; the test uses the real-damage bearings. See
  ``python src/main.py --plan`` for the run time.
* ``"smoke"`` -- the same pipeline on a tiny slice of the data (2 measurements
  per bearing, few training windows) with very short optimizer budgets, so
  that every stage runs end to end in a few minutes. Its numbers mean
  nothing; it only checks the code.

Every field can be overridden from the command line or with :func:`get_config`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from pdm.features import FeatureConfig
from pdm.paderborn import MAIN_CONDITION, MEASUREMENTS
from qopt.pce import PCEConfig
from qopt.vqe import VQEConfig

#: Version of the methodology implemented by the code. It is part of the
#: signature of a run (:meth:`Config.signature`), so results of an older
#: version are never resumed. History: 1 = cardinality penalty (first run);
#: 2 = offset relaxation, group moves, IBM-native ansatz; 3 = label-free
#: selection of 5 of 16 features (no envelope) by redundancy and ESP32-C6
#: cycles with a Hamming-weight-preserving VQE, fixed ``w``, random baselines;
#: 4 = mRMR selection (mutual-information relevance on artificial damage)
#: under a cycle budget (unbalanced penalization), supervised threshold
#: (balanced accuracy on artificial damage), test on real damage only, PCE
#: solutions saved for re-evaluation.
METHODOLOGY_VERSION = 4

#: Configuration fields that do not affect the results; left out of the signature.
RUNTIME_FIELDS = ("profile", "data_dir", "results_root", "n_jobs", "rounds")

#: Repository root (the folder that contains ``src/`` and ``PADERBORN/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Stages of the pipeline, in execution order.
STAGES = ("data", "costs", "subsets", "selection", "ocsvm", "report")

#: Display names of the methods.
LABELS = {
    "adam": "VQE + Adam (exact)",
    "spsa": "VQE + SPSA (1024 shots)",
    "cobyla": "VQE + COBYLA (exact)",
    "random": "random",
    "exact": "exact optimum",
    "pce_adam": "PCE + Adam (exact)",
    "pce_spsa": "PCE + SPSA (1024 shots)",
}


@dataclass
class Config:
    """All settings of one run.

    Attributes:
        profile: Profile name; results go to ``results_root / profile``.
        data_dir: Dataset folder (``archives/`` with the RAR files; optional
            ``raw/`` with extracted measurements; ``features/`` is the cache).
        results_root: Parent folder of the per-profile result folders.
        seed: Master seed; every random choice derives from it.
        n_jobs: Worker processes (``0`` = all logical cores).

        condition: Operating condition of the experiment.
        measurements: Measurements used per bearing (1-20).
        features: Feature-extraction parameters.
        rounds: Rounds to run (``None`` = all three).
        max_train_per_bearing: Healthy training windows kept per bearing
            (``None`` = all 80; only the smoke profile reduces it).

        n_select: Features to select (the Hamming weight of the VQE).
        cycle_budget: Largest cycles per window of the selected features
            (the inequality constraint of the selection QUBO).
        vqe: Base settings of the VQE.
        vqe_arms: Optimizers of the VQE (``adam``: exact; ``spsa``:
            ``vqe_shots`` per loss estimate; ``cobyla``: exact).
        vqe_shots: Shots per loss estimate of the SPSA arm.
        vqe_seeds: VQE runs per round and optimizer (different initial parameters).
        random_selections: Random subsets drawn per round (the random baseline).
        nu: ``nu`` of scikit-learn's one-class SVM that scores every subset.

        w: Support vectors of the binary one-class SVM.
        pce: Settings of the PCE solver (16 qubits).
        pce_arms: Training arms of the PCE (``pce_adam``: exact; ``pce_spsa``: ``pce_shots``).
        pce_shots: Shots per loss estimate of the SPSA arm (and of the final readout).
        spsa_iterations, spsa_c, spsa_target_step: SPSA settings of the PCE.
        ocsvm_seeds: PCE runs per round and arm.
        random_ocsvm: Random choices of the ``w`` support vectors per round.
        inner_restarts: Local-search restarts of each inner-validation model.
        reference_restarts: Restarts of the classical best-known solution.
        fixed_point: ``(in_bits, out_bits, exp_bits)`` of the integer inference.
    """

    profile: str = "full"
    data_dir: Path = REPO_ROOT / "PADERBORN"
    results_root: Path = REPO_ROOT / "results"
    seed: int = 0
    n_jobs: int = 0

    condition: str = MAIN_CONDITION
    measurements: tuple[int, ...] = MEASUREMENTS
    features: FeatureConfig = field(default_factory=FeatureConfig)
    rounds: tuple[int, ...] | None = None
    max_train_per_bearing: int | None = None

    n_select: int = 5
    cycle_budget: float = 18e6
    vqe: VQEConfig = field(default_factory=VQEConfig)
    vqe_arms: tuple[str, ...] = ("adam", "spsa", "cobyla")
    vqe_shots: int = 1024
    vqe_seeds: int = 30
    random_selections: int = 1000
    nu: float = 0.1

    w: int = 32
    pce: PCEConfig = field(default_factory=lambda: PCEConfig(k=2, ansatz="heavy_hex", layers=9, alpha=3.0,
                                                              max_steps=20000, patience=50, tolerance=1e-6,
                                                              readout_shots=1024, log_every=10))
    pce_arms: tuple[str, ...] = ("pce_adam", "pce_spsa")
    pce_shots: int = 1024
    spsa_iterations: int = 3000
    spsa_c: float = 0.2
    spsa_target_step: float = 0.3
    ocsvm_seeds: int = 5
    random_ocsvm: int = 100
    inner_restarts: int = 1
    reference_restarts: int = 10
    fixed_point: tuple[int, int, int] = (8, 16, 6)

    @property
    def results_dir(self) -> Path:
        """Folder of this profile's CSV files, figures and log."""
        return Path(self.results_root) / self.profile

    def vqe_config(self, arm: str) -> VQEConfig:
        """VQE settings of one optimizer."""
        if arm == "adam":
            return replace(self.vqe, optimizer="adam", training_shots=None)
        if arm == "spsa":
            return replace(self.vqe, optimizer="spsa", training_shots=self.vqe_shots)
        if arm == "cobyla":
            return replace(self.vqe, optimizer="cobyla", training_shots=None)
        raise ValueError(f"unknown VQE arm {arm!r}")

    def pce_config(self, arm: str) -> PCEConfig:
        """PCE settings of one training arm."""
        if arm == "pce_adam":
            return replace(self.pce, optimizer="adam", training_shots=None, readout_shots=self.pce_shots)
        if arm == "pce_spsa":
            return replace(self.pce, optimizer="spsa", training_shots=self.pce_shots, readout_shots=self.pce_shots,
                           spsa_iterations=self.spsa_iterations, spsa_c=self.spsa_c,
                           spsa_target_step=self.spsa_target_step)
        raise ValueError(f"unknown PCE arm {arm!r}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def signature(self) -> dict:
        """Identity of the methodology of a run: the version and a hash of every result-affecting setting."""
        settings = {k: v for k, v in json.loads(self.to_json()).items() if k not in RUNTIME_FIELDS}
        text = json.dumps({"methodology_version": METHODOLOGY_VERSION, **settings}, sort_keys=True)
        return {"methodology_version": METHODOLOGY_VERSION, "hash": hashlib.sha1(text.encode()).hexdigest()[:12]}

    def __post_init__(self) -> None:
        unknown = set(self.vqe_arms) - {"adam", "spsa", "cobyla"}
        unknown |= set(self.pce_arms) - {"pce_adam", "pce_spsa"}
        if unknown:
            raise ValueError(f"unknown arms {sorted(unknown)}")
        if self.cycle_budget <= 0:
            raise ValueError("cycle_budget must be positive")


def _smoke() -> Config:
    """Tiny, fast version of the full pipeline (checks the code, not the method)."""
    return Config(
        profile="smoke",
        measurements=(1, 2),
        max_train_per_bearing=8,
        vqe=VQEConfig(max_steps=40, patience=20, spsa_iterations=40, cobyla_maxiter=130, readout_shots=256,
                      log_every=5),
        vqe_seeds=2,
        random_selections=40,
        w=4,
        pce=PCEConfig(k=2, ansatz="heavy_hex", layers=2, alpha=3.0, max_steps=40, patience=20, tolerance=1e-6,
                      readout_shots=256, log_every=5),
        pce_shots=256,
        spsa_iterations=40,
        ocsvm_seeds=2,
        random_ocsvm=10,
        reference_restarts=3,
    )


PROFILES = {"full": Config, "smoke": _smoke}


def get_config(profile: str = "full", **overrides) -> Config:
    """A predefined profile, optionally with some fields replaced."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    return replace(PROFILES[profile](), **overrides)
