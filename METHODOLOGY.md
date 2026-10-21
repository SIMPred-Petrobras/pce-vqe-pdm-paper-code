# Methodology and supplementary details

This document specifies the experiment of the article *Predictive Maintenance at the Edge: An Embedded Anomaly
Detection Pipeline Trained with Quantum Optimization* and holds the details that did not fit in the article: the full protocol, the derivations, every
hyper-parameter, the detailed results of the reference run and what was learned in earlier versions of the study.
The code implements it as methodology version 4 (`METHODOLOGY_VERSION` in `src/experiment/config.py`). Section and
equation numbers of the article are cited as "article, Eq. (n)".

Contents:

1. [Goal](#1-goal)
2. [Pipeline](#2-pipeline)
3. [Data and protocol](#3-data-and-protocol)
4. [Features](#4-features)
5. [Cost model on the ESP32-C6](#5-cost-model-on-the-esp32-c6)
6. [Feature selection: the QUBO](#6-feature-selection-the-qubo)
7. [Feature selection: the VQE](#7-feature-selection-the-vqe)
8. [One-class SVM with binary multipliers](#8-one-class-svm-with-binary-multipliers)
9. [PCE](#9-pce)
10. [Penalty or offset: the correlator bound](#10-penalty-or-offset-the-correlator-bound)
11. [Circuit reach and gradients](#11-circuit-reach-and-gradients)
12. [Results of the reference run](#12-results-of-the-reference-run)
13. [History of the methodology](#13-history-of-the-methodology)
14. [Limitations and open problems](#14-limitations-and-open-problems)

---

## 1. Goal

The article shows how to hand constrained "choose a few of many" problems to quantum optimizers, on a real case of
predictive maintenance at the edge:

1. **When the problem fits in one qubit per variable**, impose what you can in the circuit: a VQE whose ansatz
   preserves the Hamming weight only produces choices with exactly `k` items. An inequality (a budget) does not fit in
   the circuit; it enters by **unbalanced penalization**, without slack qubits but only approximately, so every answer
   is checked at readout.
2. **When it does not fit**, encode many variables in few qubits with PCE and impose the cardinality during training
   by an **offset relaxation**, because the usual penalty fails for sparse choices (Section 10).

The application is a bearing-fault detector for a microcontroller without a floating-point unit. Every quantum answer
is compared with random choices put through the same post-processing, and assessed with the metric of the
application. The study does not aim at quantum advantage: the instances are small (feature selection is solved
exactly by enumeration) and all circuits are simulated.

## 2. Pipeline

```
Paderborn .mat files (condition N15_M07_F10, vibration_1 at 64 kHz)
  -> 1-s windows, no overlap
  -> 16 features per window (10 time-domain, 6 spectral sharing one FFT)
  -> ESP32-C6 cycle model of every feature and of the FFT stage
  -> for each of 3 bearing-wise cross-validation rounds (320 healthy training windows):
       1. feature selection, 5 of 16: mRMR (relevance: healthy vs. artificial damage; redundancy: healthy)
          under 18 M cycles, auxiliary y_fft, unbalanced penalty               -> VQE, 16 + 1 qubits
          (exact optima with and without the budget by enumeration; random subsets as baseline;
           every subset scored by scikit-learn's OneClassSVM on the real damage)
       2. one-class SVM with binary multipliers, 32 of 320 windows               -> PCE, 16 qubits
          (features: the constrained exact optimum; random support vectors as baseline)
       3. supervised threshold: maximum balanced accuracy, held-out healthy vs. artificial damage
       4. test: 160 healthy windows of 2 unseen bearings + 1113 windows of 14 real-damage bearings
  -> 2D histograms (cycles vs. mRMR gap / F1 / AUC) of the selections; violins (accuracy, F1, AUC) of the detectors
```

## 3. Data and protocol

**Dataset.** The Paderborn University bearing dataset (Lessmeier et al., 2016; KAt-DataCenter) was acquired on a
modular test rig with a permanent-magnet synchronous motor, a torque-measuring shaft, the bearing test module, a
flywheel and a load motor. Each of the 32 bearings (type 6203) was measured under four operating conditions, with 20
measurements of 4 s each. The study uses condition N15_M07_F10 (1500 rpm, 0.7 N m, 1000 N) and the vibration of the
bearing housing (`vibration_1`, 64 kHz). Damage: Tables 4 and 5 of Lessmeier et al.; manufacturer: the fact sheet in
each archive.

| Bearings | Group | Damage (location; origin) | Manufacturer |
|---|---|---|---|
| K001-K006 | healthy | none (run-in from 1 to more than 50 h) | IBU |
| KA01, KA03 | artificial | OR; electro-discharge machining, electric engraving | MTK |
| KA05, KA06 | artificial | OR; electric engraving | IBU |
| KA07, KA08, KA09 | artificial | OR; drilling | IBU |
| KI01, KI03 | artificial | IR; electro-discharge machining, electric engraving | MTK |
| KI05, KI07, KI08 | artificial | IR; electric engraving | IBU |
| KA04, KA15 | real | OR; pitting, indentations | FAG |
| KA16, KA30 | real | OR; pitting, indentations | MTK |
| KA22 | real | OR; pitting | IBU |
| KB23 | real | IR (+OR); pitting | MTK |
| KB24, KB27 | real | IR + OR; pitting, indentations | FAG |
| KI04, KI14, KI17, KI18 | real | IR; pitting | MTK |
| KI16, KI21 | real | IR; pitting | FAG |

OR: outer ring; IR: inner ring. The table is `BEARINGS` in `src/pdm/paderborn.py`.

**Windows.** Each measurement is cut into consecutive 1-s windows (64,000 samples) without overlap; the final samples
that do not complete a window are dropped, and no window crosses the boundary between measurements. A 4-s measurement
gives four windows, but 13 measurements are shorter and give three: 11 have 255,996 to 255,999 samples (KA09
measurement 12; KA15, 16; KA16, 16; KA30, 13; KB23, 2; KB27, 2; KI07, 20; KI08, 11 and 17; KI17, 10; KI21, 8) and two
have about 250,000 (KA05 measurement 16 and KA08 measurement 11). This gives **480 healthy windows, 954 with
artificial damage and 1113 with real damage**.

**Rounds.** The six healthy bearings form three folds of two, balanced by run-in time (Table 7 of Lessmeier et al.),
pairing the longest with the shortest: {K001, K003} (more than 50 h and 1 h), {K002, K004} (19 h and 5 h) and
{K005, K006} (10 h and 16 h). Round `r` tests on fold `r` and trains on the other four bearings (320 healthy
windows). Windows of one bearing are never split between training and test, so a model cannot recognize the bearing
instead of its condition (`src/pdm/protocol.py`).

**What uses faults.** The one-class SVM is trained on healthy windows only. Two steps use the 954 windows of the 12
artificial-damage bearings, which are never used for testing: the relevance of the features (Section 6) and the
decision threshold (Section 8). The 14 real-damage bearings are used only for the test.

**Test set.** The 160 healthy windows of the two test bearings of the round against the 1113 windows of the 14
real-damage bearings. The test is imbalanced (87% of the windows are faulty), so the accuracy closely follows the
recall; the balanced accuracy is in the result files.

**Metrics.** The faulty class is the positive class: accuracy, balanced accuracy, precision, recall, F1, false-alarm
rate (fraction of healthy windows flagged) and the area under the ROC curve (AUC) of the anomaly score, which does not
depend on the threshold (`src/pdm/metrics.py`).

**Repetitions and seeds.** In each round, each VQE optimizer runs 30 times and each PCE optimizer 5 times, each run
starting from initial parameters drawn uniformly in `[-pi, pi)` from its seed; for the same round and seed, all
optimizers start from the same parameters (paired comparison). The random baselines draw 1000 feature subsets and 100
choices of the 32 support vectors per round. Every random choice derives from the master seed (`--seed`, default 0)
through `numpy.random.SeedSequence`, so a run is reproducible.

## 4. Features

Notation for a window `x[n]`, `n = 1..N_s` (`N_s = 64,000`): mean `x̄`, population standard deviation `σ`, peak
`x_p = max |x[n]|`, means `mean|x|` and `mean sqrt|x|` of `|x[n]|` and `sqrt|x[n]|`; `X[q]` is the one-sided magnitude
spectrum at the frequencies `f_q`, obtained by the FFT after mean removal and a Hann window, and
`p_q = X[q]^2 / sum X^2` the normalized power spectrum. Code: `src/pdm/features.py`.

| Feature | Definition | Quantity measured |
|---|---|---|
| RMS | `sqrt(mean x^2)` | mean energy of the signal |
| Peak | `x_p` | largest absolute amplitude |
| Peak to peak | `max x - min x` | total amplitude of the oscillation |
| Crest factor | `x_p / RMS` | impulsiveness |
| Shape factor | `RMS / mean|x|` | shape of the amplitude distribution |
| Impulse factor | `x_p / mean|x|` | impulses relative to the mean level |
| Clearance factor | `x_p / (mean sqrt|x|)^2` | impulsiveness, with emphasis on peaks |
| Margin factor | `x_p / RMS^2` | peak relative to the mean power |
| Kurtosis | `mean((x - x̄)/σ)^4` | weight of the tails |
| Skewness | `mean((x - x̄)/σ)^3` | asymmetry about the mean |
| Band energy | `sum_{f_q in [2, 20] kHz} X[q]^2` | energy in the band of the structural resonances |
| Spectral centroid | `μ_f = sum f_q p_q` | center of mass of the spectrum |
| Spectral spread | `σ_f = sqrt(sum (f_q - μ_f)^2 p_q)` | width about the centroid |
| Spectral skewness | `sum (f_q - μ_f)^3 p_q / σ_f^3` | asymmetry about the centroid |
| Spectral kurtosis | `sum (f_q - μ_f)^4 p_q / σ_f^4` | concentration of the spectrum |
| Spectral entropy | `-sum p_q ln p_q` | disorder of the spectrum |

* The margin factor is sometimes defined as `x_p / (mean sqrt|x|)^2`, which is identical to the clearance factor;
  `x_p / RMS^2` avoids a duplicated feature.
* The band 2-20 kHz is where the structural resonances excited by the impacts of bearing defects usually lie.
* The features are scaled to `[0, 1]` (min-max) with the healthy training windows of each round. The code computes
  them in floating point; the cost model (Section 5) assumes their fixed-point implementation on the device.
* An earlier version also used four envelope-spectrum features (amplitudes at the characteristic defect frequencies).
  They need a band-pass filter, a Hilbert transform and two more FFTs and would cost, alone, more than all the other
  features together; they were removed.

## 5. Cost model on the ESP32-C6

**The device.** Espressif's ESP32-C6 has a 32-bit RISC-V core (RV32IMAC, up to 160 MHz) with an integer multiplier
and divider and no floating-point unit: floating-point operations are emulated in software and cost about an order of
magnitude more. All features are therefore modeled in fixed point: samples are 16-bit integers (Q15), products of two
samples fit in 32 bits, and sums over a window are accumulated in 64 bits. The few divisions, square roots and
logarithms per window use integer routines; the per-sample square root of the clearance factor uses a 16-bit integer
square root, and the per-bin logarithm of the spectral entropy a fixed-point `log2` (normalization by shifts and a
small table). Code: `src/pdm/costs.py`.

**Cycle count.** The cost of a feature is the number of clock cycles per window, counted analytically from its
definition as loops over the samples (or over the spectrum bins), each iteration costing the cycles of its operations
plus the loop overhead. A 64-bit accumulation counts 3 ALU operations (add, carry, add), a 64-bit product 2
multiplications, an absolute value 3 ALU operations (branch-free). These are estimates for a single-issue, in-order
core, **not measurements on the device**.

| Operation | Cycles |
|---|---|
| load (16 or 32 bits, with load-use stall) | 2 |
| store | 1 |
| ALU (add, subtract, shift, compare, logic) | 1 |
| 32-bit multiplication | 2 |
| 32-bit integer division (iterative divider) | 33 |
| 64-bit division (software routine) | 150 |
| 16-bit integer square root | 32 |
| 32-bit integer square root | 64 |
| fixed-point `log2` | 30 |
| loop overhead (per iteration) | 2 |
| radix-2 Q15 FFT butterfly (all included) | 44 |

**The FFT.** The six spectral features share one stage, paid once if any of them is used: mean removal, the Hann
window (tabulated coefficients), zero padding to `L = 65,536` points, the real FFT, the power spectrum
(`K = L/2 + 1 = 32,769` bins) and its sum. The real FFT of `L` points is computed as a complex FFT of `L/2` points,
with `(L/4) log2(L/2)` butterflies, plus a split step. The butterfly cost is calibrated on Espressif's ESP-DSP library:
its portable C implementation of the radix-2 Q15 FFT (`dsps_fft2r_sc16`) takes about 225,000 cycles for 1024 points
on the cores without DSP extensions, about 44 cycles per butterfly. Time-domain features have no shared stage: each
counts its own passes over the window (a device computing several of them would merge some loops; the model is
conservative). The cost of a subset `S` is

```
cycles(S) = sum_{i in S} c_i + C_fft * [S uses a spectral feature].
```

| Feature | Group | Cycles (thousands) | Share of all 16 |
|---|---|---:|---:|
| RMS | time | 576 | 1.9% |
| Peak | time | 576 | 1.9% |
| Peak to peak | time | 512 | 1.7% |
| Crest factor | time | 896 | 3.0% |
| Shape factor | time | 960 | 3.2% |
| Impulse factor | time | 768 | 2.6% |
| Clearance factor | time | 2,816 | 9.4% |
| Margin factor | time | 896 | 3.0% |
| Kurtosis | time | 1,601 | 5.3% |
| Skewness | time | 1,601 | 5.3% |
| Band energy | spectrum | 129 | 0.4% |
| Spectral centroid | spectrum | 393 | 1.3% |
| Spectral spread | spectrum | 885 | 3.0% |
| Spectral skewness | spectrum | 1,279 | 4.3% |
| Spectral kurtosis | spectrum | 1,148 | 3.8% |
| Spectral entropy | spectrum | 1,376 | 4.6% |
| FFT stage (shared) | spectrum | 13,576 | 45.3% |
| **All 16 features** | | **29,989** | 187 ms at 160 MHz |

**Inference.** The decision of the one-class SVM with `w` support vectors and `M` features costs, per support vector
and per feature, a load, a subtraction, an absolute value and an addition (`L1` distance); per support vector, two
shifts, a table lookup and an accumulation; and, at the end, one comparison with the threshold. The mean over the
support vectors is a shift absorbed into the threshold, because `w` is a power of 2. For `w = 32` and `M = 5` this is
**1,345 cycles** (`ocsvm_inference_cycles`), less than 0.01% of the cost of the features.

## 6. Feature selection: the QUBO

Code: `src/pdm/selection.py` (article, Eq. (9)). With `x_i = 1` when feature `i` is selected,

```
f(x) = -(1/5) sum_i a_i x_i + C(5,2)^-1 sum_{i<j} B_ij x_i x_j,   sum_i x_i = 5,   sum_i c_i x_i + C_fft y <= C_max.
```

**Relevance.** `a_i` measures how well feature `i` separates the 320 healthy training windows of the round from the
954 artificial-damage windows (both scaled). It is the mutual information `I_i` between the feature and the binary
label, estimated by the nearest-neighbor method (`k = 3`; Kraskov et al., 2004; Ross, 2014) of scikit-learn's
`mutual_info_classif`, mapped to `[0, 1]` by Linfoot's informational coefficient of correlation,
`a_i = sqrt(1 - exp(-2 I_i))`, which equals the absolute Pearson correlation for jointly Gaussian variables and is
therefore on the scale of the redundancy. The relevances ranged from 0.25 to 0.82.

**Redundancy.** `B_ij` is the absolute Pearson correlation between features `i` and `j` over the 320 healthy training
windows of the round, with a zero diagonal. The objective is mRMR in difference form (Peng et al., 2005): the mean
redundancy of the `C(5,2) = 10` pairs minus the mean relevance of the 5 features.

**The auxiliary variable `y_fft`.** The FFT is paid once if any spectral feature is used, which is not quadratic. With
`y = y_fft`, the cycles of a subset are `sum_i c_i x_i + C_fft y`, and the term
`P sum_{i in spectrum} x_i (1 - y)` makes it worse to use a spectral feature with `y = 0`. `P` is twice the largest
change that the budget penalty can undergo when `y` flips, and no more: a large `P` creates a barrier between the
subsets with and without the FFT, and the VQE gets stuck on one side (a test with `P` an order of magnitude larger
trapped the VQE in the subsets without the FFT in every run of one round).

**The budget: unbalanced penalization.** With the cycles divided by the cost of the 16 features (`C_total`), `w_k` the
weights of the variables `z = (x, y)` and `b = C_max / C_total`, the leftover budget is `h = b - sum_k w_k z_k`, and
the penalty `zeta = -l1 h + l2 h^2` (Montanez-Barrera et al., 2024) expands, with `z_k^2 = z_k`, into linear
coefficients `l1 w_k + l2 (w_k^2 - 2 b w_k)`, quadratic coefficients `2 l2 w_k w_l` and the constant `-l1 b + l2 b^2`.
The usual form, with slack variables, would need about six more qubits to represent the leftover; the unbalanced
penalization needs none, but it is approximate: `zeta` is smallest at `h = l1 / (2 l2)` and also grows for large
leftovers.

**Tuning the weights.** For every pair `(l1, l2)` of a logarithmic grid (`l1` from 1e-3 to 10, 17 values; `l2` from
1e-3 to 1e2, 21 values), the QUBO is evaluated on the 4368 subsets with their true `y`. A pair is *valid* if the
constrained optimum (the best mRMR within the budget) is the feasible subset of lowest energy. Among the valid pairs,
the chosen one leaves the fewest subsets over the budget with an energy below that of the optimum and, among those,
has the largest margin between the optimum and the second-best feasible subset, divided by the standard deviation of
the energies (`tune_penalty`). When the budget binds, the best subsets by mRMR lie just above it, and only pairs with
a large `l2` push them above the optimum; the landscape then becomes dominated by the penalty and the optimum barely
stands out. The readout of the VQE therefore checks the budget (Section 7). In the reference run, no subset over the
budget had a lower energy than the optimum in any round. This tuning uses the enumeration, which is affordable here;
for larger problems it would be done on smaller instances.

There are 17 binary variables: 16 features and `y`.

## 7. Feature selection: the VQE

Code: `src/qopt/vqe.py` (article, Eqs. (3)-(5) and Figure 1b).

**Circuit.** The feature register has `n = 16` qubits. `X` gates prepare a state with `k = 5` ones, and a pyramid of
RBS (reconfigurable beam splitter) gates (Landman et al., 2022) acts on neighboring qubits. An RBS gate of angle `θ`
acts only on the subspace `{|01>, |10>}`,

```
RBS(θ) |01> = cos θ |01> - sin θ |10>,      RBS(θ) |10> = sin θ |01> + cos θ |10>,
```

and leaves `|00>` and `|11>` unchanged, so it preserves the number of ones (the Hamming weight). In the pyramid, the
gate on pair `(i, i+1)` appears at time steps `t = i, i+2, ..., 2n-4-i`, so pair `i` gets `n-1-i` gates, a total of
`n(n-1)/2 = 120` gates in `2n-3 = 29` time steps. Since any permutation of the qubits can be written with swaps of
neighbors, and the pyramid has enough gates for that, every subset of 5 positions can receive amplitude. Mixers that
preserve cardinality are also the key ingredient of the quantum alternating operator ansatz (Hadfield et al., 2019).
The `y` qubit, outside the constraint, gets a single `RY(φ)` rotation. There are 121 parameters.

**Simulation.** The state is simulated exactly in the weight-5 subspace: `C(16,5) = 4368` real amplitudes for the
feature register (all gates are real rotations), times the two states of `y`. Since the QUBO is diagonal in the
computational basis, the expected energy is `sum_s p(s) E(s)`, which equals `<ψ|H|ψ>` for the Ising Hamiltonian of
the article up to a constant, and a measurement samples bitstrings with probabilities `p(s)`. The exact gradient is
obtained by the adjoint method; it was checked against finite differences (difference of 1e-10) and, on small
instances, against a simulation of the full `2^(n+1)` state vector.

**Training.**

* *Adam* (Kingma and Ba) on the exact energy and gradient: step 0.05, default moments; it stops when the loss improves
  by less than a relative 1e-6 over 50 steps, or after 3000 steps, and keeps the parameters with the lowest loss.
* *SPSA* (Spall, 1998): at iteration `k`, all parameters are perturbed along a random direction
  `Δ_k ∈ {-1, +1}^Np` and `θ_{k+1} = θ_k - a_k [L(θ_k + c_k Δ_k) - L(θ_k - c_k Δ_k)] / (2 c_k) Δ_k`, with
  `a_k = a / (k + 1 + A)^0.602`, `c_k = c / (k + 1)^0.101`, `c = 0.2`, `K = 3000` iterations, `A = 0.1 K`, and `a`
  calibrated from 5 gradient estimates at the initial point so that the first step has size 0.3. Each loss estimate
  uses 1024 shots, as on hardware; the last parameters are returned.
* *COBYLA* (Powell, 1994), the derivative-free method of SciPy, on the exact energy: initial radius 0.5, at most 3000
  evaluations.

**Readout.** At the end, 1024 bitstrings are sampled from the trained state, and the answer is the one with the
lowest QUBO energy among those that meet the budget (with the true `y`); if none does, the lowest-energy one is kept
and counted as infeasible. The probability of measuring the answer in the trained state is also recorded.

**Gap.** The normalized gap of an answer is measured on the mRMR objective, without the penalties:
`(f - f*) / (mean f - f*)`, with `f*` the objective of the constrained optimum and `mean f` the mean over the 4368
subsets; 0 is the optimum and 1 the level of a random choice. Only answers over the budget can have a negative gap.

**Comparison.** Every subset that appears in some answer, random draw or exact optimum is evaluated by a conventional
one-class SVM (scikit-learn, Section 8) whose F1 and AUC on the real damage, with the supervised threshold, measure the
quality of the selection for the detector. All 4368 subsets are scored in every round (stage `subsets`); over all of
them, the F1 ranged from 0.00 to 0.99.

## 8. One-class SVM with binary multipliers

Code: `src/pdm/ocsvm.py` (article, Eqs. (1) and (10)).

**From the dual problem to the QUBO.** The dual problem of the one-class SVM (Schölkopf et al., 2001) is
`min (1/2) sum_ij α_i α_j K_ij` subject to `sum_i α_i = 1` and `0 <= α_i <= 1/(ν n)`. With `α_i = λ_i / w`,
`λ_i ∈ {0, 1}` and `sum_i λ_i = w`, the objective is `(1/(2w^2)) (sum_i K_ii λ_i + 2 sum_{i<j} K_ij λ_i λ_j)`; since
`K_ii = 1`, the first term is the constant `w`, and minimizing the objective is equivalent to the article's
Eq. (10): choosing the `w` healthy windows least similar to each other, a set of diverse prototypes of normal
operation. The upper bound `α_i <= 1/(ν n)` holds for any `ν <= w/n`. The QUBO of the code keeps the constant term
`(1/2) sum_i K_ii λ_i`.

**Kernel and features.** `K(u, v) = 2^(-γ ||u - v||_1)` (Anguita et al., 2006), with `γ` the power of 2 closest to the
inverse of the median `L1` distance between the training windows, which places typical kernel values near `2^-1` and
lets the exponent be computed with shifts. The detector uses the 5 features of the constrained exact optimum of each
round. `w = 32` support vectors, without tuning: choosing `w` by detection quality would need fault examples.

**Score and threshold.** The code uses the mean kernel value over the support vectors,
`s(u) = (1/w) sum_{i in SV} K(u, v_i)`, and flags a window as faulty when `s(u) < τ`. In the notation of the article,
`f(u) = sum_i λ_i K(v_i, u) - ρ = w (s(u) - τ)` with `ρ = w τ`, the same rule.

**Supervised threshold.** Each of the four healthy training bearings is hidden from an inner model trained on the
other three (240 windows) by a classical local search. Each inner model scores the windows of the hidden bearing,
which it never saw, and the 954 artificial-damage windows. With the scores pooled, `τ` maximizes the balanced accuracy
of the rule "faulty if `s(u) < τ`" (the candidates are the midpoints between consecutive scores). The inner models use
a classical optimizer to keep the cost low; PCE solves only the final problem with the 320 windows, and the same `τ`
is applied to it.

**Conventional one-class SVM.** The evaluation of the feature selections uses scikit-learn's `OneClassSVM` with the
same kernel (precomputed matrix), continuous multipliers and `ν = 0.1`. Its score is the multiplier-weighted mean
kernel value, `sum_i α_i K(u, v_i) / sum_i α_i`, analogous to `s(u)`; the library's decision function includes an
offset specific to each model, which would prevent a threshold estimated with the inner models from applying to the
final model. The threshold is obtained by the same supervised procedure.

**Integer-only inference.** The scaled features are represented with 8 fractional bits, `ũ_j = round(2^8 u_j)`, and
the `L1` distance to each support vector is computed with integer subtractions, absolute values and additions. With
`γ = 2^g`, the exponent `v = γ ||u - v||_1` is obtained with 6 fractional bits by a rounded bit shift. The kernel value
with 16 fractional bits, `round(2^16 · 2^-v)`, is a lookup of `2^-frac(v)` in a table of 64 entries followed by a right
shift by `floor(v)` bits (zero if `floor(v) > 16`). The integer score is the sum of the kernel values over the support
vectors, compared with the integer threshold `round(τ w 2^16)`. No step uses multiplications or floating point; the
table takes 128 bytes (`BinaryOCSVM.score_fixed`). The integer decision agreed with the floating-point one on 99.9% of
the test windows of all detectors.

## 9. PCE

Code: `src/qopt/pce.py` (article, Eqs. (6)-(8) and Figures 1c-1d). PCE is the solver of Sciorilli et al. (Nature
Communications 16, 476, 2025), extended here to cardinality constraints.

**Encoding.** Variable `i` is encoded in the `i`-th Pauli string of the list formed by `X_a X_b` on all pairs `(a, b)`
in lexicographic order, followed by `Y_a Y_b` and `Z_a Z_b` on the same pairs: `x_i = (1 + sgn <Π_i>) / 2`. The
one-class SVM (320 variables, 16 qubits) uses the 120 pairs with `X`, the 120 with `Y` and 80 with `Z`, read from three
measurement settings (every qubit measured in the X, Y or Z basis).

**IBM-native circuit.** The couplings are those of the first `n` qubits of a heavy-hex lattice: rows of 7 qubits
coupled in sequence, joined by bridge qubits at columns 0 and 4, then 2 and 6, alternately, numbered row by row, so
the first `n` qubits always form a connected graph. For `n = 16` the graph is one heavy hexagon (a 12-qubit ring) with
two 2-qubit tails, of diameter 8 (a line of 16 qubits has diameter 15). The couplings are split into three sets of
disjoint pairs (7, 7 and 2 couplings) by a greedy edge coloring. The initial state is `|0...0>`; in layer `l`, each
qubit receives `RY(θ)` and `RZ(θ')`, with `R_P(θ) = exp(-i θ P / 2)`, and then
`RZZ(θ'') = exp(-i θ'' Z⊗Z / 2)` gates act on the couplings of set `l mod 3`. With 9 layers the circuit has 288
rotations and 48 `RZZ` gates (336 parameters). On processors without fractional `RZZ` gates, each one is compiled into
two CZ gates and single-qubit rotations. The circuit was chosen for its cost of execution on this hardware, not for
the structure of the problem.

**Circuit of the first version.** The first version of the study used the circuit of the PCE authors (`brickwork`):
in layer `l`, `RX`, `RY` or `RZ` rotations (in turn) on every qubit and Mølmer-Sørensen gates
`MS(φ0, φ1, θ) = exp(-i θ/2 σ_φ0 ⊗ σ_φ1)`, `σ_φ = cos φ X + sin φ Y`, on neighboring pairs of a line, alternating
between even and odd pairs; with 6 layers, 231 parameters. The simulator implements both circuits.

**Simulation.** A custom numpy simulator stores the `2^16` amplitudes and computes the exact gradient of the loss by
the adjoint method (a forward pass, which prepares the state and the correlators, and a backward pass for the
derivatives of all parameters). Consecutive diagonal gates (the `RZ` and `RZZ` gates of a layer) commute and are
applied at once as a vector of phases `exp(-i/2 sum_k θ_k s_k)`, where `s_k ∈ {-1, 1}^(2^n)` is the diagonal of the
generator of gate `k`; their derivatives come from a single product with the table of diagonals,
`∂<O>/∂θ_k = Im <λ| diag(s_k) |ψ>`, with `|ψ>` and the adjoint state `|λ> = O|ψ>` taken after the diagonal gates. The
two-qubit correlators of a setting are read from the `n x n` matrix `S^T diag(p) S`, where `p` holds the probabilities
of the `2^n` outcomes and `S` is the `2^n x n` matrix of signs `(-1)^(z_a)`, which needs `O(2^n n)` memory instead of
`O(2^n N)`. Correlators, probabilities and gradients agree to within 1e-14 with PennyLane's `default.qubit`
(`verify_against_pennylane`), and the gradients with finite differences and with the parameter-shift rule.

**Loss with the offset relaxation.** With `x_i = (1 + s_i)/2`, the QUBO cost is written in spin form,
`c + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j`, and the minimized loss is

```
L(θ) = (1/ν) [ sum_i h_i t̃_i + sum_{i<j} J_ij t̃_i t̃_j ] + β [ (1/N) sum_i tanh^2(α c_i) ]^2,    c_i = <Π_i>,
```

where `t̃_i = 2 x̃_i - 1 = tanh(α (c_i - b))`, with `b` such that `sum_i t̃_i = 2w - N` (article, Eq. (8)). Since the
sum decreases with `b`, `b` is found by 60 bisection steps in `[min c - 20/α, max c + 20/α]`. Differentiating the
constraint gives `∂b/∂c_j = d_j / sum_k d_k`, with `d_i = α (1 - t̃_i^2)`, and, with `g_i = ∂L/∂t̃_i`,

```
∂L/∂c_i = d_i ( g_i - sum_j g_j d_j / sum_j d_j ),
```

the gradient with respect to the smooth variables projected so as to keep the constraint; the gradient of the
regularization is added to it. This loss is the article's Eq. (7) divided by `ν`, which changes neither its minimizers
nor the Adam steps and makes the tolerances relative. The scale `ν = (1/2)(sum_i |h_i| + sum_{i<j} |J_ij|)`, computed
from the cost only, is the analog of the maximum-cut estimate used by the PCE authors for MaxCut. `β = 1/2`, `α = 3`.

**Loss of the first version (penalty).** The first version used `t̃_i = tanh(α c_i)` and added the penalty
`(ρ / 4ν) (sum_i t̃_i - (2w - N))^2`, with `ρ = max_i (|Q_ii| + sum of the w largest |Q_ij|, j ≠ i)`, the largest
change of the cost that flipping one variable of a feasible solution can cause. It is still available as
`PCEConfig(cardinality="penalty")`.

**Decoding and local search.** The `w` variables with the largest correlators are set to 1 ("decoded"). One round of
local search ("after search") visits the variables in order and, for each one, evaluates all swaps with the variables
of opposite value, applying the best one if the cost decreases; each swap is evaluated in constant time from the local
fields `g = diag(Q) + Q_off x` (`qopt.feasible.swap_search`). The full local search, used for the reference and in the
inner models, applies the best of all swaps at each step until none lowers the cost, from a random feasible solution
(`greedy_search`); the best-known solution is the best of 10 such searches. The normalized gap of a solution is
`(E - E_ref) / (mean E - E_ref)`, with `mean E` the mean energy of a random choice of `w` windows.

**Training.** Both arms start from the same initial parameters for the same round and seed.

* *Adam* on the exact simulation: step 0.01, default moments; it stops when the loss improves by less than a
  relative 1e-6 over 50 steps, or after 20,000 steps, and keeps the parameters with the lowest loss.
* *SPSA*: the rule of Section 7 with `c = 0.2`, first step 0.3, 3000 iterations, and the gain calibrated from 10
  gradient estimates; each loss estimate uses 1024 shots per measurement setting. The last parameters are returned.

The exact loss of the current parameters is recorded every 10 iterations, only for the loss curves. At the end, the
circuit is read with 1024 shots per setting, the correlators are estimated as the mean of `(-1)^(z_a + z_b)` and
decoded. The baseline is 100 random choices of 32 windows per round, evaluated before and after the same round of
search. The parameters and both solutions of every run are saved (`results/<profile>/params/`), so the detectors can
be re-evaluated without retraining.

## 10. Penalty or offset: the correlator bound

**The bound.** For each Pauli axis `P ∈ {X, Y, Z}`, since `P_a^2 = I` and `(sum_a P_a)^2` is positive semidefinite,

```
sum_{a<b} <P_a P_b> = ( <(sum_a P_a)^2> - n ) / 2 >= -n/2.
```

Consider a block of `m` variables whose strings of axis `P` occupy `e_P` of the `C(n,2)` pairs. The correlators of
that axis outside the block are at most 1 each, and so are the `p` positive correlators of the block; so the `m - p`
negative correlators of the block add up, in absolute value, to at most `Γ + p`, with
`Γ = sum_P [n/2 + C(n,2) - e_P]`. Since `tanh` is concave on `[0, ∞)`, the mean of `tanh(α |c_i|)` over the negative
ones is at most `tanh(α (Γ + p) / (m - p))`, and the positive ones contribute nonnegative values; therefore

```
sum_i tanh(α c_i) >= min_{0 <= p < m} [ -(m - p) tanh( α (Γ + p) / (m - p) ) ].
```

For the one-class SVM (`n = 16`, `m = 320`, `e_X = e_Y = 120`, `e_Z = 80`, `Γ = 64`, `α = 3`) the right-hand side
is -232: the mean of `tanh(α c_i)` cannot go below **-0.725** (`qopt.pce.mean_spin_bound`). The penalty asks for
`sum_i tanh(α c_i) = 2w - m`: -288, -256, -192 and 0 for `w` = 16, 32, 64 and 160 (means -0.9, -0.8, -0.6 and 0).
For `w = 16` and 32 the target is unreachable by any circuit; for `w = 64` only correlators close to the bound reach
it; for `w = 160`, half of the variables, the choice is not sparse and the target is loose. In the offset relaxation
the sum is imposed by moving `b`, not the correlators, and the bound no longer matters.

**Effect on the first version.** Evaluated at the trained parameters of the first version, the penalty term of the
one-class SVM was 19 to 33 and the cost term about -0.2: the optimizer spent the training trying to make the
correlators more negative, and their ordering, which decides the solution, stayed essentially random. The median
normalized gap of the decoded solution (Adam) was **0.92** with the penalty, against **0.59** (Adam) and 0.95 (SPSA)
with the offset and 0.93 for the untrained circuits. The first version also used the Mølmer-Sørensen circuit and
features chosen with labels, so the comparison does not isolate the effect of the way the constraint is imposed, but
the difference between a circuit that learns nothing and one that learns is clear. (The first version is not
reproducible with this release of the code; its results are quoted for reference.)

## 11. Circuit reach and gradients

These checks were made with the simulator of `src/qopt/pce.py` while choosing the circuit; they are not part of the
pipeline.

**Light cones.** Two correlators `<P_a P_b>` can differ from the product of the single-qubit expectations only if the
light cones of qubits `a` and `b` (the qubits that influence each of them at the end of the circuit) intersect. For
the 16-qubit circuits:

| Circuit | Layers | Two-qubit gates | Parameters | Pairs with disjoint cones (of 120) | Median shared qubits |
|---|---:|---:|---:|---:|---:|
| Line (MS) | 6 | 45 | 231 | 13 | 6 |
| Line (MS) | 10 | 75 | 385 | 0 | 12 |
| Heavy-hex (RZZ) | 9 | 48 | 336 | 0 | 13 |
| Heavy-hex (RZZ) | 12 | 64 | 448 | 0 | 16 |

The 6-layer line circuit of the first version left 13 pairs without any common influence; the heavy-hex graph, with
fewer two-qubit gates than the 10-layer line, reaches every pair.

**Gradients.** Deep random parameterized circuits have gradients that vanish exponentially with the number of qubits
(barren plateaus; McClean et al., 2018). On the 9-layer heavy-hex circuit, the mean of `(∂<Π_i>/∂θ_j)^2` over the
strings, the parameters and 12 draws of the parameters fell from 9.5e-3 with 6 qubits to 7.0e-3, 5.0e-3, 4.1e-3,
2.8e-3 and 2.0e-3 with 8, 10, 12, 14 and 16 qubits: a 4.7-fold decrease, consistent with a polynomial, not an
exponential, decay. The bottleneck of the first version (Section 10) did not come from barren plateaus.

**Why PCE does not use the VQE circuit.** In a Hamming-weight-preserving circuit, `<X_a X_b> = <Y_a Y_b>` for every
pair, which would duplicate the variables encoded in `X` and in `Y`, and the sum of the `Z_a Z_b` correlators would be
fixed. The two techniques impose the constraint in different places: the VQE in the state space, PCE in the decoding.

## 12. Results of the reference run

The reference run is `python src/main.py` with the default settings (master seed 0). All the numbers below are in
`results/full/report/summary.md` and the CSV files of the run.

### Feature selection

Exact optima per round (`l1, l2`: chosen penalty weights; F1 and AUC: scikit-learn's one-class SVM on the real
damage). The budget changed the optimum in rounds 1 and 3.

| Round | Optimum | `l1, l2` | Features | Cycles (M) | mRMR objective | F1 | AUC |
|---|---|---|---|---:|---:|---:|---:|
| 1 | within budget | 3.16, 100 | impulse, margin, skewness, band energy, spectral spread | 17.86 | -0.178 | 0.735 | 0.966 |
| 1 | without budget | | margin, kurtosis, skewness, band energy, spectral kurtosis | 18.95 | -0.208 | 0.767 | 0.972 |
| 2 | within budget | 0.562, 1 | RMS, shape, clearance, margin, skewness | 6.85 | -0.106 | 0.846 | 0.561 |
| 2 | without budget | | (the same) | 6.85 | -0.106 | 0.846 | 0.561 |
| 3 | within budget | 1, 3.16 | impulse, margin, skewness, band energy, spectral spread | 17.86 | -0.113 | 0.745 | 0.996 |
| 3 | without budget | | impulse, margin, kurtosis, skewness, band energy | 18.57 | -0.130 | 0.597 | 0.970 |

Selections per method (all rounds and seeds; gap: normalized mRMR gap):

| Method | Selections | Optimum | Within budget | Mean gap | Mean cycles (M) | Mean F1 | Mean AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQE + Adam (exact) | 90 | 36% | 90% | 0.103 | 9.8 | 0.637 | 0.793 |
| VQE + COBYLA (exact) | 90 | 32% | 100% | 0.114 | 11.5 | 0.636 | 0.787 |
| VQE + SPSA (1024 shots) | 90 | 20% | 100% | 0.159 | 7.8 | 0.611 | 0.779 |
| Random | 3000 | 0% | 36% | 0.988 | 17.9 | 0.691 | 0.845 |
| Exact optimum (within budget) | 3 | 100% | 100% | 0 | 14.2 | 0.775 | 0.841 |
| Exact optimum (without budget) | 3 | 33% | 33% | -0.061 | 14.8 | 0.737 | 0.834 |

Per round, the optimum rate was 0%, 100% and 7% for Adam (mean gaps 0.250, 0.000 and 0.059); 0%, 97% and 0% for
COBYLA; and 0%, 60% and 0% for SPSA. The budget was met by 97%, 100% and 73% of the Adam answers in the three rounds.
The mean probability of measuring the answer in the trained state was 98% with Adam, 37% with COBYLA and 33% with
SPSA. Further observations that did not fit in the article:

* 94% of the random subsets use a spectral feature (pay the FFT); they had a mean F1 of 0.700 and AUC of 0.851 at
  18.7 M cycles, against 0.547 and 0.751 at 5.7 M cycles for those without the FFT. Only 32% of the VQE + Adam answers
  used the FFT.
* The VQE + Adam answers that missed the optimum had a mean gap of 0.160, 11.0 M cycles and a mean F1 of 0.525: they
  traded the FFT for cheap time-domain features, so closeness on the mRMR objective did not always translate into
  good detection.
* The random subsets within the budget had a mean F1 of 0.672 at 15.5 M cycles.

### One-class SVM with binary multipliers (32 of 320 windows)

| Method | Solution | Detectors | Median gap | Mean accuracy | Mean F1 | Mean AUC | False alarms | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PCE + Adam (exact) | decoded | 15 | 0.587 | 0.636 | 0.742 | 0.817 | 24% | 62% |
| PCE + SPSA (1024 shots) | decoded | 15 | 0.947 | 0.587 | 0.699 | 0.840 | 18% | 55% |
| Random | decoded | 300 | 0.994 | 0.652 | 0.750 | 0.722 | 33% | 65% |
| PCE + Adam (exact) | after search | 15 | 0.013 | 0.673 | 0.777 | 0.839 | 32% | 67% |
| PCE + SPSA (1024 shots) | after search | 15 | 0.006 | 0.673 | 0.777 | 0.836 | 33% | 67% |
| Random | after search | 300 | 0.006 | 0.673 | 0.777 | 0.837 | 33% | 67% |

No run reached the best-known solution exactly (2% of the random draws did after the search). Features and
thresholds per round:

| Round | Features | Threshold `τ` | Validation balanced accuracy |
|---|---|---:|---:|
| 1 | impulse, margin, skewness, band energy, spectral spread | 0.439 | 0.662 |
| 2 | RMS, shape, clearance, margin, skewness | 0.330 | 0.794 |
| 3 | impulse, margin, skewness, band energy, spectral spread | 0.428 | 0.668 |

The rounds differ widely: after the search, the mean AUC of PCE + Adam per round ranged from 0.56 to 1.00, the
false-alarm rate on the test from 0% to 97% and the recall from 58% to 84%, which reflects how different the healthy
bearings are (and the supervised threshold calibrated on a different kind of damage).

### Computational cost

The reference run took 2.5 h on a desktop computer with 12 single-threaded worker processes. The VQE simulation in
the weight-5 subspace is light: a run took on average 23 s with Adam (1181 steps), 55 s with COBYLA and 29 s with SPSA
(6.2 million shots). The 16-qubit PCE simulation is limited by memory bandwidth (effective parallelism 1.9 with all
processes running): a run took on average 0.48 h with Adam (median of 847 steps) and 1.18 h with SPSA (3000
iterations, 6020 loss evaluations, 1.8e7 shots).

## 13. History of the methodology

Every results folder records the methodology version and a hash of the settings (`signature.json`); results of
different versions are never mixed.

1. **Penalty.** Cardinality penalty in the PCE loss, brickwork circuit with Mølmer-Sørensen gates, features chosen
   with labels. The penalty failed for the sparse one-class SVM (Section 10).
2. **Offset.** Offset relaxation of the cardinality, IBM-native heavy-hex circuit.
3. **Label-free.** Selection of 5 of 16 features (the envelope features were removed) by redundancy and ESP32-C6
   cycles, with the Hamming-weight-preserving VQE; fixed `w`; random baselines; label-free threshold (5% quantile of
   the scores of held-out healthy windows). With only four healthy training bearings this quantile was set by the most
   atypical one: in round 1 the median score of the hidden windows was 0.585, 0.559 and 0.564 for K002, K004 and K005,
   but 0.488 for K006, and the threshold fell to 0.475, below every healthy test window. In rounds 1 and 3 the best
   known detector had 0% false alarms on the test but detected only 42% and 36% of the faults (accuracy 0.49 and
   0.44), whereas a threshold with 5% false alarms on the test bearings would have detected 73% and 77% (accuracy 0.75
   and 0.79). The ordering of the scores (AUC 0.89 and 0.94) was good; the threshold was not calibrated.
4. **Current.** mRMR selection (mutual-information relevance on artificial damage) under a cycle budget with
   unbalanced penalization; supervised threshold (balanced accuracy on artificial damage); test on real damage only;
   PCE solutions saved for re-evaluation.

## 14. Limitations and open problems

* A single operating condition, six healthy bearings from one manufacturer, simulated circuits and an analytical
  cycle model, not measured on the device.
* Feature selection is small enough to be solved exactly by enumeration: it is a teaching example for the VQE, not a
  hard problem. At the size of the one-class SVM, one round of classical local search brings any starting point close
  to the best-known solution, so PCE, random choices and the search alone end with the same detectors.
* The relevance of the features and the decision threshold depend on artificial-damage bearings. Computing them
  without any faulty bearing (for instance, from healthy windows of the bearing itself right after installation) is an
  open problem.
* The PCE circuit follows the native gates of the hardware. PCE circuits designed for SVM training, which exploit the
  structure of the problem (its symmetries and the sparsity of the solution), are a topic for future research.
* Next steps: running the circuits on hardware, measuring the cycles on a real ESP32-C6, larger problems and stronger
  baselines.
