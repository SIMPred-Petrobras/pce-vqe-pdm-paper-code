"""Quantum optimization core: QUBO problems, a VQE with a Hamming-weight-preserving ansatz and the PCE solver.

Modules:

* :mod:`qopt.qubo` -- the :class:`~qopt.qubo.QUBO` container, its energy and
  its spin (Ising) form.
* :mod:`qopt.feasible` -- block-constrained feasible sets (cardinality
  constraints and free variables): exact enumeration, references used to
  score answers, local searches.
* :mod:`qopt.vqe` -- variational quantum eigensolver with one qubit per
  variable and a pyramid of RBS gates that preserves the number of ones, so
  "select exactly k of n" holds by construction; trained by Adam, SPSA or COBYLA.
* :mod:`qopt.pce` -- the PCE solver (Sciorilli et al., 2025): encoding of the
  variables in Pauli correlations of a few qubits, IBM-native heavy-hex
  ansatz (``RY``, ``RZ``, ``RZZ``) or the PCE paper's brickwork ansatz with
  Molmer-Sorensen gates, relaxed loss with a cardinality offset, training by
  Adam, SPSA or SLSQP on an exact or shot-based simulation, readout, decoding
  and local search.
"""
