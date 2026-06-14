# Literature Cortex Seed Library — L0-L4 Summary

## Structure

| Layer | File | Nodes | Role |
|-------|------|-------|------|
| L0 | `seed_L0_meta.json` | 6 | Strategy control, meta-learning, causation rules, epistemic boundaries |
| L1 | `seed_L1_axioms.json` | 12 | Mathematical foundations (ZFC, Peano, Gödel, CH, LEM, Category Theory, HoTT, Turing, Euclid, Noether) |
| L2 | `seed_L2_math.json` | 10 | Mathematical frameworks and tools |
| L3 | `seed_L3_methods.json` | 13 | Algorithm core paradigms (search, recursion, DP, adaptive, randomization, feedback, feedforward, model-based, data-driven, spectral, dimensionality reduction, parallelism, multiscale) |
| L4 | `seed_L4_physics.json` | 8 | Physical reality layer (cross-domain) |

## Total: 49 pre-built nodes

## Design Principles

1. **Idea over implementation**: L3 records "adaptive update" as a paradigm, not LMS/RLS/Adam as separate entries. The specific algorithm is a parameter choice within the paradigm.
2. **Mathematics over engineering**: Every node explicitly links to its mathematical substrate (e.g., "Adaptive Update" is linear algebra + gradient descent, regardless of whether it's in control, ML, or biology).
3. **Cross-domain generality**: L0-L4 are usable across control, materials, chemistry, biology, and economics. Examples are drawn from multiple fields to demonstrate universality.
4. **No external dependency**: All seeds are JSON files, loadable without network or AI.
5. **Keyword matching**: Each node has a `keywords` array for automatic paper-to-ontology matching during Phase F structure extraction.
6. **Causal upgrade ready**: All nodes have `node_id` for graph edge creation. L3 methods can be linked to L2 math (e.g., Adaptive Update → Optimization → Gradient Descent) and L4 physics (e.g., Feedback Regulation → Control Loop → Sensor/Actuator Limits).

## Layer Responsibilities

### L0 — Meta (6 nodes)
- **Double-Loop Learning**: framework revision vs parameter tuning
- **Association & Causation Theory**: correlation → causation, 4 constraints, chaos theory tendency
- **Dimensionality Reduction & Information Constraint**: compression as epistemic choice, not just NN distillation
- **Convergence & Stability Theory**: universal to all iterative processes (gradient descent, phase transitions, ecology)
- **Hierarchy & Emergence**: multiscale modeling, timescale separation
- **Epistemic Boundary & Unexplainability**: model limits, auto-revert threshold

### L1 — Axioms (12 nodes)
- **Set Theory & ZFC Foundation**: the ontological basis of all mathematical objects
- **Peano Arithmetic & Induction**: the engine of recursive construction and proof
- **Law of Excluded Middle & Proof Paradigms**: classical vs constructive logic, the Brouwer-Hilbert split
- **Continuum Hypothesis & Cardinality**: Cantor's uncountable, independence from ZFC (Gödel/Cohen)
- **Gödel's Incompleteness & Formal Limits**: the boundary of provability, why meta-theory is necessary
- **Euclid's Parallel Postulate & Geometric Freedom**: the paradigmatic independent axiom, non-Euclidean geometries
- **Category Theory & Structuralism**: objects are secondary to relationships (Eilenberg-Mac Lane)
- **Homotopy Type Theory & Univalent Foundations**: Voevodsky, equivalence as equality, proofs as paths
- **Turing Computability & Effective Procedure**: Church-Turing thesis, halting problem, boundary of mechanical reasoning
- **Symmetry & Conservation (Noether)**: the bridge from mathematics to physical law
- **Linearization & Local Approximation**: Taylor expansion, the epistemological basis of all local analysis
- **Conservation & Closure**: invariants, dissipation, Lyapunov functions, unitarity

### L2 — Math (10 nodes)
- **Function Approximation & Representation**: basis functions, universal approximation, prior encoding
- **Dynamical Systems & Differential Equations**: ODE/PDE, qualitative theory, bifurcation, chaos
- **Optimization Theory**: convex/non-convex, KKT, duality, gradient descent
- **Probability & Statistical Inference**: Bayesian, MLE, Kalman, Monte Carlo, entropy
- **Spectral Analysis & Decomposition**: Fourier, SVD, eigenvalue, PCA, wavelet (same idea: diagonalization)
- **Information Theory & Entropy**: Shannon, mutual information, Kolmogorov, MDL
- **Graph Theory & Network Topology**: adjacency, PageRank, spectral clustering, percolation
- **Algebraic Structure & Group Theory**: symmetry, Lie groups, representation theory, crystal groups
- **Calculus of Variations & Extremal Principles**: Euler-Lagrange, Hamilton, Pontryagin, action principle
- **Topology & Connectivity**: TDA, persistent homology, homotopy, connectedness without metric

### L3 — Methods (13 nodes)
- **Search & Traversal**: the fundamental paradigm of exploring a possibility space (heuristic, Monte Carlo, genetic, simulated annealing)
- **Recursion & Divide-and-Conquer**: breaking problems into subproblems (Master Theorem, FFT, multigrid, hierarchical control)
- **Dynamic Programming & Optimal Substructure**: exploiting sequential decision structure (Bellman equation, MPC, reinforcement learning)
- **Adaptive Update & Online Learning**: incremental parameter correction (LMS/RLS/SGD/Adam are instances, not separate entries)
- **Randomization & Sampling**: using randomness to escape local optima and approximate intractable integrals (Monte Carlo, MCMC, dithering)
- **Feedback & Causal Loop**: self-regulation by acting on error (PID, biology, economics, ecology—all the same structure)
- **Feedforward & Predictive Compensation**: acting on predicted disturbance before it arrives (FxLMS, thermal pre-compensation, anticipatory homeostasis)
- **Model-Based Simulation & Prediction**: digital twin paradigm (MPC, molecular dynamics, CFD, GCM, epidemiology)
- **Data-Driven Function Approximation**: learning mappings from data without physics (neural networks, GP, SVM, kernel methods—universal approximation)
- **Spectral Decomposition & Frequency Analysis**: diagonalization paradigm (Fourier, wavelet, SVD, PCA, modal analysis, phonon DOS)
- **Dimensionality Reduction & Coarse-Graining**: abstraction by discarding detail (PCA, autoencoder, t-SNE, renormalization group, persistent homology)
- **Parallelism & Concurrent Execution**: spatial decomposition for speed (MapReduce, MPI, GPU, FPGA, domain decomposition—Amdahl's law)
- **Multiscale & Multi-Resolution Coupling**: hierarchical modeling across scales (DFT→FEM, fast→slow loops, DNS→LES→RANS, heterogenous multiscale)

### L4 — Physics (8 nodes)
- **Mechanical Oscillation & Wave Dynamics**: vibration, acoustics, modal analysis, resonance
- **Thermal Transport & Thermodynamics**: conduction, convection, radiation, entropy, Carnot
- **Electromagnetic Coupling & Field Theory**: Maxwell, piezoelectric, capacitive, EMI
- **Material Response & Constitutive Relations**: stress-strain, plasticity, viscoelasticity, damage
- **Signal & Noise Floor**: SNR, thermal/shot/flicker noise, quantum limit
- **Spatiotemporal Delay & Causal Boundary**: propagation delay, light cone, phase lag
- **Phase Transition & Critical Phenomena**: melting, bifurcation, critical exponents, universality
- **Structural Stability & Failure**: buckling, fracture, fatigue, creep, homeostasis

## Usage

```python
from lcortex.seeds import SeedLoader, auto_initialize

# Load all seeds
nodes = SeedLoader.load_all("lcortex/seeds/")

# Auto-initialize graph store if empty
stats = auto_initialize(graph_store, "lcortex/seeds/")
```
