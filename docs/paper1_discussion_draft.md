# Discussion — Draft for Paper #1

> **Working title (context):** Inverse design of continuous auxetic metamaterials via WIRE INR and KAN.
> **Scope of this draft:** Discussion section only, built on the verified 2026-08-23 numbers
> (`task_progress.md`). All figures below are re-measured with the current
> evaluation pipeline (`property_accuracy()`, same test set, surrogate v2) unless stated otherwise.
> **Status:** draft for review — numbers must be re-confirmed with one additional seed run before submission.

---

## Discussion

### The reconstruction–fidelity paradox: why `val_loss` is the wrong oracle for mechanics

The single most consequential methodological observation of this study is that, on a KAN-based
conditional VAE, reconstruction quality and physical-property fidelity are **anti-correlated**.
The KAN checkpoint that achieved the best reconstruction loss of all runs (`val ≈ 774`, lr 5e-4)
yielded a macroscopic property coefficient of determination of **R² = −1.34** — worse than
predicting the constant mean — while the checkpoint selected by periodically measured
finite-element R² (`--select-by fe_r2`) reached R² = 0.64. In other words, a model that "looks
best" under the probabilistic loss provably collapses as a mechanical-property predictor.

We attribute this to the expressiveness of B-spline activations. KAN layers replace fixed
nonlinearities with learnable splines whose local degrees of freedom can absorb reconstruction
error with high precision while leaving the latent representation unaligned with any physical
meaning. The variational loss, dominated by reconstruction and KL terms, provides no gradient
signal that distinguishes "a structure that reconstructs well" from "a structure whose effective
properties are correct". For mechanics-sensitive tasks, the model-selection oracle must therefore
be a physical metric, not a likelihood-based one. This finding extends and sharpens earlier
evidence in our pipeline that `val_loss` is unsafe for cVAE checkpointing under annealed KL
weights (gamma > 0), and generalizes it to the KAN architecture where the failure is far more
severe (R² = −1.34 vs. a merely premature early-epoch checkpoint).

### KAN is not self-sufficient: real-physics supervision is a mandatory filter

Replacing the Linear regression head of the Co-VAE with KAN layers alone produced R² = 0.19 —
*below* the already modest Linear baseline (0.35). Only when the frozen-FE physics loss
(`real_physics_prior_loss`, an analytic FE solve inside the training loop) was added did KAN
surpass the baseline, in a two-stage schedule: base KAN → real-physics fine-tune at λ = 2.0
reached R² = 0.64 (ν₁₂ R² = 0.82, ν₂₁ R² = 0.45) with a peak true-FE R² of **0.8889** — a
near-2× improvement over the re-measured Linear finetuned baseline (0.35). The physics loss acts
as a filter that steers the spline degrees of freedom toward physically admissible manifolds;
without it, the spline's freedom manifests as property collapse rather than as improved
expressiveness. We interpret this as evidence that, in small-data generative design settings,
architectural capacity (KAN) and physical supervision are complements, not substitutes: capacity
amplifies whatever signal the training objective provides, for better or worse.

### Reproducibility of baselines: re-measurement changed the comparison frame

Our initially reported baseline of R² ≈ 0.85 (July evaluation report) **did not reproduce** with
the current code: re-measuring the identical checkpoint (`cvae_v2_finetuned.pt`) on the identical
test set with the current evaluation pipeline yields R² = 0.35 (surrogate v2) and −1.81 with the
surrogate version used by the original report. We could not trace the discrepancy to a functional
change in either the evaluation routine or the forward pass for the relevant configuration; the
most plausible explanation is that the July figures were produced by a different code path that no
longer exists. The practical consequence is that every architecture comparison in this paper is
performed with one frozen evaluation protocol, and we encourage the community to treat
historical headline numbers with caution unless re-measured end-to-end. This episode
simultaneously illustrates the value of version-pinned evaluation pipelines for ML-driven
materials design.

### Interpretation of per-property results

The asymmetry between ν₁₂ (R² = 0.82) and ν₂₁ (R² = 0.45) warrants discussion. Both quantities
derive from the same homogenized stiffness tensor, yet their predictability differs markedly.
We hypothesize that the imbalance reflects dataset-level label asymmetry: the anisotropic seeds
used in this study (hourglass, hexagonal, reentrant bow-tie) produce a narrower and more
clustered ν₂₁ distribution, which is intrinsically harder for a regression head to resolve and is
also less informative under R², which is sensitive to target variance. A pragmatic implication
for the inverse-design loop is that optimization targets should be formulated primarily on ν₁₂,
with ν₂₁ treated as a constraint, and that future data generation should deliberately enrich the
ν₂₁ range.

### Limitations

- The headline figures (0.64 / 0.8889) come from a single round-2 fine-tune; one additional
  seed run is required before these numbers are used in the final manuscript.
- The surrogate v1/v2 discrepancy on the same checkpoint (−1.81 vs 0.35) remains unexplained at
  the root-cause level and is documented as a known caveat.
- True-FE R² reported here is the *best during training*, evaluated periodically; reported
  end-of-training property R² refers to the surrogate-based `property_accuracy` protocol. The
  distinction is stated explicitly in the tables to avoid conflating the two.
- ν₂₁ predictability (0.45) sets a floor on achievable joint-fidelity claims.

### Implications and outlook

For the mechanics-informed generative modeling community, our results suggest three actionable
rules: (i) select generative checkpoints by physical fidelity metrics rather than reconstruction
loss; (ii) treat high-capacity regression heads as requiring physical supervision, not as
drop-in replacements; and (iii) version-pin evaluation pipelines so that architecture
comparisons survive code evolution. Within this project, the combination of a WIRE INR decoder
(resolution-agnostic continuous generation) with the physics-filtered KAN Co-VAE and a
KAN-based surrogate is expected to enable inverse design that exceeds the historical OOD
saturation of the raw cVAE; validation of that closed loop is the immediate next step.

---

## Figures cited in this draft (verified values)

| Metric | Linear base | Linear finetuned | KAN (val_loss) | KAN + Real-Physics v1 | KAN + Real-Physics v2 |
|---|---|---|---|---|---|
| Macro R² | 0.29 | 0.35 | 0.19 | 0.58 | **0.64** |
| ν₁₂ R² | 0.27 | 0.24 | 0.35 | 0.76 | **0.82** |
| ν₂₁ R² | 0.30 | 0.47 | 0.02 | 0.41 | 0.45 |
| Best true-FE R² | — | — | — | 0.7865 | **0.8889** |
| Property R² under val_loss selection | — | — | **−1.34** | — | — |
