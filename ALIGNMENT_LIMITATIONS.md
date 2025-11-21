## Alignment plots versus ℓ\_∥ / ℓ\_⊥

The current alignment plot (`alignment_angles.png`) uses cross-product histograms that are already summed over all θ and ϕ, so it cannot be restricted to the parallel/perpendicular wedges used in the anisotropic S₂ plots.

To plot alignment angles versus ℓ\_∥ or ℓ\_⊥, we would need angle-resolved cross-product histograms (or equivalent θ/ϕ-tagged data) added to the pipeline (e.g., extend `histograms.py` and downstream saves). Until then, only all-angle alignment is available.
