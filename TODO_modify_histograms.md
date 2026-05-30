Updated task list for alignment-overhaul branch
-----------------------------------------------

Key conventions
- Apply Sioulas et al. (2024) 5-point weights globally for every field that uses stencil_width=5.
- Local mean B uses weights (1, 4, 6, 4, 1)/16 for direction b̂; do not convert δB to velocity units. b̂ only means B/|B|.

Plan of work
1) Replace 5-point stencil and mean-B kernels everywhere
   - Increment: (1, -4, 6, -4, 1)/sqrt(35) for 5-point differences.
   - Mean B_ell: (1, 4, 6, 4, 1)/16; use b̂ from this B_ell for all ⟂ projections.
2) Alignment angle definitions (~θ existing, θ new)
   - Retain existing cross/product accumulators (~θ).
   - Add ratio accumulators for θ (instantaneous cross/prod) and store new ratio channel(s), e.g., D_Vperp_D_Bperp_cross_MAG_ratio (one per alignment pair).
   - Perp quantities use δB directly (no μ0 scaling), with b̂ from weighted B_ell.
3) Alignment binning modes (CLI selectable, default = ℓ–θ–ϕ)
   - Mode A (default): bin alignment quantities on (ℓ, θ, ϕ) like MAG channels.
   - Mode B: bin alignment quantities on (ℓ) only (current behavior).
   - Add a CLI flag to choose mode; propagate through run_node_analysis → histogram build → saved outputs.
4) Downstream pipeline updates
   - Update combine_histograms_fast (and any combiner) to merge both ~θ and θ ratio channels for both binning modes.
   - Update plotting to read/report both θ and ~θ and handle both binning modes (default ℓ–θ–ϕ; optional ℓ-only).
5) Consistent bins and naming
   - Keep existing channels; add ratio channels alongside. Theta/phi bin edges stay 18×16; product bin edges unchanged. Use clear output naming to distinguish binning mode.

Progress notes
- Unified all histogram channels into a single Channel enum (23 entries) binned on (ℓ, θ, ϕ, Δ) with per-channel Δ edges and a single histogram output saved/combined/plotted end-to-end.

Acceptance notes from clarifications
- Never use lowercase b for “Alfvén units”; b̂ only means the unit vector of B.
- No μ0 scaling; keep δB in field units for alignment/eddy geometry.
