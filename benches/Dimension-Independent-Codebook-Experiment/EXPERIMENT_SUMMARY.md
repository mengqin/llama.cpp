# PQ/TQ Independent Codebook Experiment Summary

## Goal

Evaluate whether the PQ/TQ KV-cache path can be improved by:

- introducing D-specific codebooks for `D64 / D128 / D256`
- introducing D-aware QJL correction
- keeping the rest of the PQ/TQ implementation structure intact

## What Was Tried

- Replaced the original unified codebook path with D-specific codebooks.
- Removed the old `D64` compensation shell that existed to adapt `D64` data to the unified `D128` codebook domain.
- Added D-aware QJL correction experiments.
- Added D-specific fp16-convert / FA prefill support needed to make those experiments mathematically consistent on GPU.

## What Was Confirmed

- The old `D128 4-bit` table is still the better table for the current engineering path.
  - Reverting only `D128 4-bit` immediately restored the old `pq4/tq4` results.
- `D64 native 2-bit` did not improve quality.
  - `pq2/tq2` became worse or stayed unstable.
- `D256` native codebooks showed only very small gains, not enough to justify the added complexity.
- D-aware QJL did not produce a clean global win.
  - Some cases improved.
  - Other cases regressed.
  - The result was not consistent enough for a new baseline.

## Final Decision

This experiment is treated as a failed baseline replacement.

The code was rolled back to the original, validated behavior:

- unified codebook semantics
- original `D64` WHT / residual / `rnorm` compensation path
- original QJL semantics
- old `D128 4-bit` table

## Practical Conclusion

For this codebase and test matrix, the simpler original implementation is better:

- more stable
- easier to reason about
- closer to the previously validated results

If D-specific codebooks are revisited later, they should be tested as much smaller, isolated experiments instead of replacing the whole baseline at once.
