# Changelog

All notable changes to contort are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Renamed the crate from `twister` to `contort` and scoped it to the deformed
  Reed–Solomon family (twisted / folded / interleaved GRS, Roth–Lempel).
- Retargeted list decoding onto `gs-engine`; removed the self-contained
  pre-`gs-engine` field and Guruswami–Sudan prototype.
- Added the twisted GRS family: `TgrsCode` encoding and Guruswami–Sudan list
  and unique decoding, filtering ambient GRS candidates by the twist
  constraints (Zhu–Jin). Depends on `gs-engine`, `fgf`, and `butterfly-fft`.
- Added the Roth–Lempel family: `RothLempelCode` encoding and Guruswami–Sudan
  list and unique decoding, puncturing the exceptional coordinate to a GRS code
  and re-encoding candidates for a full Hamming-distance check (Zhu–Jin).
- Promoted the per-family error to a shared `Error` and shared the
  `UniqueDecode` outcome across families.
- Made warmed decoding allocation-free: both decoders gained a
  `prepare_scratch` method, list decoding overwrites and truncates retained
  output polynomials instead of clearing and re-pushing, and unique decoding
  filters through scratch-owned storage. Proven by a counting-allocator
  integration test (`tests/zero_alloc.rs`) over a multi-candidate word.
- Replaced the twisted GRS decoder's nested per-degree twist tables with a
  flat compressed-row constraint layout.
- Roth–Lempel decoding no longer re-evaluates candidates: it consumes
  `gs-engine`'s scored decode (`decode_scored_into`), reading each candidate's
  exact punctured distance and adding only the one-symbol exceptional mismatch
  to obtain the full distance.
- Bumped the `gs-engine` pin to the revision exposing scored decode candidates.
- Added a `criterion` benchmark harness (`benches/encoder.rs`,
  `benches/decoder.rs`) and recorded steady-state baselines in `BENCHMARKS.md`.
- Added the folded Reed–Solomon construction (`FoldedRsCode`): multiplicative
  orbit geometry with order validation, an allocation-free block-major encoder,
  and the block-Hamming metric. Construction only; the capacity list decoder is
  a future upstream capability.
- Added the homogeneous interleaved Reed–Solomon construction
  (`InterleavedRsCode`): column-major encoding of `ℓ` independent rows over a
  shared domain and the column-Hamming metric. Construction only; the
  collaborative decoder is a future upstream capability.
- Added the punctured GRS family (`PuncturedGrsCode`): deleting a coordinate set
  `S` yields the GRS code on the surviving points, decoded by one Guruswami–Sudan
  plan on the subdomain with no filter. Carries the block-alignment capability
  flag for composed folds.
- Added the Möbius-transformed GRS family (`MobiusGrsCode`): a projective-linear
  relabelling `φ(x) = (a·x + b)/(c·x + d)` of the evaluation points, decoded by
  either the moved-point plan on `β = φ(α)` or the normalized-multiplier plan on
  `α` followed by the inverse-map pullback — the two routes return identical
  lists. Singular maps and a pole on the domain are rejected; the map exposes the
  orbit-preservation flag.
- Generalized Roth–Lempel to the extended GRS family (`ExtendedGrsCode`):
  appending coordinates given by arbitrary linear functionals on the message
  coefficients, including the projective evaluation-at-infinity coordinate.
  `RothLempelCode` is now the single-functional `f_{k-2} + δ·f_{k-1}` instance of
  this engine, decoded by the shared puncture / Guruswami–Sudan / re-encode
  reduction with no duplicated scorer.
- Added the internals-gated canonical transform descriptor over Twist, Möbius,
  Puncture, Extend, Fold, and Interleave. L1–L4 normalization composes maps,
  unions/cancels domain edits, merges twist collisions, and multiplies grouping
  parameters; the versioned descriptor is no larger than its generating word
  and strictly contracts law-absorbing words. Frozen wire bytes, concrete-family
  encoder equality, composed encoder/decoder equality, capability downgrades,
  checked overflow, and allocation-free descriptor encoding are tested.