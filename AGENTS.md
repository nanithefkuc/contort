# AGENTS.md — contort

Contributor guide. For what the library *is* and how to use it, read the
rustdoc; this file covers the things a change can silently break.

## What this crate is

Planned crate for deformed Reed–Solomon codes — twisted, folded, and
interleaved generalized Reed–Solomon and the Roth–Lempel family. Each code is
an ordinary Reed–Solomon evaluation code with a structural deformation applied.

`contort` owns the deformation, not the decoder: the code construction, the
reduction of a received word to a Guruswami–Sudan interpolation problem, and
the admissibility filtering of the returned message polynomials. The
list-decoding machinery lives in `gs-engine`.

**Status:** early R&D. Twisted GRS (`TgrsCode`) and Roth–Lempel
(`RothLempelCode`) are implemented; folded and interleaved are not.

## Invariants (do not break)

- Field arithmetic comes from `fgf`; never hand-roll a field loop. The crate
  root carries `#![forbid(unsafe_code)]` as a consumer of kernel-owning crates.
- The Guruswami–Sudan algorithm — parameter search, interpolation, root
  extraction — belongs to `gs-engine`; never re-implement it here. `contort`
  is an adapter: it feeds `gs-engine` a received word and filters the message
  polynomials it returns by the deformation's admissibility rules.
- `gs-engine` decodes plain Reed–Solomon evaluation codes (`f(α_i)`), to the
  Johnson radius via bivariate Guruswami–Sudan. Column multipliers (GRS `v_i`)
  and the twist/fold structure are applied by this crate before and after the
  decode call, never inside `gs-engine`. Roth–Lempel punctures its exceptional
  last coordinate, decodes the resulting GRS code, and re-encodes candidates to
  check the full Hamming distance — the puncture and re-encode are this crate's.
- Do not put development history in doc comments: no milestone tags, no
  references to superseded designs, no phase numbering.

## Dependencies

- `gs-engine` — Guruswami–Sudan list decoding (parameter search, interpolation,
  Alekhnovich root extraction). Pinned by git revision.
- `fgf` — field arithmetic and packed-element kernels. Pinned to the same
  revision `gs-engine` pins so cargo resolves a single copy.
- `butterfly-fft` — its `ButterflyKernels` trait is the bound `gs-engine`'s
  domain and plan are generic over, so it appears in this crate's public
  signatures. Pinned to the same revision `gs-engine` pins.

Capacity-achieving folded / interleaved decoding needs multivariate
interpolation and root finding that `gs-engine` does not yet expose; those are
an upstream `gs-engine` addition, not a private decoder here.

## Tooling

`just validate` is the PR gate: lint, the dependency allowlist, docs, the
feature matrix, the tier matrix, Miri, and coverage in one run, and the
crate's CI workflow drives the same recipes. The shared recipe surface is
documented once in the umbrella's root `AGENTS.md`; only contort's values
are below.

- `TIERS := 'v3 v2 scalar'`. No kernels live here — these are the backends
  `fgf`'s field kernels and `butterfly-fft`'s transforms dispatch over
  underneath `gs-engine`. `SIMD_BACKEND` resolves once per process, so
  `just test-tiers` and `just cover` re-run the suite pinned to each tier.
- `MIRI` is empty: `#![forbid(unsafe_code)]` leaves nothing to interpret, so
  `just unsafe-check` reports the empty surface and skips.
- Bench targets are `encoder` and `decoder`: `just bench decoder`,
  `just perf-bench decoder 20`.
- `COV_IGNORE` is deliberately empty, so `just cover` holds the whole crate
  to 95% lines while folded and interleaved are unimplemented — close a gap
  with a test, never with an exclusion.
- `justfile` is a byte-identical vendored copy — never edit it here.
  Crate-specific values and recipes belong in `crate.just`.

## Build & test

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```
