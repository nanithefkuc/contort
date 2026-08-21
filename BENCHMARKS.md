# Benchmarks

Measured steady-state numbers for `contort`. Every figure here was produced by
the committed `criterion` harness on warmed, allocation-free scratch (the same
steady state `tests/zero_alloc.rs` proves). Reasoned or estimated numbers do
not belong in this file; a doc comment states a decision and points here.

## Environment

| Field | Value |
| --- | --- |
| CPU | Intel Core Ultra 7 258V |
| Target | `x86_64-unknown-linux-gnu` |
| Toolchain | `rustc 1.93.0` |
| Features | `std,simd` (default) |
| `gs-engine` | rev `b845adf9dd1be5ae64a15829ce0c615302745172` |
| `fgf` | rev `d0e331cec2e6ae5645e529597fa913db467a44cd` |
| `butterfly-fft` | rev `6b3f52485bfff2664d6b0c92452137e618a60932` |

Command (compact sampling; raise the budget for release-quality numbers):

```sh
cargo bench --bench encoder -- --warm-up-time 0.3 --measurement-time 1 --sample-size 20
cargo bench --bench decoder -- --warm-up-time 0.3 --measurement-time 1 --sample-size 20
```

Reported values are the criterion median estimate.

## Encode — `encode_into`, `k = 2`, one twist / `δ`

| Family | Field | `n` | Median |
| --- | --- | --- | --- |
| Twisted GRS | GF(2^8) | 8 | 46.9 ns |
| Roth–Lempel | GF(2^8) | 8 | 34.9 ns |
| Twisted GRS | GF(2^4) | 16 | 147.7 ns |
| Roth–Lempel | GF(2^4) | 16 | 100.5 ns |

Horner encode (`Polynomial::evaluate_many`); the prepared domain-aware fast path
is optimization #3 and is not yet in the tree.

## List decode — `list_decode_into`, warmed scratch and output

| Family | Field | `(n, k, τ)` | Median |
| --- | --- | --- | --- |
| Twisted GRS | GF(2^8) | (8, 2, 3) | 1.58 µs |
| Roth–Lempel | GF(2^8) | (8, 2, 3) | 2.70 µs |
| Twisted GRS | GF(2^4) | (16, 2, 6) | 4.30 µs |
| Roth–Lempel | GF(2^4) | (16, 2, 6) | 3.42 µs |

## Unique decode — `unique_decode`, warmed scratch

| Family | Field | `(n, k, τ)` | Median |
| --- | --- | --- | --- |
| Twisted GRS | GF(2^8) | (8, 2, 3) | 1.60 µs |
| Roth–Lempel | GF(2^8) | (8, 2, 3) | 2.72 µs |
| Twisted GRS | GF(2^4) | (16, 2, 6) | 4.30 µs |
| Roth–Lempel | GF(2^4) | (16, 2, 6) | 3.42 µs |

The list and unique paths are within noise of each other: the unique collapse
adds only a bounded scan over the already-filtered candidates. The decode cost
is dominated by the ambient Guruswami–Sudan call in `gs-engine`, as expected;
`contort`'s normalization, twist filter, and exceptional check are a small
constant on top.
