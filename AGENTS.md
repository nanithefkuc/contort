# AGENTS.md — twister

Contributor guide. For what the library *is* and how to use it, read the
rustdoc; this file covers the things a change can silently break.

## What this crate is

Planned crate for twisted and folded generalized Reed-Solomon codes
(Roth-Lempel and related families), with capacity list decoding. It absorbs
the folded Reed-Solomon decoder (formerly `power-decoder`) into the
deformed-GRS family.

**Status:** early R&D. No public API. The source modules are private and
`#[allow(dead_code)]` until the decoding surface is settled.

## Invariants (do not break)

- Field arithmetic comes from `fgf`; never hand-roll a field loop. When the
  crate gains a public API, it will carry `#![forbid(unsafe_code)]` as a
  consumer of kernel-owning crates.
- Do not put development history in doc comments: no milestone tags, no
  references to superseded designs, no phase numbering.

## Build & test

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```