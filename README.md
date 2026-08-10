> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# twister

Planned crate for twisted and folded generalized Reed-Solomon codes
(Roth-Lempel and related families), with capacity list decoding. It absorbs
the folded Reed-Solomon decoder into the deformed-GRS family.

**Status:** early R&D. No public API yet.

## Build

```sh
cargo build
cargo test --all-features
```

## Minimum supported Rust version

1.89, edition 2024.

## License

MIT. See [LICENSE](LICENSE).
