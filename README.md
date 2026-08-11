> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# contort - Deformed Reed-Solomon Codes

`contort` builds deformed Reed–Solomon codes: codes obtained by applying a
structural deformation — a twist, a fold, or an interleave — to an ordinary
Reed–Solomon evaluation code, yielding a new and often non-GRS code.

The crate owns the deformation: code construction, the reduction of a received
word to a Guruswami–Sudan problem, and the coefficient filter or re-encode check
that recovers the deformed code's messages. List decoding is delegated to
[`gs-engine`](https://github.com/nanithefkuc/gs-engine) (parameter search,
interpolation, Alekhnovich root extraction); field arithmetic and vector kernels
come from [`fgf`](https://github.com/nanithefkuc/fgf); evaluation-domain plans
come from [`butterfly-fft`](https://github.com/nanithefkuc/butterfly-fft).

## Usage

The MSRV is Rust 1.89, edition 2024.

`contort` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
contort = { git = "https://github.com/nanithefkuc/contort" }
```

Portable `no_std` builds are available; they use `alloc`:

```toml
[dependencies]
contort = { git = "https://github.com/nanithefkuc/contort", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | standard-library errors and runtime-dispatched `gs-engine`/`fgf` kernels |
| `std` without `simd` | portable kernels |
| `internals` | unstable implementation APIs for benchmarking and research |
| `--no-default-features` | `no_std` plus `alloc`, portable kernels |

### Supported fields

Decoding runs over the binary extension fields `fgf::Gf8` and `fgf::Gf16` — the
fields `gs-engine` validates. Evaluation points are canonical field elements and
column multipliers are nonzero field elements; adapters normalize external bytes
before calling in.

## Twisted GRS

A twisted GRS code adds `ℓ` twist terms `η_j · f_{h_j} · x^{k-1+t_j}` to a
degree-`< k` message polynomial before GRS evaluation, producing a non-GRS code
of pseudo-dimension `k' = k + max_j t_j`. Decoding runs Guruswami–Sudan on the
ambient `[n, k']` GRS code and keeps the candidates satisfying the twist
coefficient constraints (Zhu–Jin).

```rust
use contort::{AlekhnovichLimits, EvaluationDomain, ParameterLimits, TgrsCode, TgrsScratch, Twist};
use fgf::Gf8;
use fgf::gf8::Elem;

// [8, 2] twisted GRS with one twist η=2 at offset t=1, hook h=0.
let domain = EvaluationDomain::<Gf8>::additive_subspace(8).expect("domain");
let code = TgrsCode::new(domain, vec![Elem::ONE; 8], 2, vec![Twist::new(1, 0, Elem(2))])
    .expect("code");

let mut codeword = vec![Elem::ZERO; code.length()];
code.encode_into(&[Elem(5), Elem(9)], &mut codeword).expect("encode");

let decoder = code
    .list_decoder(
        2,
        ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
        AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128),
    )
    .expect("decoder");
let mut scratch = TgrsScratch::new();
let mut messages = Vec::new();
decoder.list_decode_into(&codeword, &mut scratch, &mut messages).expect("decode");
assert!(messages.iter().any(|m| m.coefficient(0) == Elem(5) && m.coefficient(1) == Elem(9)));
```

## Roth–Lempel

A Roth–Lempel code evaluates a degree-`< k` message polynomial as a GRS code
over `n-1` points and adds an exceptional final coordinate
`v_n · (f_{k-2} + δ · f_{k-1})`. Puncturing that coordinate yields a GRS code, so
decoding runs Guruswami–Sudan on the punctured code, re-encodes each candidate to
a full Roth–Lempel codeword, and keeps those within the decoding radius
(Zhu–Jin). There are few public Roth–Lempel implementations.

```rust
use contort::{AlekhnovichLimits, EvaluationDomain, ParameterLimits, RothLempelCode, RothLempelScratch};
use fgf::Gf8;
use fgf::gf8::Elem;

// [8, 2] Roth–Lempel with twist δ=3 over a 7-point evaluation domain.
let points: Vec<Elem> = (1..=7u8).map(Elem).collect();
let domain = EvaluationDomain::<Gf8>::arbitrary(points).expect("domain");
let multipliers: Vec<Elem> = (1..=8u8).map(Elem).collect();
let code = RothLempelCode::new(domain, multipliers, 2, Elem(3)).expect("code");

let mut codeword = vec![Elem::ZERO; code.length()];
code.encode_into(&[Elem(6), Elem(31)], &mut codeword).expect("encode");

let decoder = code
    .unique_decoder(
        ParameterLimits::new(8, 16, usize::MAX, usize::MAX),
        AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128),
    )
    .expect("decoder");
let mut scratch = RothLempelScratch::new();
let decoded = decoder.unique_decode(&codeword, &mut scratch).expect("decode");
assert!(decoded.is_unique());
```

## Roadmap

- [x] Twisted GRS (`TgrsCode`) — encode, list decode, unique decode
- [x] Roth–Lempel (`RothLempelCode`) — encode, list decode, unique decode
- [ ] Folded Reed–Solomon
- [ ] Interleaved Reed–Solomon

## Building

`contort` builds on stable Rust (edition 2024, MSRV 1.89); the `gs-engine`,
`fgf`, and `butterfly-fft` SIMD kernels are selected at runtime:

```sh
cargo build                        # default: std + simd
cargo build --no-default-features  # portable no_std + alloc
cargo test --all-features
```

## License

MIT - see [LICENSE](LICENSE)
