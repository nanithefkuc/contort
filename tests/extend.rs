//! End-to-end extended-GRS decoding checked against a brute-force Hamming-ball
//! oracle over GF(2^8), plus an MDS distance spot-check. Steady-state
//! allocation behaviour is proven in `tests/zero_alloc.rs`.
//!
//! Each code has a base evaluation domain of `n_base` points, dimension
//! `k = 2`, and `L` extension functionals, so the oracle enumerates every
//! message, encodes the full `n = n_base + L`-symbol codeword (base and
//! extended coordinates), and keeps those within the decoding radius. The
//! decoder's list must equal that set, and the unique decoder must agree with
//! the ball's cardinality.

use contort::{
    Error, ExtendedGrsCode, ExtendedGrsDecoder, ExtendedGrsScratch, TgrsCode, TgrsScratch, Twist,
    UniqueDecode,
};
use fgf::Gf8B;
use fgf::gf8b::Elem;
use gs_engine::{EvaluationDomain, ParameterLimits};
use poly_ring::{AlekhnovichLimits, Polynomial};

fn e(byte: u8) -> Elem {
    Elem::from_raw(byte)
}

fn parameter_limits() -> ParameterLimits {
    ParameterLimits::new(8, 16, usize::MAX, usize::MAX)
}

fn root_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128)
}

/// `n` multipliers `1, 2, …, n` — all nonzero.
fn ramp_multipliers(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

/// An `n_base`-point arbitrary evaluation domain over distinct nonzero
/// elements.
fn base_domain(n_base: usize) -> EvaluationDomain<Gf8B> {
    let points: Vec<Elem> = (1..=n_base as u8).map(e).collect();
    EvaluationDomain::<Gf8B>::arbitrary(points).unwrap()
}

/// The unit functional `e_i`: coefficient `1` at index `i`, else `0`.
fn unit(k: usize, i: usize) -> Vec<Elem> {
    let mut f = vec![Elem::ZERO; k];
    f[i] = Elem::ONE;
    f
}

fn hamming(a: &[Elem], b: &[Elem]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every message in `GF(2^8)^k` whose codeword is within `tau` of `received`,
/// as sorted message-coefficient vectors.
fn brute_force_ball(code: &ExtendedGrsCode<Gf8B>, received: &[Elem], tau: usize) -> Vec<Vec<Elem>> {
    let k = code.dimension();
    let n = code.length();
    let mut ball = Vec::new();
    let mut message = vec![Elem::ZERO; k];
    let mut codeword = vec![Elem::ZERO; n];
    let mut counter = vec![0u16; k];
    loop {
        for (slot, &value) in message.iter_mut().zip(counter.iter()) {
            *slot = e(value as u8);
        }
        code.encode_into(&message, &mut codeword).unwrap();
        if hamming(&codeword, received) <= tau {
            ball.push(message.clone());
        }
        let mut position = 0;
        let mut overflow = true;
        while position < k {
            counter[position] += 1;
            if counter[position] == 256 {
                counter[position] = 0;
                position += 1;
            } else {
                overflow = false;
                break;
            }
        }
        if overflow {
            break;
        }
    }
    ball.sort();
    ball
}

fn decoded_messages(candidates: &[Polynomial<Gf8B>], k: usize) -> Vec<Vec<Elem>> {
    let mut messages: Vec<Vec<Elem>> = candidates
        .iter()
        .map(|poly| (0..k).map(|d| poly.coefficient(d)).collect())
        .collect();
    messages.sort();
    messages
}

fn check_against_oracle(
    code: &ExtendedGrsCode<Gf8B>,
    decoder: &ExtendedGrsDecoder<Gf8B>,
    received: &[Elem],
) {
    let mut scratch = ExtendedGrsScratch::new();
    let mut candidates = Vec::new();
    decoder
        .list_decode_into(received, &mut scratch, &mut candidates)
        .unwrap();

    let decoded = decoded_messages(&candidates, code.dimension());
    let oracle = brute_force_ball(code, received, decoder.target_radius());
    assert_eq!(
        decoded, oracle,
        "list decode disagreed with brute-force ball"
    );

    let unique = decoder.unique_decode(received, &mut scratch).unwrap();
    match oracle.len() {
        0 => assert!(matches!(unique, UniqueDecode::NoCandidate)),
        1 => {
            let message = unique.message().expect("unique message");
            let coefficients: Vec<Elem> = (0..code.dimension())
                .map(|d| message.coefficient(d))
                .collect();
            assert_eq!(coefficients, oracle[0]);
        }
        _ => assert!(matches!(unique, UniqueDecode::Ambiguous)),
    }
}

/// Codeword-derived words with 0..=`max_errors` errors, plus words that corrupt
/// only extended coordinates and arbitrary words.
fn probe_words(
    code: &ExtendedGrsCode<Gf8B>,
    message: &[Elem],
    max_errors: usize,
) -> Vec<Vec<Elem>> {
    let n = code.length();
    let base = code.base_length();
    let mut codeword = vec![Elem::ZERO; n];
    code.encode_into(message, &mut codeword).unwrap();

    let mut words = Vec::new();
    for errors in 0..=max_errors {
        let mut word = codeword.clone();
        for position in 0..errors {
            let index = (position * 3 + 1) % n;
            word[index] = word[index].add(e((position + 1) as u8));
        }
        words.push(word);
    }
    // Corrupt each extended coordinate on its own.
    for j in base..n {
        let mut word = codeword.clone();
        word[j] = word[j].add(e((j + 1) as u8));
        words.push(word);
    }
    // A base error plus an extended-coordinate error.
    let mut mixed = codeword.clone();
    mixed[0] = mixed[0].add(e(9));
    mixed[n - 1] = mixed[n - 1].add(e(4));
    words.push(mixed);
    // Arbitrary patterns.
    words.push((0..n).map(|i| e((5 * i + 3) as u8)).collect());
    words.push((0..n).map(|_| Elem::ZERO).collect());
    words
}

#[test]
fn projective_matches_oracle() {
    let n_base = 7;
    let k = 2;
    let code =
        ExtendedGrsCode::projective(base_domain(n_base), ramp_multipliers(n_base + 1), k).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(6), e(31)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn projective_matches_mobius_route() {
    // The Möbius map φ(x) = 1/x sends {α_i} ∪ {∞} to {1/α_i} ∪ {0}, so the
    // projective extension is codeword-for-codeword equal to the plain GRS on
    // those points evaluating g = reverse(f), with multipliers v_i·α_i^{k-1} on
    // the moved points and v_∞ on the point 0. Decoding that GRS (a twist-free
    // TgrsCode) and reversing each candidate's coefficients must reproduce the
    // puncture-route list exactly.
    let n_base = 6;
    let k = 2;
    let multipliers = ramp_multipliers(n_base + 1);
    let projective =
        ExtendedGrsCode::projective(base_domain(n_base), multipliers.clone(), k).unwrap();
    let radius = 2;
    let puncture_decoder = projective
        .list_decoder(radius, parameter_limits(), root_limits())
        .unwrap();

    let alpha: Vec<Elem> = (1..=n_base as u8).map(e).collect();
    let mut moved_points: Vec<Elem> = alpha.iter().map(|&a| a.inv()).collect();
    moved_points.push(Elem::ZERO);
    let mut moved_multipliers: Vec<Elem> = alpha
        .iter()
        .zip(multipliers.iter())
        .map(|(&a, &v)| v.mul(a.pow((k - 1) as u64)))
        .collect();
    moved_multipliers.push(multipliers[n_base]);
    let equivalent = TgrsCode::new(
        EvaluationDomain::arbitrary(moved_points).unwrap(),
        moved_multipliers,
        k,
        Vec::<Twist<Gf8B>>::new(),
    )
    .unwrap();
    let mobius_decoder = equivalent
        .list_decoder(radius, parameter_limits(), root_limits())
        .unwrap();

    let mut puncture_scratch = ExtendedGrsScratch::new();
    let mut mobius_scratch = TgrsScratch::new();
    let mut puncture_out = Vec::new();
    let mut mobius_out = Vec::new();

    for word in probe_words(&projective, &[e(5), e(9)], 3) {
        puncture_decoder
            .list_decode_into(&word, &mut puncture_scratch, &mut puncture_out)
            .unwrap();
        mobius_decoder
            .list_decode_into(&word, &mut mobius_scratch, &mut mobius_out)
            .unwrap();
        let mut from_mobius: Vec<Vec<Elem>> = mobius_out
            .iter()
            .map(|poly| (0..k).rev().map(|d| poly.coefficient(d)).collect())
            .collect();
        from_mobius.sort();
        assert_eq!(
            decoded_messages(&puncture_out, k),
            from_mobius,
            "projective puncture route disagreed with the Möbius route"
        );
    }
}

#[test]
fn multi_functional_matches_oracle() {
    let n_base = 7;
    let k = 2;
    // λ_0 = e_{k-1} (evaluation at infinity); λ_1 = e_{k-2} + δ·e_{k-1}.
    let mut lambda1 = unit(k, k - 2);
    lambda1[k - 1] = e(3);
    let functionals = vec![unit(k, k - 1), lambda1];
    let n = n_base + functionals.len();
    let code =
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n), k, functionals).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(200), e(1)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn projective_distance_is_mds() {
    let n_base = 7;
    let k = 2;
    let code =
        ExtendedGrsCode::projective(base_domain(n_base), ramp_multipliers(n_base + 1), k).unwrap();
    let n = code.length();

    let mut min_weight = usize::MAX;
    let mut message = vec![Elem::ZERO; k];
    let mut codeword = vec![Elem::ZERO; n];
    let mut counter = vec![0u16; k];
    loop {
        for (slot, &value) in message.iter_mut().zip(counter.iter()) {
            *slot = e(value as u8);
        }
        let nonzero_message = message.iter().any(|&s| s != Elem::ZERO);
        if nonzero_message {
            code.encode_into(&message, &mut codeword).unwrap();
            let weight = codeword.iter().filter(|&&s| s != Elem::ZERO).count();
            min_weight = min_weight.min(weight);
        }
        let mut position = 0;
        let mut overflow = true;
        while position < k {
            counter[position] += 1;
            if counter[position] == 256 {
                counter[position] = 0;
                position += 1;
            } else {
                overflow = false;
                break;
            }
        }
        if overflow {
            break;
        }
    }

    assert_eq!(
        min_weight,
        n - k + 1,
        "projective [{n}, {k}] code is not MDS"
    );
}

#[test]
fn corrupts_each_extended_coordinate() {
    let n_base = 7;
    let k = 2;
    let mut lambda1 = unit(k, k - 2);
    lambda1[k - 1] = e(5);
    let functionals = vec![unit(k, k - 1), lambda1];
    let n = n_base + functionals.len();
    let code =
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n), k, functionals).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    let message = [e(42), e(17)];
    let mut codeword = vec![Elem::ZERO; n];
    code.encode_into(&message, &mut codeword).unwrap();

    // Corrupt exactly one extended coordinate at a time and confirm the decoder
    // still matches the oracle and recovers the message uniquely.
    for j in n_base..n {
        let mut received = codeword.clone();
        received[j] = received[j].add(e(1));
        check_against_oracle(&code, &decoder, &received);

        let mut scratch = ExtendedGrsScratch::new();
        let mut candidates = Vec::new();
        decoder
            .list_decode_into(&received, &mut scratch, &mut candidates)
            .unwrap();
        let decoded = decoded_messages(&candidates, k);
        assert!(
            decoded.contains(&message.to_vec()),
            "message lost after corrupting extended coordinate {j}"
        );
    }
}

#[test]
fn construction_rejects_bad_parameters() {
    let n_base = 7;
    let k = 2;
    let l = 2;
    let n = n_base + l;
    let good = || vec![unit(k, k - 1), unit(k, k - 2)];

    // Functional of the wrong length.
    let bad_functionals = vec![unit(k, k - 1), vec![Elem::ONE; k + 1]];
    assert_eq!(
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n), k, bad_functionals)
            .unwrap_err(),
        Error::FunctionalLength {
            expected: k,
            got: k + 1
        }
    );

    // Dimension exceeding the base length.
    assert_eq!(
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n), n_base + 1, good())
            .unwrap_err(),
        Error::InvalidDimension {
            dimension: n_base + 1,
            length: n_base
        }
    );

    // Zero dimension.
    assert_eq!(
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n), 0, good()).unwrap_err(),
        Error::InvalidDimension {
            dimension: 0,
            length: n_base
        }
    );

    // Multiplier-count mismatch.
    assert_eq!(
        ExtendedGrsCode::new(base_domain(n_base), ramp_multipliers(n - 1), k, good()).unwrap_err(),
        Error::MultiplierCount {
            expected: n,
            got: n - 1
        }
    );

    // Zero multiplier.
    let mut multipliers = ramp_multipliers(n);
    multipliers[3] = Elem::ZERO;
    assert_eq!(
        ExtendedGrsCode::new(base_domain(n_base), multipliers, k, good()).unwrap_err(),
        Error::ZeroMultiplier { index: 3 }
    );

    // Projective rejects zero dimension.
    assert_eq!(
        ExtendedGrsCode::projective(base_domain(n_base), ramp_multipliers(n_base + 1), 0)
            .unwrap_err(),
        Error::InvalidDimension {
            dimension: 0,
            length: n_base
        }
    );
}

#[test]
fn io_length_checks() {
    let n_base = 7;
    let k = 2;
    let code =
        ExtendedGrsCode::projective(base_domain(n_base), ramp_multipliers(n_base + 1), k).unwrap();
    let n = code.length();

    let mut codeword = vec![Elem::ZERO; n];
    assert_eq!(
        code.encode_into(&[e(1)], &mut codeword).unwrap_err(),
        Error::MessageLength {
            expected: k,
            got: 1
        }
    );
    let mut short = vec![Elem::ZERO; n - 1];
    assert_eq!(
        code.encode_into(&[e(1), e(2)], &mut short).unwrap_err(),
        Error::CodewordLength {
            expected: n,
            got: n - 1
        }
    );

    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    let mut scratch = ExtendedGrsScratch::new();
    let mut candidates = Vec::new();
    assert_eq!(
        decoder
            .list_decode_into(&vec![Elem::ZERO; n + 1], &mut scratch, &mut candidates)
            .unwrap_err(),
        Error::ReceivedLength {
            expected: n,
            got: n + 1
        }
    );
}
