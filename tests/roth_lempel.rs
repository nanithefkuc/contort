//! End-to-end Roth–Lempel decoding checked against a brute-force Hamming-ball
//! oracle over GF(2^8).
//!
//! The code has length `n`, an `n-1`-point evaluation domain, and dimension
//! `k = 2`, so the oracle enumerates every message, encodes the full
//! Roth–Lempel codeword (including the exceptional last coordinate), and keeps
//! those within the decoding radius. The decoder's list must equal that set,
//! and the unique decoder must agree with the ball's cardinality.

use contort::{
    AlekhnovichLimits, Error, EvaluationDomain, ParameterLimits, Polynomial, RothLempelCode,
    RothLempelDecoder, RothLempelScratch, UniqueDecode,
};
use fgf::Gf8;
use fgf::gf8::Elem;

fn e(byte: u8) -> Elem {
    Elem(byte)
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

/// An `n-1`-point arbitrary evaluation domain over distinct nonzero elements.
fn punctured_domain(n: usize) -> EvaluationDomain<Gf8> {
    let points: Vec<Elem> = (1..n as u8).map(e).collect();
    EvaluationDomain::<Gf8>::arbitrary(points).unwrap()
}

fn hamming(a: &[Elem], b: &[Elem]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn brute_force_ball(code: &RothLempelCode<Gf8>, received: &[Elem], tau: usize) -> Vec<Vec<Elem>> {
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

fn decoded_messages(candidates: &[Polynomial<Gf8>], k: usize) -> Vec<Vec<Elem>> {
    let mut messages: Vec<Vec<Elem>> = candidates
        .iter()
        .map(|poly| (0..k).map(|d| poly.coefficient(d)).collect())
        .collect();
    messages.sort();
    messages
}

fn check_against_oracle(
    code: &RothLempelCode<Gf8>,
    decoder: &RothLempelDecoder<Gf8>,
    received: &[Elem],
) {
    let mut scratch = RothLempelScratch::new();
    let mut candidates = Vec::new();
    decoder
        .list_decode_into(received, &mut scratch, &mut candidates)
        .unwrap();

    let decoded = decoded_messages(&candidates, code.dimension());
    let oracle = brute_force_ball(code, received, decoder.target_radius());
    assert_eq!(decoded, oracle, "list decode disagreed with brute-force ball");

    let unique = decoder.unique_decode(received, &mut scratch).unwrap();
    match oracle.len() {
        0 => assert!(matches!(unique, UniqueDecode::NoCandidate)),
        1 => {
            let message = unique.message().expect("unique message");
            let coefficients: Vec<Elem> =
                (0..code.dimension()).map(|d| message.coefficient(d)).collect();
            assert_eq!(coefficients, oracle[0]);
        }
        _ => assert!(matches!(unique, UniqueDecode::Ambiguous)),
    }
}

/// Codeword-derived words with 0..=`max_errors` errors (including some that hit
/// the exceptional last coordinate), plus arbitrary words.
fn probe_words(code: &RothLempelCode<Gf8>, message: &[Elem], max_errors: usize) -> Vec<Vec<Elem>> {
    let n = code.length();
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
    // Corrupt only the exceptional last coordinate.
    let mut last_only = codeword.clone();
    last_only[n - 1] = last_only[n - 1].add(e(1));
    words.push(last_only);
    // Punctured errors plus a last-coordinate error.
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
fn nonzero_twist_matches_oracle() {
    let n = 8;
    let code = RothLempelCode::new(punctured_domain(n), ramp_multipliers(n), 2, e(3)).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(6), e(31)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn zero_twist_matches_oracle() {
    let n = 8;
    let code = RothLempelCode::new(punctured_domain(n), ramp_multipliers(n), 2, Elem::ZERO).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(200), e(1)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn unit_multipliers_match_oracle() {
    let n = 8;
    let code = RothLempelCode::new(punctured_domain(n), vec![Elem::ONE; n], 2, e(7)).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(15), e(240)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn round_trip_is_lossless() {
    let n = 8;
    let code = RothLempelCode::new(punctured_domain(n), ramp_multipliers(n), 2, e(3)).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    let mut scratch = RothLempelScratch::new();
    for &(a, b) in &[(1u8, 1u8), (5, 9), (200, 3), (0, 17), (255, 254)] {
        let message = [e(a), e(b)];
        let mut codeword = vec![Elem::ZERO; n];
        code.encode_into(&message, &mut codeword).unwrap();
        let mut candidates = Vec::new();
        decoder
            .list_decode_into(&codeword, &mut scratch, &mut candidates)
            .unwrap();
        let decoded = decoded_messages(&candidates, code.dimension());
        assert!(
            decoded.contains(&message.to_vec()),
            "encoded message not recovered for ({a}, {b})"
        );
    }
}

#[test]
fn construction_rejects_bad_parameters() {
    let n = 8;
    let domain = || punctured_domain(n);

    assert_eq!(
        RothLempelCode::new(domain(), ramp_multipliers(n - 1), 2, e(3)).unwrap_err(),
        Error::MultiplierCount {
            expected: n,
            got: n - 1
        }
    );

    let mut multipliers = ramp_multipliers(n);
    multipliers[2] = Elem::ZERO;
    assert_eq!(
        RothLempelCode::new(domain(), multipliers, 2, e(3)).unwrap_err(),
        Error::ZeroMultiplier { index: 2 }
    );

    assert_eq!(
        RothLempelCode::new(domain(), ramp_multipliers(n), 1, e(3)).unwrap_err(),
        Error::MinimumDimension {
            dimension: 1,
            minimum: 2
        }
    );

    assert_eq!(
        RothLempelCode::new(domain(), ramp_multipliers(n), n, e(3)).unwrap_err(),
        Error::InvalidDimension {
            dimension: n,
            length: n
        }
    );
}

#[test]
fn io_length_checks() {
    let n = 8;
    let code = RothLempelCode::new(punctured_domain(n), ramp_multipliers(n), 2, e(3)).unwrap();

    let mut codeword = vec![Elem::ZERO; n];
    assert_eq!(
        code.encode_into(&[e(1)], &mut codeword).unwrap_err(),
        Error::MessageLength {
            expected: 2,
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
    let mut scratch = RothLempelScratch::new();
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
