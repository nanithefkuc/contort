//! End-to-end twisted GRS decoding checked against a brute-force Hamming-ball
//! oracle over GF(2^8).
//!
//! For a small code (dimension `k = 2` over an 8-point domain) the oracle
//! enumerates every message, encodes it, and keeps those within the decoding
//! radius. The decoder's list must equal that set exactly, and the unique
//! decoder must agree with the ball's cardinality.

use contort::{
    AlekhnovichLimits, EvaluationDomain, ParameterLimits, Polynomial, TgrsCode, Error,
    TgrsScratch, Twist, UniqueDecode,
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

/// Multipliers `1, 2, …, n` — all nonzero.
fn ramp_multipliers(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

fn hamming(a: &[Elem], b: &[Elem]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every message in `GF(2^8)^k` whose codeword is within `tau` of `received`,
/// as sorted message-coefficient vectors.
fn brute_force_ball(code: &TgrsCode<Gf8>, received: &[Elem], tau: usize) -> Vec<Vec<Elem>> {
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
        // Odometer increment over base 256.
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

/// Decoded message polynomials as sorted coefficient vectors.
fn decoded_messages(candidates: &[Polynomial<Gf8>], k: usize) -> Vec<Vec<Elem>> {
    let mut messages: Vec<Vec<Elem>> = candidates
        .iter()
        .map(|poly| (0..k).map(|d| poly.coefficient(d)).collect())
        .collect();
    messages.sort();
    messages
}

/// Assert the decoder's list and unique output both agree with the oracle for
/// one received word.
fn check_against_oracle(
    code: &TgrsCode<Gf8>,
    decoder: &contort::TgrsDecoder<Gf8>,
    received: &[Elem],
) {
    let mut scratch = TgrsScratch::new();
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

/// Received words derived from a codeword plus 0..=`max_errors` deterministic
/// errors, followed by a few arbitrary words.
fn probe_words(code: &TgrsCode<Gf8>, message: &[Elem], max_errors: usize) -> Vec<Vec<Elem>> {
    let n = code.length();
    let mut codeword = vec![Elem::ZERO; n];
    code.encode_into(message, &mut codeword).unwrap();

    let mut words = Vec::new();
    for errors in 0..=max_errors {
        let mut word = codeword.clone();
        for position in 0..errors {
            let index = (position * 3 + 1) % n;
            // Flip to a deterministically different symbol.
            word[index] = word[index].add(e((position + 1) as u8));
        }
        words.push(word);
    }
    // Arbitrary patterns unrelated to any particular codeword.
    words.push((0..n).map(|i| e((7 * i + 2) as u8)).collect());
    words.push((0..n).map(|_| Elem::ZERO).collect());
    words
}

#[test]
fn single_twist_matches_oracle_on_subspace_domain() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    let twists = vec![Twist::new(1, 0, e(2))];
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, twists).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    for word in probe_words(&code, &[e(5), e(9)], 4) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn single_twist_matches_oracle_on_arbitrary_domain() {
    let n = 8;
    let points: Vec<Elem> = (1..=n as u8).map(e).collect();
    let domain = EvaluationDomain::<Gf8>::arbitrary(points).unwrap();
    let twists = vec![Twist::new(2, 1, e(3))];
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, twists).unwrap();
    // k' = k + max t = 2 + 2 = 4; keep the radius feasible for this geometry.
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    for word in probe_words(&code, &[e(17), e(200)], 4) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn repeated_destination_twists_match_oracle() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    // Both twists land on degree k-1+1 = 2 and must accumulate.
    let twists = vec![Twist::new(1, 0, e(2)), Twist::new(1, 1, e(5))];
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, twists).unwrap();
    assert_eq!(code.pseudo_dimension(), 3);
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    for word in probe_words(&code, &[e(11), e(240)], 4) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn no_twists_reduces_to_grs() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, Vec::new()).unwrap();
    assert_eq!(code.pseudo_dimension(), code.dimension());
    let decoder = code
        .list_decoder(3, parameter_limits(), root_limits())
        .unwrap();

    for word in probe_words(&code, &[e(42), e(99)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn unit_multipliers_match_oracle() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    let twists = vec![Twist::new(1, 0, e(2))];
    let code = TgrsCode::new(domain, vec![Elem::ONE; n], 2, twists).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    for word in probe_words(&code, &[e(30), e(70)], 4) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn round_trip_is_lossless() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    let twists = vec![Twist::new(1, 0, e(2))];
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, twists).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();

    let mut scratch = TgrsScratch::new();
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
    let subspace = || EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();

    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n - 1), 2, Vec::new()).unwrap_err(),
        Error::MultiplierCount {
            expected: n,
            got: n - 1
        }
    );

    let mut multipliers = ramp_multipliers(n);
    multipliers[3] = Elem::ZERO;
    assert_eq!(
        TgrsCode::new(subspace(), multipliers, 2, Vec::new()).unwrap_err(),
        Error::ZeroMultiplier { index: 3 }
    );

    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), 0, Vec::new()).unwrap_err(),
        Error::InvalidDimension {
            dimension: 0,
            length: n
        }
    );
    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), n, Vec::new()).unwrap_err(),
        Error::InvalidDimension {
            dimension: n,
            length: n
        }
    );

    // t must lie in 1..=n-k = 1..=6.
    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), 2, vec![Twist::new(0, 0, e(2))])
            .unwrap_err(),
        Error::TwistOffset { offset: 0, max: 6 }
    );
    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), 2, vec![Twist::new(7, 0, e(2))])
            .unwrap_err(),
        Error::TwistOffset { offset: 7, max: 6 }
    );

    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), 2, vec![Twist::new(1, 2, e(2))])
            .unwrap_err(),
        Error::TwistHook {
            hook: 2,
            dimension: 2
        }
    );

    assert_eq!(
        TgrsCode::new(subspace(), ramp_multipliers(n), 2, vec![Twist::new(1, 0, Elem::ZERO)])
            .unwrap_err(),
        Error::ZeroTwistCoefficient { index: 0 }
    );

    assert_eq!(
        TgrsCode::new(
            subspace(),
            ramp_multipliers(n),
            2,
            vec![Twist::new(1, 0, e(2)), Twist::new(1, 0, e(3))]
        )
        .unwrap_err(),
        Error::DuplicateTwist { offset: 1, hook: 0 }
    );
}

#[test]
fn io_length_checks() {
    let n = 8;
    let domain = EvaluationDomain::<Gf8>::additive_subspace(n).unwrap();
    let code = TgrsCode::new(domain, ramp_multipliers(n), 2, vec![Twist::new(1, 0, e(2))]).unwrap();

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
    let mut scratch = TgrsScratch::new();
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
