//! End-to-end punctured GRS decoding checked against a brute-force Hamming-ball
//! oracle over GF(2^8), plus structural checks on the MDS distance, the
//! idempotent union semantics of puncture composition, the block-alignment
//! capability flag, and construction validation. Steady-state allocation
//! behaviour is proven in `tests/zero_alloc.rs`.
//!
//! A punctured GRS code is exactly the GRS code on the surviving coordinates,
//! so for a small code the oracle enumerates every message, encodes the
//! punctured codeword, and keeps those within the decoding radius. The
//! decoder's list must equal that set, and the unique decoder must agree with
//! the ball's cardinality.

use contort::{Error, PuncturedGrsCode, PuncturedGrsDecoder, PuncturedGrsScratch, UniqueDecode};
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

/// An `n`-point arbitrary evaluation domain over distinct nonzero elements.
fn base_domain(n: usize) -> EvaluationDomain<Gf8B> {
    let points: Vec<Elem> = (1..=n as u8).map(e).collect();
    EvaluationDomain::<Gf8B>::arbitrary(points).unwrap()
}

fn hamming(a: &[Elem], b: &[Elem]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every message in `GF(2^8)^k` whose punctured codeword is within `tau` of
/// `received`, as sorted message-coefficient vectors.
fn brute_force_ball(
    code: &PuncturedGrsCode<Gf8B>,
    received: &[Elem],
    tau: usize,
) -> Vec<Vec<Elem>> {
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

/// Decoded message polynomials as sorted coefficient vectors.
fn decoded_messages(candidates: &[Polynomial<Gf8B>], k: usize) -> Vec<Vec<Elem>> {
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
    code: &PuncturedGrsCode<Gf8B>,
    decoder: &PuncturedGrsDecoder<Gf8B>,
    received: &[Elem],
) {
    let mut scratch = PuncturedGrsScratch::new();
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

/// Codeword-derived words with 0..=`max_errors` errors, plus arbitrary words.
fn probe_words(
    code: &PuncturedGrsCode<Gf8B>,
    message: &[Elem],
    max_errors: usize,
) -> Vec<Vec<Elem>> {
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
    words.push((0..n).map(|i| e((5 * i + 3) as u8)).collect());
    words.push((0..n).map(|_| Elem::ZERO).collect());
    words
}

/// Surviving base indices of a code, in ascending order.
fn surviving_indices(code: &PuncturedGrsCode<Gf8B>) -> Vec<usize> {
    (0..code.base_length())
        .filter(|i| !code.punctures().contains(i))
        .collect()
}

#[test]
fn punctures_match_oracle() {
    // Unique-decoding regime: a punctured [6, 2] MDS code.
    let code = PuncturedGrsCode::new(base_domain(8), ramp_multipliers(8), 2, vec![2, 5]).unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(6), e(31)], 3) {
        check_against_oracle(&code, &decoder, &word);
    }

    // List-decoding regime beyond the unique radius: a punctured [16, 2] code
    // decoded at radius 8, where a word can sit inside several codewords' balls.
    let code = PuncturedGrsCode::new(base_domain(17), ramp_multipliers(17), 2, vec![8]).unwrap();
    let decoder = code
        .list_decoder(8, parameter_limits(), root_limits())
        .unwrap();
    for word in probe_words(&code, &[e(19), e(211)], 9) {
        check_against_oracle(&code, &decoder, &word);
    }
}

#[test]
fn distance_is_mds() {
    // A punctured [5, 3] code: minimum nonzero weight equals n' - k + 1 = 3.
    let code = PuncturedGrsCode::new(base_domain(7), ramp_multipliers(7), 3, vec![2, 5]).unwrap();
    let k = code.dimension();
    let n = code.length();

    let mut message = vec![Elem::ZERO; k];
    let mut codeword = vec![Elem::ZERO; n];
    let mut counter = vec![0u16; k];
    let mut minimum = usize::MAX;
    loop {
        for (slot, &value) in message.iter_mut().zip(counter.iter()) {
            *slot = e(value as u8);
        }
        code.encode_into(&message, &mut codeword).unwrap();
        let weight = codeword
            .iter()
            .filter(|symbol| **symbol != Elem::ZERO)
            .count();
        if weight > 0 {
            minimum = minimum.min(weight);
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
    assert_eq!(minimum, n - k + 1, "punctured GRS code is not MDS");
}

#[test]
fn double_puncture_is_union() {
    let n = 9;
    let first = vec![2usize, 5];
    let second = [3usize, 7];
    let mut union = first.clone();
    union.extend(second.iter().copied());

    let code_first =
        PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), 2, first.clone()).unwrap();
    let code_union = PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), 2, union).unwrap();

    // Puncturing `first` and then the `second` coordinates of the survivors
    // leaves exactly the survivors of the union.
    let composed: Vec<usize> = surviving_indices(&code_first)
        .into_iter()
        .filter(|i| !second.contains(i))
        .collect();
    assert_eq!(composed, surviving_indices(&code_union));

    // The surviving evaluation points agree coordinate for coordinate.
    let base = base_domain(n);
    let points_union: Vec<Elem> = surviving_indices(&code_union)
        .iter()
        .map(|&i| base.points()[i])
        .collect();
    let points_composed: Vec<Elem> = composed.iter().map(|&i| base.points()[i]).collect();
    assert_eq!(points_union, points_composed);

    // Encoding through the union equals encoding through `first` and then
    // dropping the survivor coordinates whose base index lies in `second`.
    let message = [e(3), e(5)];
    let mut codeword_first = vec![Elem::ZERO; code_first.length()];
    code_first
        .encode_into(&message, &mut codeword_first)
        .unwrap();
    let survivors_first = surviving_indices(&code_first);
    let codeword_composed: Vec<Elem> = survivors_first
        .iter()
        .zip(codeword_first)
        .filter(|(i, _)| !second.contains(i))
        .map(|(_, symbol)| symbol)
        .collect();

    let mut codeword_union = vec![Elem::ZERO; code_union.length()];
    code_union
        .encode_into(&message, &mut codeword_union)
        .unwrap();
    assert_eq!(codeword_composed, codeword_union);
}

#[test]
fn block_alignment_flag() {
    // Blocks of 2 over n = 8: puncture blocks 0 and 2 entirely.
    let aligned =
        PuncturedGrsCode::new(base_domain(8), ramp_multipliers(8), 2, vec![0, 1, 4, 5]).unwrap();
    assert!(
        aligned.is_block_aligned(1),
        "unit blocks are always aligned"
    );
    assert!(
        aligned.is_block_aligned(2),
        "whole-block puncture is aligned"
    );
    assert!(
        !aligned.is_block_aligned(4),
        "half-punctured 4-blocks are not aligned"
    );
    assert!(
        !aligned.is_block_aligned(3),
        "a fold not dividing n is never aligned"
    );
    assert!(!aligned.is_block_aligned(0), "fold zero is never aligned");

    // A puncture straddling a 2-block boundary is not aligned.
    let mid = PuncturedGrsCode::new(base_domain(8), ramp_multipliers(8), 2, vec![0, 2]).unwrap();
    assert!(
        !mid.is_block_aligned(2),
        "mid-block puncture is not aligned"
    );
}

#[test]
fn construction_rejects_bad_parameters() {
    let n = 8;

    assert_eq!(
        PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n - 1), 3, vec![2, 5]).unwrap_err(),
        Error::MultiplierCount {
            expected: n,
            got: n - 1
        }
    );

    let mut multipliers = ramp_multipliers(n);
    multipliers[4] = Elem::ZERO;
    assert_eq!(
        PuncturedGrsCode::new(base_domain(n), multipliers, 3, vec![2, 5]).unwrap_err(),
        Error::ZeroMultiplier { index: 4 }
    );

    assert_eq!(
        PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), n, vec![]).unwrap_err(),
        Error::InvalidDimension {
            dimension: n,
            length: n
        }
    );

    assert_eq!(
        PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), 3, vec![8]).unwrap_err(),
        Error::PunctureIndex {
            index: 8,
            length: n
        }
    );

    assert_eq!(
        PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), 3, vec![2, 2]).unwrap_err(),
        Error::DuplicatePuncture { index: 2 }
    );

    assert_eq!(
        PuncturedGrsCode::new(
            base_domain(n),
            ramp_multipliers(n),
            3,
            vec![0, 1, 2, 3, 4, 5]
        )
        .unwrap_err(),
        Error::PunctureLength {
            remaining: 2,
            dimension: 3
        }
    );
}

#[test]
fn io_length_checks() {
    let n = 8;
    let code = PuncturedGrsCode::new(base_domain(n), ramp_multipliers(n), 3, vec![2, 5]).unwrap();
    let effective = code.length();

    let mut codeword = vec![Elem::ZERO; effective];
    assert_eq!(
        code.encode_into(&[e(1)], &mut codeword).unwrap_err(),
        Error::MessageLength {
            expected: 3,
            got: 1
        }
    );

    let mut short = vec![Elem::ZERO; effective - 1];
    assert_eq!(
        code.encode_into(&[e(1), e(2), e(3)], &mut short)
            .unwrap_err(),
        Error::CodewordLength {
            expected: effective,
            got: effective - 1
        }
    );

    let decoder = code
        .list_decoder(1, parameter_limits(), root_limits())
        .unwrap();
    let mut scratch = PuncturedGrsScratch::new();
    let mut candidates = Vec::new();
    assert_eq!(
        decoder
            .list_decode_into(
                &vec![Elem::ZERO; effective + 1],
                &mut scratch,
                &mut candidates
            )
            .unwrap_err(),
        Error::ReceivedLength {
            expected: effective,
            got: effective + 1
        }
    );
}
