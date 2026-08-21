//! Möbius-transformed GRS decoding against a brute-force Hamming-ball oracle,
//! plus the differential invariant that the moved-point and normalized-multiplier
//! routes return byte-identical lists.
//!
//! A small code (`k = 2` over an 8-point domain in `GF(2^8)`) lets the oracle
//! enumerate every message, encode it, and keep those within the decoding
//! radius; the decoder's moved-point list must equal that set exactly. The
//! normalized-multiplier route decodes the same code through a different plan,
//! so its list must equal the moved-point list as a set.
//! Steady-state allocation behaviour is proven in `tests/zero_alloc.rs`.

use contort::{
    AlekhnovichLimits, Error, EvaluationDomain, MobiusGrsCode, MobiusGrsScratch, MobiusMap,
    ParameterLimits, Polynomial, UniqueDecode,
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

fn ramp_multipliers(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

fn arbitrary_domain(n: usize) -> EvaluationDomain<Gf8> {
    let points: Vec<Elem> = (1..=n as u8).map(e).collect();
    EvaluationDomain::arbitrary(points).unwrap()
}

fn hamming(a: &[Elem], b: &[Elem]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Every message whose codeword is within `tau` of `received`, as sorted
/// coefficient vectors.
fn brute_force_ball(code: &MobiusGrsCode<Gf8>, received: &[Elem], tau: usize) -> Vec<Vec<Elem>> {
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

/// Received words: a codeword plus 0..=`max_errors` deterministic errors, then
/// a couple of arbitrary words.
fn probe_words(code: &MobiusGrsCode<Gf8>, message: &[Elem], max_errors: usize) -> Vec<Vec<Elem>> {
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
    words.push((0..n).map(|i| e((7 * i + 2) as u8)).collect());
    words.push((0..n).map(|_| Elem::ZERO).collect());
    words
}

/// A map with `c ≠ 0` whose pole `d/c = 200` avoids the domain `1..=8`.
fn affine_pole_free_map() -> MobiusMap<Gf8> {
    MobiusMap::new(e(1), e(1), e(1), e(200))
}

#[test]
fn moved_route_matches_oracle() {
    let n = 8;
    let code = MobiusGrsCode::new(
        arbitrary_domain(n),
        ramp_multipliers(n),
        2,
        affine_pole_free_map(),
    )
    .unwrap();
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    let mut scratch = MobiusGrsScratch::new();
    let mut candidates = Vec::new();

    for word in probe_words(&code, &[e(5), e(9)], 4) {
        decoder
            .list_decode_moved_into(&word, &mut scratch, &mut candidates)
            .unwrap();
        let decoded = decoded_messages(&candidates, code.dimension());
        let oracle = brute_force_ball(&code, &word, decoder.target_radius());
        assert_eq!(
            decoded, oracle,
            "moved route disagreed with brute-force ball"
        );

        let unique = decoder.unique_decode(&word, &mut scratch).unwrap();
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
}

#[test]
fn routes_are_byte_identical() {
    let n = 8;
    // Both a general (non-orbit) map and an orbit-preserving scaling.
    let maps = [
        affine_pole_free_map(),
        MobiusMap::new(e(2), e(0), e(0), e(1)),
    ];
    for map in maps {
        let code = MobiusGrsCode::new(arbitrary_domain(n), ramp_multipliers(n), 2, map).unwrap();
        let decoder = code
            .list_decoder(2, parameter_limits(), root_limits())
            .unwrap();
        let mut scratch = MobiusGrsScratch::new();
        let mut moved = Vec::new();
        let mut normalized = Vec::new();

        for word in probe_words(&code, &[e(11), e(3)], 4) {
            decoder
                .list_decode_moved_into(&word, &mut scratch, &mut moved)
                .unwrap();
            decoder
                .list_decode_normalized_into(&word, &mut scratch, &mut normalized)
                .unwrap();
            assert_eq!(
                decoded_messages(&moved, code.dimension()),
                decoded_messages(&normalized, code.dimension()),
                "moved and normalized routes disagreed"
            );
            // The normalized route must also match the oracle directly.
            let oracle = brute_force_ball(&code, &word, decoder.target_radius());
            assert_eq!(decoded_messages(&normalized, code.dimension()), oracle);
        }
    }
}

#[test]
fn orbit_preserving_flag() {
    let n = 8;
    let scaling = MobiusGrsCode::new(
        arbitrary_domain(n),
        ramp_multipliers(n),
        2,
        MobiusMap::new(e(2), e(0), e(0), e(1)),
    )
    .unwrap();
    assert!(scaling.is_orbit_preserving());

    let inversion = MobiusGrsCode::new(
        arbitrary_domain(n),
        ramp_multipliers(n),
        2,
        MobiusMap::new(e(0), e(1), e(1), e(0)),
    )
    .unwrap();
    assert!(inversion.is_orbit_preserving());

    let general = MobiusGrsCode::new(
        arbitrary_domain(n),
        ramp_multipliers(n),
        2,
        affine_pole_free_map(),
    )
    .unwrap();
    assert!(!general.is_orbit_preserving());
}

#[test]
fn construction_rejects_singular_and_pole() {
    let n = 8;
    // Δ = 1·1 + 1·1 = 0.
    assert_eq!(
        MobiusGrsCode::new(
            arbitrary_domain(n),
            ramp_multipliers(n),
            2,
            MobiusMap::new(e(1), e(1), e(1), e(1)),
        )
        .unwrap_err(),
        Error::MobiusDelta
    );
    // Pole d/c = 3 lands on domain point at index 2 (the point α = 3).
    assert_eq!(
        MobiusGrsCode::new(
            arbitrary_domain(n),
            ramp_multipliers(n),
            2,
            MobiusMap::new(e(1), e(0), e(1), e(3)),
        )
        .unwrap_err(),
        Error::MobiusPole { index: 2 }
    );
}

#[test]
fn construction_rejects_bad_shape() {
    let n = 8;
    assert_eq!(
        MobiusGrsCode::new(
            arbitrary_domain(n),
            ramp_multipliers(n - 1),
            2,
            affine_pole_free_map(),
        )
        .unwrap_err(),
        Error::MultiplierCount {
            expected: n,
            got: n - 1
        }
    );
    let mut zeroed = ramp_multipliers(n);
    zeroed[3] = Elem::ZERO;
    assert_eq!(
        MobiusGrsCode::new(arbitrary_domain(n), zeroed, 2, affine_pole_free_map()).unwrap_err(),
        Error::ZeroMultiplier { index: 3 }
    );
    assert_eq!(
        MobiusGrsCode::new(
            arbitrary_domain(n),
            ramp_multipliers(n),
            0,
            affine_pole_free_map()
        )
        .unwrap_err(),
        Error::InvalidDimension {
            dimension: 0,
            length: n
        }
    );
}

#[test]
fn io_length_checks() {
    let n = 8;
    let code = MobiusGrsCode::new(
        arbitrary_domain(n),
        ramp_multipliers(n),
        2,
        affine_pole_free_map(),
    )
    .unwrap();
    let mut codeword = vec![Elem::ZERO; n];
    assert_eq!(
        code.encode_into(&[e(1)], &mut codeword).unwrap_err(),
        Error::MessageLength {
            expected: 2,
            got: 1
        }
    );
    assert_eq!(
        code.encode_into(&[e(1), e(2)], &mut codeword[..n - 1])
            .unwrap_err(),
        Error::CodewordLength {
            expected: n,
            got: n - 1
        }
    );
    let decoder = code
        .list_decoder(2, parameter_limits(), root_limits())
        .unwrap();
    let mut scratch = MobiusGrsScratch::new();
    let mut candidates = Vec::new();
    assert_eq!(
        decoder
            .list_decode_into(&[e(1); 7], &mut scratch, &mut candidates)
            .unwrap_err(),
        Error::ReceivedLength {
            expected: n,
            got: 7
        }
    );
}
