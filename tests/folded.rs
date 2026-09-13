//! Folded Reed–Solomon construction checks over GF(2^8).
//!
//! The encoder is validated against an independent evaluation oracle
//! (`Polynomial::evaluate_many`) exhaustively over every `k = 2` message, the
//! block metric is checked to count one block once regardless of how many
//! scalar components inside it differ, and every geometry rejection is
//! asserted by variant.

use contort::{Error, FoldedRsCode};
use fgf::Gf8B;
use fgf::gf8b::Elem;
use poly_ring::Polynomial;

fn e(value: u8) -> Elem {
    Elem::from_raw(value)
}

fn ramp(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

/// `γ = 2` has multiplicative order well above the small orbit lengths used
/// here, so it indexes them with distinct powers.
const GAMMA: u8 = 2;

#[test]
fn encoder_matches_evaluation_oracle() {
    let n = 6;
    let m = 2;
    let k = 2;
    let code = FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(n), k, m).unwrap();
    let orbit = code.orbit().to_vec();

    let mut codeword = vec![Elem::ZERO; n];
    for a0 in 0..=255u8 {
        for a1 in 0..=255u8 {
            let message = [e(a0), e(a1)];
            code.encode_into(&message, &mut codeword).unwrap();

            // Independent oracle: v_i * f(γ^i).
            let poly = Polynomial::<Gf8B>::from_coefficients(&message).unwrap();
            let values = poly.evaluate_many(&orbit).unwrap();
            for i in 0..n {
                let expected = code.multipliers()[i].mul(values[i]);
                assert_eq!(codeword[i], expected, "coordinate {i} for ({a0},{a1})");
            }
        }
    }
}

#[test]
fn block_distance_counts_a_block_once() {
    let n = 6;
    let m = 2;
    let code = FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(n), 2, m).unwrap();
    let mut a = vec![Elem::ZERO; n];
    code.encode_into(&[e(3), e(9)], &mut a).unwrap();

    // Two scalar changes inside one block (indices 0,1) → one block error.
    let mut one_block = a.clone();
    one_block[0] = Elem::from_raw(one_block[0].to_raw() ^ 1);
    one_block[1] = Elem::from_raw(one_block[1].to_raw() ^ 1);
    assert_eq!(code.block_distance(&a, &one_block), 1);

    // One scalar change in each of two blocks (indices 0 and 2) → two errors.
    let mut two_blocks = a.clone();
    two_blocks[0] = Elem::from_raw(two_blocks[0].to_raw() ^ 1);
    two_blocks[2] = Elem::from_raw(two_blocks[2].to_raw() ^ 1);
    assert_eq!(code.block_distance(&a, &two_blocks), 2);

    // Blocks partition the word.
    let rebuilt: Vec<Elem> = (0..code.blocks())
        .flat_map(|i| code.block(&a, i).to_vec())
        .collect();
    assert_eq!(rebuilt, a);
}

#[test]
fn construction_rejects_bad_geometry() {
    // Zero generator.
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(0), ramp(6), 2, 2),
        Err(Error::InsufficientOrbit { length: 6 })
    ));
    // Order-1 generator repeats γ^0 immediately, so it cannot index n > 1.
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(1), ramp(4), 2, 2),
        Err(Error::InsufficientOrbit { length: 4 })
    ));
    // Fold zero and fold not dividing n.
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(6), 2, 0),
        Err(Error::FoldParameter { fold: 0, length: 6 })
    ));
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(6), 2, 4),
        Err(Error::FoldParameter { fold: 4, length: 6 })
    ));
    // Bad dimension and zero multiplier.
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(6), 6, 2),
        Err(Error::InvalidDimension {
            dimension: 6,
            length: 6
        })
    ));
    let mut bad = ramp(6);
    bad[3] = e(0);
    assert!(matches!(
        FoldedRsCode::<Gf8B>::new(e(GAMMA), bad, 2, 2),
        Err(Error::ZeroMultiplier { index: 3 })
    ));
}

#[test]
fn io_length_checks() {
    let code = FoldedRsCode::<Gf8B>::new(e(GAMMA), ramp(6), 2, 2).unwrap();
    let mut codeword = vec![Elem::ZERO; 6];
    assert!(matches!(
        code.encode_into(&[e(1)], &mut codeword),
        Err(Error::MessageLength {
            expected: 2,
            got: 1
        })
    ));
    let mut short = vec![Elem::ZERO; 5];
    assert!(matches!(
        code.encode_into(&[e(1), e(2)], &mut short),
        Err(Error::CodewordLength {
            expected: 6,
            got: 5
        })
    ));
}
