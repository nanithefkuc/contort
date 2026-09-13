//! Homogeneous interleaved Reed–Solomon construction checks over GF(2^8).
//!
//! The encoder is validated against an independent per-row evaluation oracle
//! exhaustively over every `ℓ = 2`, `k = 2` message batch, the column metric is
//! checked to count one column once regardless of how many rows differ in it,
//! and every geometry rejection is asserted by variant.

use contort::{Error, InterleavedRsCode};
use fgf::Gf8B;
use fgf::gf8b::Elem;
use gs_engine::EvaluationDomain;
use poly_ring::Polynomial;

fn e(value: u8) -> Elem {
    Elem::from_raw(value)
}

fn ramp(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

fn domain(n: usize) -> EvaluationDomain<Gf8B> {
    EvaluationDomain::arbitrary((0..n as u8).map(e).collect()).unwrap()
}

#[test]
fn encoder_matches_evaluation_oracle() {
    let n = 5;
    let k = 2;
    let ell = 2;
    let code = InterleavedRsCode::<Gf8B>::new(domain(n), ramp(n), k, ell).unwrap();
    let points: Vec<Elem> = (0..n as u8).map(e).collect();

    let mut codeword = vec![Elem::ZERO; ell * n];
    for a0 in 0..=255u8 {
        for a1 in 0..=255u8 {
            // Row 0 = (a0, a1); row 1 = (a1, a0) to keep rows distinct.
            let messages = [e(a0), e(a1), e(a1), e(a0)];
            code.encode_into(&messages, &mut codeword).unwrap();

            for row in 0..ell {
                let poly =
                    Polynomial::<Gf8B>::from_coefficients(&messages[row * k..row * k + k]).unwrap();
                let values = poly.evaluate_many(&points).unwrap();
                for col in 0..n {
                    let expected = code.multipliers()[col].mul(values[col]);
                    assert_eq!(
                        codeword[col * ell + row],
                        expected,
                        "row {row} col {col} for ({a0},{a1})"
                    );
                }
            }
        }
    }
}

#[test]
fn column_distance_counts_a_column_once() {
    let n = 5;
    let ell = 3;
    let code = InterleavedRsCode::<Gf8B>::new(domain(n), ramp(n), 2, ell).unwrap();
    let mut a = vec![Elem::ZERO; ell * n];
    code.encode_into(&[e(3), e(9), e(2), e(7), e(5), e(1)], &mut a)
        .unwrap();

    // All ℓ rows of column 0 corrupted → one column error.
    let mut one_col = a.clone();
    for slot in one_col[0..ell].iter_mut() {
        *slot = Elem::from_raw(slot.to_raw() ^ 1);
    }
    assert_eq!(code.column_distance(&a, &one_col), 1);

    // One row corrupted in each of two columns (0 and 2) → two column errors.
    let mut two_cols = a.clone();
    two_cols[0] = Elem::from_raw(two_cols[0].to_raw() ^ 1);
    two_cols[2 * ell] = Elem::from_raw(two_cols[2 * ell].to_raw() ^ 1);
    assert_eq!(code.column_distance(&a, &two_cols), 2);

    // Columns partition the word.
    let rebuilt: Vec<Elem> = (0..code.length())
        .flat_map(|i| code.column(&a, i).to_vec())
        .collect();
    assert_eq!(rebuilt, a);
}

#[test]
fn construction_rejects_bad_geometry() {
    assert!(matches!(
        InterleavedRsCode::<Gf8B>::new(domain(5), ramp(4), 2, 2),
        Err(Error::MultiplierCount {
            expected: 5,
            got: 4
        })
    ));
    assert!(matches!(
        InterleavedRsCode::<Gf8B>::new(domain(5), ramp(5), 5, 2),
        Err(Error::InvalidDimension {
            dimension: 5,
            length: 5
        })
    ));
    assert!(matches!(
        InterleavedRsCode::<Gf8B>::new(domain(5), ramp(5), 2, 0),
        Err(Error::ZeroInterleave)
    ));
    let mut bad = ramp(5);
    bad[2] = e(0);
    assert!(matches!(
        InterleavedRsCode::<Gf8B>::new(domain(5), bad, 2, 2),
        Err(Error::ZeroMultiplier { index: 2 })
    ));
}

#[test]
fn io_length_checks() {
    let code = InterleavedRsCode::<Gf8B>::new(domain(5), ramp(5), 2, 2).unwrap();
    let mut codeword = vec![Elem::ZERO; 2 * 5];
    // Wrong number of row messages (ℓ·k = 4 expected).
    assert!(matches!(
        code.encode_into(&[e(1), e(2)], &mut codeword),
        Err(Error::MessageCount {
            expected: 2,
            got: 1
        })
    ));
    let mut short = vec![Elem::ZERO; 9];
    assert!(matches!(
        code.encode_into(&[e(1), e(2), e(3), e(4)], &mut short),
        Err(Error::CodewordLength {
            expected: 10,
            got: 9
        })
    ));
}
