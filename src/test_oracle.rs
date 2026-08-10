use crate::{
    field::Field,
    polynomial::Polynomial,
    reference::{enumerate_polynomials, hamming_distance},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OracleCandidate<F: Field> {
    pub(crate) message: Polynomial<F>,
    pub(crate) codeword: Vec<F>,
    pub(crate) distance: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OracleError {
    CodewordLengthMismatch,
}

pub(crate) fn decode_by_enumeration<F, Encode>(
    field_elements: &[F],
    message_degree_bound: usize,
    received: &[F],
    radius: usize,
    mut encode: Encode,
) -> Result<Vec<OracleCandidate<F>>, OracleError>
where
    F: Field,
    Encode: FnMut(&Polynomial<F>) -> Vec<F>,
{
    let mut candidates = Vec::new();

    for message in enumerate_polynomials(field_elements, message_degree_bound) {
        let codeword = encode(&message);
        let distance =
            hamming_distance(&codeword, received).ok_or(OracleError::CodewordLengthMismatch)?;
        if distance <= radius {
            candidates.push(OracleCandidate {
                message,
                codeword,
                distance,
            });
        }
    }

    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use crate::{
        field::{Field, PrimeField},
        polynomial::Polynomial,
        reference::{enumerate_polynomials, hamming_distance},
    };

    use super::decode_by_enumeration;

    fn field_elements<const MODULUS: u64>() -> Vec<PrimeField<MODULUS>> {
        (0..MODULUS).map(PrimeField::new).collect()
    }

    fn grs_encode<F: Field>(
        message: &Polynomial<F>,
        evaluation_points: &[F],
        multipliers: &[F],
    ) -> Vec<F> {
        assert_eq!(evaluation_points.len(), multipliers.len());
        evaluation_points
            .iter()
            .zip(multipliers)
            .map(|(&point, &multiplier)| message.evaluate(point).mul(multiplier))
            .collect()
    }

    #[test]
    fn enumerates_every_bounded_polynomial_once() {
        type F3 = PrimeField<3>;
        let elements = field_elements::<3>();
        let polynomials = enumerate_polynomials(&elements, 2);

        assert_eq!(polynomials.len(), 9);
        for left in 0..polynomials.len() {
            for right in left + 1..polynomials.len() {
                assert_ne!(polynomials[left], polynomials[right]);
            }
        }
        assert_eq!(
            enumerate_polynomials(&elements, 0),
            vec![Polynomial::<F3>::zero()]
        );
    }

    #[test]
    fn exhaustively_checks_polynomial_identities_over_f3() {
        let elements = field_elements::<3>();
        let polynomials = enumerate_polynomials(&elements, 3);

        for left in &polynomials {
            for right in &polynomials {
                assert_eq!(left.add(right).sub(right), *left);

                let product = left.mul(right);
                let composition = left.compose(right);
                for &point in &elements {
                    assert_eq!(
                        product.evaluate(point),
                        left.evaluate(point).mul(right.evaluate(point))
                    );
                    assert_eq!(
                        composition.evaluate(point),
                        left.evaluate(right.evaluate(point))
                    );
                }

                if !right.is_zero() {
                    let (quotient, remainder) =
                        left.div_rem(right).expect("nonzero divisor over a field");
                    assert_eq!(quotient.mul(right).add(&remainder), *left);
                    assert!(
                        remainder
                            .degree()
                            .is_none_or(|degree| degree < right.degree().expect("nonzero divisor"))
                    );
                }
            }
        }
    }

    #[test]
    fn hamming_distance_rejects_different_lengths() {
        type F5 = PrimeField<5>;
        assert_eq!(hamming_distance(&[F5::ZERO], &[F5::ZERO, F5::ONE]), None);
        assert_eq!(
            hamming_distance(&[F5::ZERO, F5::ONE], &[F5::ONE, F5::ONE]),
            Some(1)
        );
    }

    #[test]
    fn grs_oracle_finds_the_exact_radius_one_candidate() {
        type F5 = PrimeField<5>;
        let elements = field_elements::<5>();
        let points = [F5::new(0), F5::new(1), F5::new(2), F5::new(3)];
        let multipliers = [F5::new(1), F5::new(2), F5::new(3), F5::new(4)];
        let message = Polynomial::from_coefficients(vec![F5::new(1), F5::new(2)]);
        let mut received = grs_encode(&message, &points, &multipliers);
        received[1] = received[1].add(F5::ONE);

        let candidates = decode_by_enumeration(&elements, 2, &received, 1, |candidate| {
            grs_encode(candidate, &points, &multipliers)
        })
        .expect("all codewords have the expected length");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].message, message);
        assert_eq!(candidates[0].distance, 1);
    }

    #[test]
    fn tgrs_oracle_uses_twisted_high_coefficients() {
        type F7 = PrimeField<7>;
        let elements = field_elements::<7>();
        let points = [F7::new(0), F7::new(1), F7::new(2), F7::new(3), F7::new(4)];
        let multipliers = [F7::ONE; 5];
        let message = Polynomial::from_coefficients(vec![F7::new(2), F7::new(1)]);
        let encode = |candidate: &Polynomial<F7>| {
            let mut twisted_coefficients = candidate.coefficients().to_vec();
            twisted_coefficients.resize(3, F7::ZERO);
            twisted_coefficients[2] =
                twisted_coefficients[2].add(F7::new(3).mul(candidate.coefficient(0)));
            grs_encode(
                &Polynomial::from_coefficients(twisted_coefficients),
                &points,
                &multipliers,
            )
        };
        let mut received = encode(&message);
        received[3] = received[3].add(F7::ONE);

        let candidates = decode_by_enumeration(&elements, 2, &received, 1, encode)
            .expect("all codewords have the expected length");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].message, message);
        assert_eq!(candidates[0].distance, 1);
    }

    #[test]
    fn roth_lempel_oracle_checks_the_exceptional_coordinate() {
        type F7 = PrimeField<7>;
        let elements = field_elements::<7>();
        let points = [F7::new(0), F7::new(1), F7::new(2), F7::new(3), F7::new(4)];
        let multipliers = [F7::ONE; 6];
        let delta = F7::new(2);
        let message = Polynomial::from_coefficients(vec![F7::new(1), F7::new(2), F7::new(3)]);
        let encode = |candidate: &Polynomial<F7>| {
            let mut codeword = grs_encode(candidate, &points, &multipliers[..5]);
            let exceptional = candidate
                .coefficient(1)
                .add(delta.mul(candidate.coefficient(2)))
                .mul(multipliers[5]);
            codeword.push(exceptional);
            codeword
        };
        let mut received = encode(&message);
        received[5] = received[5].add(F7::ONE);

        let candidates = decode_by_enumeration(&elements, 3, &received, 1, encode)
            .expect("all codewords have the expected length");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].message, message);
        assert_eq!(candidates[0].distance, 1);
        assert_eq!(candidates[0].codeword[5], F7::new(1));
    }
}
