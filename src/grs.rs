use crate::{
    field::Field,
    polynomial::{Polynomial, binomial_in_field},
    reference::{enumerate_polynomials, hamming_distance},
};

const MAX_REFERENCE_MESSAGES: usize = 1_000_000;
const MAX_REFERENCE_MATRIX_CELLS: usize = 4_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GrsError {
    EmptyCode,
    MultiplierLengthMismatch,
    ZeroDimension,
    DimensionExceedsLength,
    DuplicateEvaluationPoint,
    ZeroMultiplier,
    MessageDegreeTooLarge,
    ReceivedWordLengthMismatch,
    ZeroMultiplicity,
    RadiusExceedsLength,
    RadiusOutsideInterpolationBound {
        radius: usize,
        exclusive_bound: usize,
    },
    ArithmeticOverflow,
    InvalidFieldEnumeration,
    ReferenceInstanceTooLarge,
    InterpolationHasNoNullspace,
    InterpolationVerificationFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GuruswamiSudanParameters {
    radius: usize,
    multiplicity: usize,
    weighted_degree_bound: usize,
    radius_exclusive_bound: usize,
    list_size_bound: Option<usize>,
}

impl GuruswamiSudanParameters {
    pub(crate) fn new(
        length: usize,
        dimension: usize,
        radius: usize,
        multiplicity: usize,
    ) -> Result<Self, GrsError> {
        if length == 0 {
            return Err(GrsError::EmptyCode);
        }
        if dimension == 0 {
            return Err(GrsError::ZeroDimension);
        }
        if dimension > length {
            return Err(GrsError::DimensionExceedsLength);
        }
        if multiplicity == 0 {
            return Err(GrsError::ZeroMultiplicity);
        }
        if radius > length {
            return Err(GrsError::RadiusExceedsLength);
        }

        if dimension == 1 {
            let radius_exclusive_bound = length;
            if radius >= radius_exclusive_bound {
                return Err(GrsError::RadiusOutsideInterpolationBound {
                    radius,
                    exclusive_bound: radius_exclusive_bound,
                });
            }
            return Ok(Self {
                radius,
                multiplicity,
                weighted_degree_bound: 0,
                radius_exclusive_bound,
                list_size_bound: None,
            });
        }

        let length = length as u128;
        let dimension_minus_one = (dimension - 1) as u128;
        let multiplicity_u128 = multiplicity as u128;
        let multiplicity_plus_one = multiplicity_u128
            .checked_add(1)
            .ok_or(GrsError::ArithmeticOverflow)?;

        let weighted_radicand = checked_product(&[
            length,
            dimension_minus_one,
            multiplicity_u128,
            multiplicity_plus_one,
        ])?;
        let weighted_degree_bound =
            usize::try_from(weighted_radicand.isqrt()).map_err(|_| GrsError::ArithmeticOverflow)?;

        let radius_radicand_numerator =
            checked_product(&[length, dimension_minus_one, multiplicity_plus_one])?;
        let radius_root = usize::try_from((radius_radicand_numerator / multiplicity_u128).isqrt())
            .map_err(|_| GrsError::ArithmeticOverflow)?;
        let radius_exclusive_bound = usize::try_from(length)
            .map_err(|_| GrsError::ArithmeticOverflow)?
            .saturating_sub(radius_root);
        if radius >= radius_exclusive_bound {
            return Err(GrsError::RadiusOutsideInterpolationBound {
                radius,
                exclusive_bound: radius_exclusive_bound,
            });
        }

        let list_radicand_numerator =
            checked_product(&[length, multiplicity_u128, multiplicity_plus_one])?;
        let list_size_bound =
            usize::try_from((list_radicand_numerator / dimension_minus_one).isqrt())
                .map_err(|_| GrsError::ArithmeticOverflow)?;

        Ok(Self {
            radius,
            multiplicity,
            weighted_degree_bound,
            radius_exclusive_bound,
            list_size_bound: Some(list_size_bound),
        })
    }

    pub(crate) const fn radius(self) -> usize {
        self.radius
    }

    pub(crate) const fn multiplicity(self) -> usize {
        self.multiplicity
    }

    pub(crate) const fn weighted_degree_bound(self) -> usize {
        self.weighted_degree_bound
    }

    pub(crate) const fn radius_exclusive_bound(self) -> usize {
        self.radius_exclusive_bound
    }

    pub(crate) const fn list_size_bound(self) -> Option<usize> {
        self.list_size_bound
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedCandidate<F: Field> {
    pub(crate) polynomial: Polynomial<F>,
    pub(crate) codeword: Vec<F>,
    pub(crate) distance: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GrsCode<F: Field> {
    evaluation_points: Vec<F>,
    multipliers: Vec<F>,
    dimension: usize,
}

impl<F: Field> GrsCode<F> {
    pub(crate) fn new(
        evaluation_points: Vec<F>,
        multipliers: Vec<F>,
        dimension: usize,
    ) -> Result<Self, GrsError> {
        if evaluation_points.is_empty() {
            return Err(GrsError::EmptyCode);
        }
        if evaluation_points.len() != multipliers.len() {
            return Err(GrsError::MultiplierLengthMismatch);
        }
        if dimension == 0 {
            return Err(GrsError::ZeroDimension);
        }
        if dimension > evaluation_points.len() {
            return Err(GrsError::DimensionExceedsLength);
        }
        for (index, &point) in evaluation_points.iter().enumerate() {
            if evaluation_points[..index].contains(&point) {
                return Err(GrsError::DuplicateEvaluationPoint);
            }
        }
        if multipliers.iter().any(|multiplier| multiplier.is_zero()) {
            return Err(GrsError::ZeroMultiplier);
        }

        Ok(Self {
            evaluation_points,
            multipliers,
            dimension,
        })
    }

    pub(crate) fn length(&self) -> usize {
        self.evaluation_points.len()
    }

    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn encode(&self, message: &Polynomial<F>) -> Result<Vec<F>, GrsError> {
        if message
            .degree()
            .is_some_and(|degree| degree >= self.dimension)
        {
            return Err(GrsError::MessageDegreeTooLarge);
        }

        Ok(self
            .evaluation_points
            .iter()
            .zip(&self.multipliers)
            .map(|(&point, &multiplier)| message.evaluate(point).mul(multiplier))
            .collect())
    }

    pub(crate) fn list_decode_reference(
        &self,
        received: &[F],
        radius: usize,
        multiplicity: usize,
        field_elements: &[F],
    ) -> Result<Vec<DecodedCandidate<F>>, GrsError> {
        if received.len() != self.length() {
            return Err(GrsError::ReceivedWordLengthMismatch);
        }
        let parameters =
            GuruswamiSudanParameters::new(self.length(), self.dimension, radius, multiplicity)?;
        validate_field_enumeration(field_elements)?;
        check_reference_message_count(field_elements.len(), self.dimension)?;

        if self.dimension == 1 {
            return self.filter_reference_candidates(
                enumerate_polynomials(field_elements, self.dimension),
                received,
                parameters.radius(),
            );
        }

        let normalized_received = received
            .iter()
            .zip(&self.multipliers)
            .map(|(&symbol, &multiplier)| {
                symbol
                    .divide(multiplier)
                    .expect("validated nonzero GRS multiplier must be invertible")
            })
            .collect::<Vec<_>>();
        let interpolation = self.interpolate(&normalized_received, parameters)?;

        let roots = enumerate_polynomials(field_elements, self.dimension)
            .into_iter()
            .filter(|candidate| interpolation.substitute_y(candidate).is_zero())
            .collect();
        self.filter_reference_candidates(roots, received, parameters.radius())
    }

    fn interpolate(
        &self,
        normalized_received: &[F],
        parameters: GuruswamiSudanParameters,
    ) -> Result<BivariatePolynomial<F>, GrsError> {
        let monomials = interpolation_monomials(self.dimension, parameters.weighted_degree_bound());
        let constraint_count = checked_constraint_count(self.length(), parameters.multiplicity())?;
        let matrix_cells = constraint_count
            .checked_mul(monomials.len())
            .ok_or(GrsError::ArithmeticOverflow)?;
        if matrix_cells > MAX_REFERENCE_MATRIX_CELLS {
            return Err(GrsError::ReferenceInstanceTooLarge);
        }

        let mut matrix = Vec::with_capacity(constraint_count);
        for (&point, &value) in self.evaluation_points.iter().zip(normalized_received) {
            for x_order in 0..parameters.multiplicity() {
                for y_order in 0..parameters.multiplicity() - x_order {
                    matrix.push(
                        monomials
                            .iter()
                            .map(|monomial| {
                                monomial_hasse_value(*monomial, x_order, y_order, point, value)
                            })
                            .collect(),
                    );
                }
            }
        }

        let coefficients = nonzero_null_vector(matrix, monomials.len())?;
        let interpolation = BivariatePolynomial::new(monomials, coefficients);
        if interpolation.is_zero()
            || !interpolation.satisfies_constraints(
                &self.evaluation_points,
                normalized_received,
                parameters.multiplicity(),
            )
        {
            return Err(GrsError::InterpolationVerificationFailed);
        }

        Ok(interpolation)
    }

    fn filter_reference_candidates(
        &self,
        candidates: Vec<Polynomial<F>>,
        received: &[F],
        radius: usize,
    ) -> Result<Vec<DecodedCandidate<F>>, GrsError> {
        let mut output: Vec<DecodedCandidate<F>> = Vec::new();

        for polynomial in candidates {
            if output
                .iter()
                .any(|candidate| candidate.polynomial == polynomial)
            {
                continue;
            }

            let codeword = self.encode(&polynomial)?;
            let distance = hamming_distance(&codeword, received)
                .expect("GRS encoder always returns the code length");
            if distance <= radius {
                output.push(DecodedCandidate {
                    polynomial,
                    codeword,
                    distance,
                });
            }
        }

        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BivariateMonomial {
    x_degree: usize,
    y_degree: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BivariateTerm<F: Field> {
    monomial: BivariateMonomial,
    coefficient: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BivariatePolynomial<F: Field> {
    terms: Vec<BivariateTerm<F>>,
}

impl<F: Field> BivariatePolynomial<F> {
    fn new(monomials: Vec<BivariateMonomial>, coefficients: Vec<F>) -> Self {
        let terms = monomials
            .into_iter()
            .zip(coefficients)
            .filter_map(|(monomial, coefficient)| {
                (!coefficient.is_zero()).then_some(BivariateTerm {
                    monomial,
                    coefficient,
                })
            })
            .collect();
        Self { terms }
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn hasse_evaluate(&self, x_order: usize, y_order: usize, x: F, y: F) -> F {
        self.terms.iter().fold(F::ZERO, |sum, term| {
            sum.add(term.coefficient.mul(monomial_hasse_value(
                term.monomial,
                x_order,
                y_order,
                x,
                y,
            )))
        })
    }

    fn satisfies_constraints(&self, points: &[F], values: &[F], multiplicity: usize) -> bool {
        points.iter().zip(values).all(|(&point, &value)| {
            (0..multiplicity).all(|x_order| {
                (0..multiplicity - x_order).all(|y_order| {
                    self.hasse_evaluate(x_order, y_order, point, value)
                        .is_zero()
                })
            })
        })
    }

    fn substitute_y(&self, polynomial: &Polynomial<F>) -> Polynomial<F> {
        let maximum_y_degree = self
            .terms
            .iter()
            .map(|term| term.monomial.y_degree)
            .max()
            .unwrap_or(0);
        let mut powers = Vec::with_capacity(maximum_y_degree + 1);
        powers.push(Polynomial::one());
        for y_degree in 1..=maximum_y_degree {
            powers.push(powers[y_degree - 1].mul(polynomial));
        }

        let mut coefficients = Vec::new();
        for term in &self.terms {
            let power = &powers[term.monomial.y_degree];
            let required_length = term.monomial.x_degree + power.coefficients().len();
            if coefficients.len() < required_length {
                coefficients.resize(required_length, F::ZERO);
            }
            for (power_degree, &power_coefficient) in power.coefficients().iter().enumerate() {
                let output_degree = term.monomial.x_degree + power_degree;
                coefficients[output_degree] =
                    coefficients[output_degree].add(term.coefficient.mul(power_coefficient));
            }
        }

        Polynomial::from_coefficients(coefficients)
    }
}

fn interpolation_monomials(
    dimension: usize,
    weighted_degree_bound: usize,
) -> Vec<BivariateMonomial> {
    let y_weight = dimension - 1;
    let maximum_y_degree = weighted_degree_bound / y_weight;
    let mut monomials = Vec::new();

    for y_degree in 0..=maximum_y_degree {
        let maximum_x_degree = weighted_degree_bound - y_weight * y_degree;
        for x_degree in 0..=maximum_x_degree {
            monomials.push(BivariateMonomial { x_degree, y_degree });
        }
    }

    monomials
}

fn monomial_hasse_value<F: Field>(
    monomial: BivariateMonomial,
    x_order: usize,
    y_order: usize,
    x: F,
    y: F,
) -> F {
    if x_order > monomial.x_degree || y_order > monomial.y_degree {
        return F::ZERO;
    }

    binomial_in_field::<F>(monomial.x_degree, x_order)
        .mul(binomial_in_field::<F>(monomial.y_degree, y_order))
        .mul(x.pow(u64::try_from(monomial.x_degree - x_order).expect("degree exceeds u64")))
        .mul(y.pow(u64::try_from(monomial.y_degree - y_order).expect("degree exceeds u64")))
}

fn nonzero_null_vector<F: Field>(
    mut matrix: Vec<Vec<F>>,
    column_count: usize,
) -> Result<Vec<F>, GrsError> {
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;

    for column in 0..column_count {
        let Some(found_row) = (pivot_row..matrix.len()).find(|&row| !matrix[row][column].is_zero())
        else {
            continue;
        };
        matrix.swap(pivot_row, found_row);

        let pivot_inverse = matrix[pivot_row][column]
            .inverse()
            .ok_or(GrsError::InterpolationHasNoNullspace)?;
        for entry in &mut matrix[pivot_row][column..] {
            *entry = entry.mul(pivot_inverse);
        }

        let (rows_before_pivot, pivot_and_rows_after) = matrix.split_at_mut(pivot_row);
        let (normalized_pivot, rows_after_pivot) = pivot_and_rows_after
            .split_first_mut()
            .expect("pivot row is within the interpolation matrix");
        for row in rows_before_pivot
            .iter_mut()
            .chain(rows_after_pivot.iter_mut())
        {
            let factor = row[column];
            if factor.is_zero() {
                continue;
            }
            for (entry, &pivot_entry) in row[column..].iter_mut().zip(&normalized_pivot[column..]) {
                *entry = entry.sub(factor.mul(pivot_entry));
            }
        }

        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == matrix.len() {
            break;
        }
    }

    let free_column = (0..column_count)
        .find(|column| !pivot_columns.contains(column))
        .ok_or(GrsError::InterpolationHasNoNullspace)?;
    let mut solution = vec![F::ZERO; column_count];
    solution[free_column] = F::ONE;
    for (row, &column) in pivot_columns.iter().enumerate() {
        solution[column] = matrix[row][free_column].neg();
    }

    Ok(solution)
}

fn validate_field_enumeration<F: Field>(field_elements: &[F]) -> Result<(), GrsError> {
    if field_elements.is_empty()
        || !field_elements.contains(&F::ZERO)
        || !field_elements.contains(&F::ONE)
    {
        return Err(GrsError::InvalidFieldEnumeration);
    }
    for (index, element) in field_elements.iter().enumerate() {
        if field_elements[..index].contains(element) {
            return Err(GrsError::InvalidFieldEnumeration);
        }
    }
    Ok(())
}

fn check_reference_message_count(field_size: usize, dimension: usize) -> Result<(), GrsError> {
    let mut count = 1usize;
    for _ in 0..dimension {
        count = count
            .checked_mul(field_size)
            .ok_or(GrsError::ArithmeticOverflow)?;
        if count > MAX_REFERENCE_MESSAGES {
            return Err(GrsError::ReferenceInstanceTooLarge);
        }
    }
    Ok(())
}

fn checked_constraint_count(length: usize, multiplicity: usize) -> Result<usize, GrsError> {
    let multiplicity_plus_one = multiplicity
        .checked_add(1)
        .ok_or(GrsError::ArithmeticOverflow)?;
    length
        .checked_mul(multiplicity)
        .and_then(|value| value.checked_mul(multiplicity_plus_one))
        .map(|value| value / 2)
        .ok_or(GrsError::ArithmeticOverflow)
}

fn checked_product(factors: &[u128]) -> Result<u128, GrsError> {
    factors.iter().try_fold(1u128, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(GrsError::ArithmeticOverflow)
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        field::{Field, PrimeField},
        polynomial::Polynomial,
        reference::enumerate_polynomials,
        test_oracle::decode_by_enumeration,
    };

    use super::{GrsCode, GrsError, GuruswamiSudanParameters, check_reference_message_count};

    type F5 = PrimeField<5>;

    fn f5(values: &[u64]) -> Vec<F5> {
        values.iter().copied().map(F5::new).collect()
    }

    fn code_4_2() -> GrsCode<F5> {
        GrsCode::new(f5(&[0, 1, 2, 3]), vec![F5::ONE; 4], 2).expect("valid GRS code")
    }

    #[test]
    fn validates_grs_construction() {
        assert_eq!(
            GrsCode::<F5>::new(vec![], vec![], 1),
            Err(GrsError::EmptyCode)
        );
        assert_eq!(
            GrsCode::new(f5(&[0, 1]), vec![F5::ONE], 1),
            Err(GrsError::MultiplierLengthMismatch)
        );
        assert_eq!(
            GrsCode::new(f5(&[0, 1]), vec![F5::ONE; 2], 0),
            Err(GrsError::ZeroDimension)
        );
        assert_eq!(
            GrsCode::new(f5(&[0, 1]), vec![F5::ONE; 2], 3),
            Err(GrsError::DimensionExceedsLength)
        );
        assert_eq!(
            GrsCode::new(f5(&[0, 0]), vec![F5::ONE; 2], 1),
            Err(GrsError::DuplicateEvaluationPoint)
        );
        assert_eq!(
            GrsCode::new(f5(&[0, 1]), vec![F5::ONE, F5::ZERO], 1),
            Err(GrsError::ZeroMultiplier)
        );
    }

    #[test]
    fn computes_exact_checked_parameter_bounds() {
        let parameters = GuruswamiSudanParameters::new(5, 2, 2, 3).expect("admissible radius");
        assert_eq!(parameters.radius(), 2);
        assert_eq!(parameters.multiplicity(), 3);
        assert_eq!(parameters.weighted_degree_bound(), 7);
        assert_eq!(parameters.radius_exclusive_bound(), 3);
        assert_eq!(parameters.list_size_bound(), Some(7));

        assert_eq!(
            GuruswamiSudanParameters::new(4, 2, 2, 1),
            Err(GrsError::RadiusOutsideInterpolationBound {
                radius: 2,
                exclusive_bound: 2,
            })
        );
        assert_eq!(
            GuruswamiSudanParameters::new(4, 2, 0, 0),
            Err(GrsError::ZeroMultiplicity)
        );
        assert_eq!(
            GuruswamiSudanParameters::new(4, 2, 5, 1),
            Err(GrsError::RadiusExceedsLength)
        );
        assert_eq!(
            GuruswamiSudanParameters::new(0, 0, 0, 1),
            Err(GrsError::EmptyCode)
        );
        assert_eq!(
            GuruswamiSudanParameters::new(4, 0, 0, 1),
            Err(GrsError::ZeroDimension)
        );
        assert_eq!(
            GuruswamiSudanParameters::new(4, 5, 0, 1),
            Err(GrsError::DimensionExceedsLength)
        );
    }

    #[test]
    fn validates_messages_received_words_and_reference_limits() {
        let code = code_4_2();
        assert_eq!(code.length(), 4);
        assert_eq!(code.dimension(), 2);
        assert_eq!(
            code.encode(&Polynomial::from_coefficients(f5(&[1, 2, 3]))),
            Err(GrsError::MessageDegreeTooLarge)
        );
        assert_eq!(
            code.list_decode_reference(&f5(&[0, 0, 0]), 1, 1, &f5(&[0, 1, 2, 3, 4])),
            Err(GrsError::ReceivedWordLengthMismatch)
        );
        assert_eq!(
            code.list_decode_reference(&f5(&[0, 0, 0, 0]), 1, 1, &f5(&[0, 1, 1, 3, 4])),
            Err(GrsError::InvalidFieldEnumeration)
        );
        assert_eq!(
            check_reference_message_count(11, 6),
            Err(GrsError::ReferenceInstanceTooLarge)
        );
    }

    #[test]
    fn constructs_and_verifies_multiplicity_interpolation() {
        let code = code_4_2();
        let message = Polynomial::from_coefficients(f5(&[1, 2]));
        let mut received = code.encode(&message).expect("valid message");
        received[0] = received[0].add(F5::ONE);
        let parameters = GuruswamiSudanParameters::new(4, 2, 1, 2).expect("admissible radius");
        let interpolation = code
            .interpolate(&received, parameters)
            .expect("interpolation polynomial");

        assert!(!interpolation.is_zero());
        assert!(interpolation.satisfies_constraints(
            &f5(&[0, 1, 2, 3]),
            &received,
            parameters.multiplicity()
        ));
        assert!(interpolation.substitute_y(&message).is_zero());
    }

    #[test]
    fn normalizes_nontrivial_column_multipliers() {
        let elements = f5(&[0, 1, 2, 3, 4]);
        let code = GrsCode::new(f5(&[0, 1, 2, 3]), f5(&[1, 2, 3, 4]), 2).expect("valid GRS code");
        let message = Polynomial::from_coefficients(f5(&[3, 1]));
        let mut received = code.encode(&message).expect("valid message");
        received[2] = received[2].add(F5::ONE);

        let decoded = code
            .list_decode_reference(&received, 1, 1, &elements)
            .expect("valid decoder parameters");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].polynomial, message);
        assert_eq!(decoded[0].distance, 1);
    }

    #[test]
    fn decodes_zero_exact_and_beyond_radius_errors() {
        let code = code_4_2();
        let elements = f5(&[0, 1, 2, 3, 4]);
        let message = Polynomial::from_coefficients(f5(&[1, 2]));
        let codeword = code.encode(&message).expect("valid message");

        let exact = code
            .list_decode_reference(&codeword, 1, 1, &elements)
            .expect("valid decoder parameters");
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].polynomial, message);
        assert_eq!(exact[0].distance, 0);
        assert_eq!(exact[0].codeword, codeword);

        let one_error = f5(&[2, 3, 0, 2]);
        let decoded = code
            .list_decode_reference(&one_error, 1, 1, &elements)
            .expect("valid decoder parameters");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].polynomial, message);
        assert_eq!(decoded[0].distance, 1);

        let two_errors = f5(&[0, 1, 0, 2]);
        let decoded = code
            .list_decode_reference(&two_errors, 1, 1, &elements)
            .expect("valid decoder parameters");
        assert!(decoded.is_empty());
    }

    #[test]
    fn returns_empty_and_multiple_lists_deterministically() {
        let elements = f5(&[0, 1, 2, 3, 4]);
        let empty = code_4_2()
            .list_decode_reference(&f5(&[0, 0, 1, 1]), 1, 1, &elements)
            .expect("valid decoder parameters");
        assert!(empty.is_empty());

        let code = GrsCode::new(f5(&[0, 1, 2, 3, 4]), vec![F5::ONE; 5], 2).expect("valid GRS code");
        let multiple = code
            .list_decode_reference(&f5(&[0, 1, 2, 0, 0]), 2, 3, &elements)
            .expect("valid decoder parameters");
        assert_eq!(
            multiple
                .iter()
                .map(|candidate| candidate.polynomial.clone())
                .collect::<Vec<_>>(),
            vec![
                Polynomial::zero(),
                Polynomial::from_coefficients(f5(&[0, 1])),
            ]
        );
        assert!(multiple.iter().all(|candidate| candidate.distance == 2));
    }

    #[test]
    fn handles_dimension_one_without_weight_zero_interpolation() {
        let elements = f5(&[0, 1, 2, 3, 4]);
        let code =
            GrsCode::new(f5(&[0, 1, 2]), vec![F5::ONE; 3], 1).expect("valid constant GRS code");
        let decoded = code
            .list_decode_reference(&f5(&[2, 2, 1]), 1, 1, &elements)
            .expect("valid decoder parameters");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].polynomial, Polynomial::constant(F5::new(2)));
    }

    #[test]
    fn exhaustively_matches_radius_one_hamming_balls() {
        let code = code_4_2();
        let elements = f5(&[0, 1, 2, 3, 4]);

        for received_polynomial in enumerate_polynomials(&elements, code.length()) {
            let received = (0..code.length())
                .map(|index| received_polynomial.coefficient(index))
                .collect::<Vec<_>>();
            let expected =
                decode_by_enumeration(&elements, code.dimension(), &received, 1, |message| {
                    code.encode(message)
                        .expect("enumerated message has valid degree")
                })
                .expect("all GRS words have the expected length");
            let actual = code
                .list_decode_reference(&received, 1, 1, &elements)
                .expect("valid decoder parameters");

            assert_eq!(
                actual
                    .iter()
                    .map(|candidate| candidate.polynomial.clone())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|candidate| candidate.message.clone())
                    .collect::<Vec<_>>(),
                "received word: {received:?}"
            );
            assert!(actual.iter().all(|candidate| candidate.distance <= 1));
        }
    }

    #[test]
    fn exhaustively_matches_constant_code_over_f3() {
        type F3 = PrimeField<3>;
        let elements = [F3::new(0), F3::new(1), F3::new(2)];
        let code =
            GrsCode::new(elements.to_vec(), vec![F3::ONE; 3], 1).expect("valid constant GRS code");

        for received_polynomial in enumerate_polynomials(&elements, code.length()) {
            let received = (0..code.length())
                .map(|index| received_polynomial.coefficient(index))
                .collect::<Vec<_>>();
            let expected =
                decode_by_enumeration(&elements, code.dimension(), &received, 1, |message| {
                    code.encode(message)
                        .expect("enumerated message has valid degree")
                })
                .expect("all GRS words have the expected length");
            let actual = code
                .list_decode_reference(&received, 1, 1, &elements)
                .expect("valid decoder parameters");

            assert_eq!(
                actual
                    .iter()
                    .map(|candidate| candidate.polynomial.clone())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|candidate| candidate.message.clone())
                    .collect::<Vec<_>>(),
                "received word: {received:?}"
            );
        }
    }
}
