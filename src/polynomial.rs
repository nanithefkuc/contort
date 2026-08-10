use crate::field::Field;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PolynomialError {
    DivisionByZero,
    NonInvertibleLeadingCoefficient,
    PointValueLengthMismatch,
    DuplicateInterpolationPoint,
}

/// Canonical dense polynomial in coefficient order.
///
/// The coefficient at index `i` belongs to `x^i`. Trailing zero coefficients
/// are removed, and the zero polynomial is represented by an empty vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Polynomial<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    pub(crate) fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    pub(crate) fn one() -> Self {
        Self::constant(F::ONE)
    }

    pub(crate) fn constant(value: F) -> Self {
        Self::from_coefficients(vec![value])
    }

    pub(crate) fn from_coefficients(mut coefficients: Vec<F>) -> Self {
        while coefficients
            .last()
            .is_some_and(|coefficient| coefficient.is_zero())
        {
            coefficients.pop();
        }
        Self { coefficients }
    }

    pub(crate) fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    pub(crate) fn coefficient(&self, index: usize) -> F {
        self.coefficients.get(index).copied().unwrap_or(F::ZERO)
    }

    pub(crate) fn degree(&self) -> Option<usize> {
        self.coefficients.len().checked_sub(1)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        let length = self.coefficients.len().max(rhs.coefficients.len());
        let mut coefficients = Vec::with_capacity(length);

        for index in 0..length {
            coefficients.push(self.coefficient(index).add(rhs.coefficient(index)));
        }

        Self::from_coefficients(coefficients)
    }

    pub(crate) fn sub(&self, rhs: &Self) -> Self {
        let length = self.coefficients.len().max(rhs.coefficients.len());
        let mut coefficients = Vec::with_capacity(length);

        for index in 0..length {
            coefficients.push(self.coefficient(index).sub(rhs.coefficient(index)));
        }

        Self::from_coefficients(coefficients)
    }

    pub(crate) fn scale(&self, scalar: F) -> Self {
        if scalar.is_zero() || self.is_zero() {
            return Self::zero();
        }

        Self::from_coefficients(
            self.coefficients
                .iter()
                .map(|coefficient| coefficient.mul(scalar))
                .collect(),
        )
    }

    pub(crate) fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let mut coefficients = vec![F::ZERO; self.coefficients.len() + rhs.coefficients.len() - 1];
        for (left_index, &left) in self.coefficients.iter().enumerate() {
            for (right_index, &right) in rhs.coefficients.iter().enumerate() {
                let output = &mut coefficients[left_index + right_index];
                *output = output.add(left.mul(right));
            }
        }

        Self::from_coefficients(coefficients)
    }

    pub(crate) fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolynomialError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(PolynomialError::DivisionByZero);
        };
        let divisor_leading_inverse = divisor
            .coefficient(divisor_degree)
            .inverse()
            .ok_or(PolynomialError::NonInvertibleLeadingCoefficient)?;

        let Some(dividend_degree) = self.degree() else {
            return Ok((Self::zero(), Self::zero()));
        };
        if dividend_degree < divisor_degree {
            return Ok((Self::zero(), self.clone()));
        }

        let mut quotient = vec![F::ZERO; dividend_degree - divisor_degree + 1];
        let mut remainder = self.clone();

        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }

            let shift = remainder_degree - divisor_degree;
            let factor = remainder
                .coefficient(remainder_degree)
                .mul(divisor_leading_inverse);
            quotient[shift] = quotient[shift].add(factor);

            for index in 0..=divisor_degree {
                let remainder_index = shift + index;
                remainder.coefficients[remainder_index] = remainder.coefficients[remainder_index]
                    .sub(factor.mul(divisor.coefficient(index)));
            }
            remainder.trim();
        }

        Ok((Self::from_coefficients(quotient), remainder))
    }

    pub(crate) fn derivative(&self) -> Self {
        if self.coefficients.len() < 2 {
            return Self::zero();
        }

        let coefficients = self
            .coefficients
            .iter()
            .copied()
            .enumerate()
            .skip(1)
            .map(|(degree, coefficient)| {
                coefficient.mul(F::from_u64(
                    u64::try_from(degree).expect("polynomial degree exceeds u64"),
                ))
            })
            .collect();
        Self::from_coefficients(coefficients)
    }

    pub(crate) fn hasse_derivative(&self, order: usize) -> Self {
        if order == 0 {
            return self.clone();
        }
        if self.degree().is_none_or(|degree| order > degree) {
            return Self::zero();
        }

        let coefficients = self
            .coefficients
            .iter()
            .copied()
            .enumerate()
            .skip(order)
            .map(|(degree, coefficient)| coefficient.mul(binomial_in_field::<F>(degree, order)))
            .collect();
        Self::from_coefficients(coefficients)
    }

    pub(crate) fn evaluate(&self, point: F) -> F {
        self.coefficients
            .iter()
            .rev()
            .fold(F::ZERO, |value, &coefficient| {
                value.mul(point).add(coefficient)
            })
    }

    pub(crate) fn multipoint_evaluate(&self, points: &[F]) -> Vec<F> {
        points.iter().map(|&point| self.evaluate(point)).collect()
    }

    pub(crate) fn compose(&self, inner: &Self) -> Self {
        let mut result = Self::zero();

        for &coefficient in self.coefficients.iter().rev() {
            result = result.mul(inner);
            if coefficient.is_zero() {
                continue;
            }
            if result.coefficients.is_empty() {
                result.coefficients.push(coefficient);
            } else {
                result.coefficients[0] = result.coefficients[0].add(coefficient);
                result.trim();
            }
        }

        result
    }

    pub(crate) fn interpolate(points: &[F], values: &[F]) -> Result<Self, PolynomialError> {
        if points.len() != values.len() {
            return Err(PolynomialError::PointValueLengthMismatch);
        }

        let mut result = Self::zero();
        for (index, (&point, &value)) in points.iter().zip(values).enumerate() {
            let mut basis = Self::one();
            let mut denominator = F::ONE;

            for (other_index, &other_point) in points.iter().enumerate() {
                if index == other_index {
                    continue;
                }
                basis = basis.mul(&Self::from_coefficients(vec![other_point.neg(), F::ONE]));
                denominator = denominator.mul(point.sub(other_point));
            }

            let denominator_inverse = denominator
                .inverse()
                .ok_or(PolynomialError::DuplicateInterpolationPoint)?;
            result = result.add(&basis.scale(value.mul(denominator_inverse)));
        }

        Ok(result)
    }

    fn trim(&mut self) {
        while self
            .coefficients
            .last()
            .is_some_and(|coefficient| coefficient.is_zero())
        {
            self.coefficients.pop();
        }
    }
}

pub(crate) fn binomial_in_field<F: Field>(n: usize, k: usize) -> F {
    if k > n {
        return F::ZERO;
    }

    let k = k.min(n - k);
    let mut row = vec![F::ZERO; k + 1];
    row[0] = F::ONE;

    for current_n in 1..=n {
        for current_k in (1..=k.min(current_n)).rev() {
            row[current_k] = row[current_k].add(row[current_k - 1]);
        }
    }

    row[k]
}

#[cfg(test)]
mod tests {
    use crate::field::{Field, PrimeField};

    use super::{Polynomial, PolynomialError};

    type F5 = PrimeField<5>;
    type F7 = PrimeField<7>;

    fn f5(values: &[u64]) -> Polynomial<F5> {
        Polynomial::from_coefficients(values.iter().copied().map(F5::new).collect())
    }

    fn f7(values: &[u64]) -> Polynomial<F7> {
        Polynomial::from_coefficients(values.iter().copied().map(F7::new).collect())
    }

    #[test]
    fn canonicalizes_zero_and_trailing_coefficients() {
        assert_eq!(f5(&[]), Polynomial::zero());
        assert_eq!(f5(&[0, 0, 0]), Polynomial::zero());
        assert_eq!(f5(&[1, 2, 0, 0]).coefficients(), &[F5::new(1), F5::new(2)]);
        assert_eq!(f5(&[1, 2, 0]).degree(), Some(1));
    }

    #[test]
    fn zero_and_constant_boundaries_are_stable() {
        let zero = Polynomial::<F5>::zero();
        let constant = Polynomial::constant(F5::new(3));

        assert_eq!(zero.add(&constant), constant);
        assert_eq!(constant.sub(&constant), zero);
        assert_eq!(constant.mul(&zero), zero);
        assert_eq!(constant.scale(F5::ZERO), zero);
        assert_eq!(constant.derivative(), zero);
        assert_eq!(constant.hasse_derivative(1), zero);
        assert_eq!(constant.evaluate(F5::new(4)), F5::new(3));
    }

    #[test]
    fn arithmetic_obeys_polynomial_identities() {
        let left = f7(&[1, 2, 3]);
        let right = f7(&[4, 0, 1]);
        let third = f7(&[2, 5]);

        assert_eq!(left.add(&right).sub(&right), left);
        assert_eq!(left.mul(&Polynomial::one()), left);
        assert_eq!(
            left.mul(&right.add(&third)),
            left.mul(&right).add(&left.mul(&third))
        );
        assert_eq!(left.scale(F7::new(3)), f7(&[3, 6, 2]));
    }

    #[test]
    fn division_reports_exact_and_non_exact_remainders() {
        let divisor = f7(&[1, 1]);
        let exact = f7(&[2, 3, 1]);
        let (quotient, remainder) = exact.div_rem(&divisor).expect("valid divisor");
        assert_eq!(quotient, f7(&[2, 1]));
        assert!(remainder.is_zero());

        let non_exact = f7(&[1, 0, 1]);
        let (quotient, remainder) = non_exact.div_rem(&divisor).expect("valid divisor");
        assert_eq!(quotient, f7(&[6, 1]));
        assert_eq!(remainder, f7(&[2]));
        assert_eq!(quotient.mul(&divisor).add(&remainder), non_exact);
    }

    #[test]
    fn division_handles_small_dividends_and_zero_divisor() {
        let constant = f7(&[3]);
        let linear = f7(&[1, 1]);
        assert_eq!(
            constant.div_rem(&linear),
            Ok((Polynomial::zero(), constant.clone()))
        );
        assert_eq!(
            constant.div_rem(&Polynomial::zero()),
            Err(PolynomialError::DivisionByZero)
        );
        assert_eq!(
            Polynomial::<F7>::zero().div_rem(&linear),
            Ok((Polynomial::zero(), Polynomial::zero()))
        );
    }

    #[test]
    fn derivative_and_hasse_derivative_respect_characteristic() {
        let polynomial = f5(&[0, 0, 0, 0, 0, 1]);
        assert_eq!(polynomial.derivative(), Polynomial::zero());
        assert_eq!(polynomial.hasse_derivative(1), Polynomial::zero());
        assert_eq!(polynomial.hasse_derivative(5), Polynomial::one());
        assert_eq!(polynomial.hasse_derivative(0), polynomial);
    }

    #[test]
    fn evaluation_and_composition_cover_zero() {
        let outer = f7(&[1, 0, 1]);
        let inner = f7(&[1, 1]);

        assert_eq!(outer.evaluate(F7::ZERO), F7::ONE);
        assert_eq!(outer.compose(&inner), f7(&[2, 2, 1]));
        assert_eq!(Polynomial::<F7>::zero().compose(&inner), Polynomial::zero());
        assert_eq!(f7(&[1, 1]).compose(&f7(&[6])), Polynomial::zero());
    }

    #[test]
    fn naive_multipoint_evaluation_and_interpolation_round_trip() {
        let polynomial = f7(&[3, 5, 2, 1]);
        let points = [F7::new(0), F7::new(1), F7::new(2), F7::new(4)];
        let values = polynomial.multipoint_evaluate(&points);

        assert_eq!(values[0], F7::new(3));
        assert_eq!(Polynomial::interpolate(&points, &values), Ok(polynomial));
        assert_eq!(
            Polynomial::<F7>::interpolate(&[], &[]),
            Ok(Polynomial::zero())
        );
    }

    #[test]
    fn interpolation_rejects_malformed_samples() {
        let points = [F7::new(1), F7::new(1)];
        let values = [F7::new(2), F7::new(3)];
        assert_eq!(
            Polynomial::interpolate(&points, &values),
            Err(PolynomialError::DuplicateInterpolationPoint)
        );
        assert_eq!(
            Polynomial::interpolate(&points[..1], &values),
            Err(PolynomialError::PointValueLengthMismatch)
        );
    }
}
