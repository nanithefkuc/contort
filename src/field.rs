use core::fmt::Debug;

/// Minimal finite-field operations needed by the scalar reference algorithms.
///
/// Implementations must keep elements canonical. Division and inversion return
/// `None` for zero instead of panicking. Converting an integer reduces it in the
/// field, which also provides characteristic-aware integer coefficients.
pub(crate) trait Field: Copy + Debug + Eq {
    const ZERO: Self;
    const ONE: Self;

    fn from_u64(value: u64) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn inverse(self) -> Option<Self>;

    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    fn divide(self, rhs: Self) -> Option<Self> {
        rhs.inverse().map(|inverse| self.mul(inverse))
    }

    fn pow(self, mut exponent: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = self;

        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul(base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(base);
            }
        }

        result
    }
}

/// Simple scalar prime field used by the correctness implementation and its
/// exhaustive small-field tests.
///
/// `MODULUS` must be prime and greater than one. The tuple field is private so
/// every value is canonical and lies in `0..MODULUS`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PrimeField<const MODULUS: u64>(u64);

impl<const MODULUS: u64> PrimeField<MODULUS> {
    pub(crate) fn new(value: u64) -> Self {
        assert!(MODULUS > 1, "prime-field modulus must exceed one");
        Self(value % MODULUS)
    }

    pub(crate) const fn value(self) -> u64 {
        self.0
    }
}

impl<const MODULUS: u64> Field for PrimeField<MODULUS> {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    fn from_u64(value: u64) -> Self {
        Self::new(value)
    }

    fn add(self, rhs: Self) -> Self {
        let sum = (u128::from(self.0) + u128::from(rhs.0)) % u128::from(MODULUS);
        Self(sum as u64)
    }

    fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self(MODULUS - (rhs.0 - self.0))
        }
    }

    fn mul(self, rhs: Self) -> Self {
        let product = (u128::from(self.0) * u128::from(rhs.0)) % u128::from(MODULUS);
        Self(product as u64)
    }

    fn neg(self) -> Self {
        if self.is_zero() {
            self
        } else {
            Self(MODULUS - self.0)
        }
    }

    fn inverse(self) -> Option<Self> {
        (!self.is_zero()).then(|| self.pow(MODULUS - 2))
    }
}

#[cfg(test)]
mod tests {
    use super::{Field, PrimeField};

    type F5 = PrimeField<5>;

    #[test]
    fn construction_canonicalizes_values() {
        assert_eq!(F5::new(0).value(), 0);
        assert_eq!(F5::new(7).value(), 2);
        assert_eq!(F5::from_u64(15), F5::ZERO);
    }

    #[test]
    fn arithmetic_stays_in_the_field() {
        let two = F5::new(2);
        let four = F5::new(4);

        assert_eq!(four.add(two), F5::new(1));
        assert_eq!(two.sub(four), F5::new(3));
        assert_eq!(four.mul(two), F5::new(3));
        assert_eq!(two.neg(), F5::new(3));
        assert_eq!(F5::ZERO.neg(), F5::ZERO);
    }

    #[test]
    fn inversion_and_division_reject_zero() {
        for value in 1..5 {
            let element = F5::new(value);
            let inverse = element.inverse().expect("nonzero field element");
            assert_eq!(element.mul(inverse), F5::ONE);
        }

        assert_eq!(F5::ZERO.inverse(), None);
        assert_eq!(F5::ONE.divide(F5::ZERO), None);
        assert_eq!(F5::new(3).divide(F5::new(2)), Some(F5::new(4)));
    }

    #[test]
    fn exponentiation_handles_zero_and_one_exponents() {
        let three = F5::new(3);
        assert_eq!(three.pow(0), F5::ONE);
        assert_eq!(three.pow(1), three);
        assert_eq!(three.pow(4), F5::ONE);
    }

    #[test]
    #[should_panic(expected = "prime-field modulus must exceed one")]
    fn invalid_modulus_is_rejected() {
        let _ = PrimeField::<1>::new(0);
    }
}
