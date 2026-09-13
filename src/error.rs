//! Error type shared by the deformed Reed–Solomon families.

use core::fmt;

use gs_engine::{ConfigError, DecodeError, DomainError};

/// Failure while building, encoding, or decoding a deformed GRS code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The column-multiplier vector length disagreed with the domain length.
    MultiplierCount {
        /// Evaluation-domain length `n`.
        expected: usize,
        /// Supplied multiplier count.
        got: usize,
    },
    /// A column multiplier was zero; every multiplier must be nonzero.
    ZeroMultiplier {
        /// Index of the offending multiplier.
        index: usize,
    },
    /// The dimension `k` was zero or not strictly below the code length `n`.
    InvalidDimension {
        /// Requested dimension `k`.
        dimension: usize,
        /// Evaluation-domain length `n`.
        length: usize,
    },
    /// The dimension `k` was below the family minimum (Roth–Lempel needs
    /// `k >= 2` for its exceptional coordinate).
    MinimumDimension {
        /// Requested dimension `k`.
        dimension: usize,
        /// Smallest admissible dimension.
        minimum: usize,
    },
    /// A twist degree offset `t` was outside the range `1..=n-k`.
    TwistOffset {
        /// Offending offset `t`.
        offset: usize,
        /// Largest admissible offset `n-k`.
        max: usize,
    },
    /// A twist hook `h` was not strictly below the dimension `k`.
    TwistHook {
        /// Offending hook `h`.
        hook: usize,
        /// Dimension `k`.
        dimension: usize,
    },
    /// A twist coefficient `η` was zero.
    ZeroTwistCoefficient {
        /// Index of the offending twist.
        index: usize,
    },
    /// Two twists shared the same `(t, h)` pair.
    DuplicateTwist {
        /// Shared offset `t`.
        offset: usize,
        /// Shared hook `h`.
        hook: usize,
    },
    /// A message slice did not have exactly `k` symbols.
    MessageLength {
        /// Expected dimension `k`.
        expected: usize,
        /// Supplied message length.
        got: usize,
    },
    /// A codeword slice did not have exactly `n` symbols.
    CodewordLength {
        /// Expected length `n`.
        expected: usize,
        /// Supplied codeword length.
        got: usize,
    },
    /// A received slice did not have exactly `n` symbols.
    ReceivedLength {
        /// Expected length `n`.
        expected: usize,
        /// Supplied received length.
        got: usize,
    },
    /// The generator does not produce `length` distinct powers, so it cannot
    /// index a multiplicative orbit of that length.
    InsufficientOrbit {
        /// Number of distinct orbit points required.
        length: usize,
    },
    /// The fold `m` was zero or did not divide the code length `n`.
    FoldParameter {
        /// The offending fold value.
        fold: usize,
        /// The code length it must divide.
        length: usize,
    },
    /// The interleaving order `ℓ` was zero.
    ZeroInterleave,
    /// A batch of row messages did not have exactly `ℓ` entries.
    MessageCount {
        /// Interleaving order `ℓ`.
        expected: usize,
        /// Number of row messages supplied.
        got: usize,
    },
    /// A puncture set left fewer than `k` surviving coordinates.
    PunctureLength {
        /// Surviving coordinate count `n - |S|`.
        remaining: usize,
        /// Dimension `k`.
        dimension: usize,
    },
    /// A puncture index was outside `0..n`.
    PunctureIndex {
        /// Offending index.
        index: usize,
        /// Base code length `n`.
        length: usize,
    },
    /// A puncture set listed the same coordinate twice.
    DuplicatePuncture {
        /// Repeated index.
        index: usize,
    },
    /// The Möbius map was singular (`Δ = ad - bc = 0`).
    MobiusDelta,
    /// The Möbius pole `-d/c` coincided with a domain evaluation point.
    MobiusPole {
        /// Index of the domain point hit by the pole.
        index: usize,
    },
    /// An extension functional did not have exactly `k` coefficients.
    FunctionalLength {
        /// Expected dimension `k`.
        expected: usize,
        /// Supplied functional length.
        got: usize,
    },
    /// Ring polynomial arithmetic failed.
    Polynomial(poly_ring::PolynomialError),
    /// The ambient Guruswami–Sudan configuration was infeasible.
    Configuration(ConfigError),
    /// The ambient Guruswami–Sudan decode failed.
    Decoding(DecodeError),
    /// Evaluation-domain construction failed (e.g. repeated points).
    Domain(DomainError),
}

impl From<ConfigError> for Error {
    fn from(error: ConfigError) -> Self {
        Self::Configuration(error)
    }
}

impl From<DecodeError> for Error {
    fn from(error: DecodeError) -> Self {
        Self::Decoding(error)
    }
}

impl From<DomainError> for Error {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<poly_ring::PolynomialError> for Error {
    fn from(error: poly_ring::PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<poly_ring::ConfigError> for Error {
    fn from(error: poly_ring::ConfigError) -> Self {
        Self::Polynomial(poly_ring::PolynomialError::Config(error))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultiplierCount { expected, got } => write!(
                formatter,
                "column-multiplier count {got} does not match domain length {expected}"
            ),
            Self::ZeroMultiplier { index } => {
                write!(formatter, "column multiplier at index {index} is zero")
            }
            Self::InvalidDimension { dimension, length } => write!(
                formatter,
                "dimension {dimension} must satisfy 1 <= k < n for length {length}"
            ),
            Self::MinimumDimension { dimension, minimum } => write!(
                formatter,
                "dimension {dimension} is below the family minimum {minimum}"
            ),
            Self::TwistOffset { offset, max } => {
                write!(formatter, "twist offset {offset} is outside 1..={max}")
            }
            Self::TwistHook { hook, dimension } => {
                write!(
                    formatter,
                    "twist hook {hook} is not below dimension {dimension}"
                )
            }
            Self::ZeroTwistCoefficient { index } => {
                write!(formatter, "twist coefficient at index {index} is zero")
            }
            Self::DuplicateTwist { offset, hook } => {
                write!(formatter, "twists share the pair (t={offset}, h={hook})")
            }
            Self::MessageLength { expected, got } => {
                write!(
                    formatter,
                    "message length {got} does not match dimension {expected}"
                )
            }
            Self::CodewordLength { expected, got } => {
                write!(
                    formatter,
                    "codeword length {got} does not match code length {expected}"
                )
            }
            Self::ReceivedLength { expected, got } => {
                write!(
                    formatter,
                    "received length {got} does not match code length {expected}"
                )
            }
            Self::InsufficientOrbit { length } => write!(
                formatter,
                "generator does not produce {length} distinct orbit powers"
            ),
            Self::FoldParameter { fold, length } => write!(
                formatter,
                "fold {fold} must be nonzero and divide code length {length}"
            ),
            Self::ZeroInterleave => write!(formatter, "interleaving order must be nonzero"),
            Self::MessageCount { expected, got } => write!(
                formatter,
                "row-message count {got} does not match interleaving order {expected}"
            ),
            Self::PunctureLength {
                remaining,
                dimension,
            } => write!(
                formatter,
                "puncture leaves {remaining} coordinates, below dimension {dimension}"
            ),
            Self::PunctureIndex { index, length } => {
                write!(formatter, "puncture index {index} is outside 0..{length}")
            }
            Self::DuplicatePuncture { index } => {
                write!(formatter, "puncture set repeats coordinate {index}")
            }
            Self::MobiusDelta => write!(formatter, "Möbius map is singular (Δ = 0)"),
            Self::MobiusPole { index } => {
                write!(formatter, "Möbius pole coincides with domain point {index}")
            }
            Self::FunctionalLength { expected, got } => write!(
                formatter,
                "extension functional length {got} does not match dimension {expected}"
            ),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Configuration(error) => write!(formatter, "ambient GS configuration: {error}"),
            Self::Decoding(error) => write!(formatter, "ambient GS decode: {error}"),
            Self::Domain(error) => write!(formatter, "evaluation domain: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
