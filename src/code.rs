//! The twisted generalized Reed–Solomon code and its encoder.

use alloc::vec;
use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{AlekhnovichLimits, EvaluationDomain, ParameterLimits, Polynomial};

use crate::decode::TgrsDecoder;
use crate::error::Error;
use crate::twist::Twist;

/// A twisted generalized Reed–Solomon code `C_TGRS(α, v, k, T)`.
///
/// A message `(f_0, …, f_{k-1})` is mapped to the polynomial
/// `∑ f_i x^i + ∑_j η_j f_{h_j} x^{k-1+t_j}` and then to the codeword
/// `(v_i · f(α_i))_i`. The code is a subcode of the ambient
/// `[n, k']` GRS code, where the *pseudo-dimension* `k' = k + max_j t_j` is the
/// number of monomials the twisted polynomials span. Decoding runs the
/// Guruswami–Sudan decoder on that ambient GRS code and filters the returned
/// polynomials by the twist coefficient constraints.
#[derive(Debug)]
pub struct TgrsCode<F: ButterflyKernels> {
    pub(crate) domain: EvaluationDomain<F>,
    pub(crate) multipliers: Vec<F::Elem>,
    pub(crate) dimension: usize,
    pub(crate) pseudo_dimension: usize,
    pub(crate) twists: Vec<Twist<F>>,
}

impl<F: ButterflyKernels> TgrsCode<F> {
    /// Build a twisted GRS code over an evaluation domain and nonzero column
    /// multipliers.
    ///
    /// `dimension` is the message dimension `k` and must satisfy `1 <= k < n`.
    /// Each twist must satisfy `1 <= t <= n-k`, `0 <= h < k`, `η != 0`, and the
    /// `(t, h)` pairs must be distinct. An empty twist set yields the ordinary
    /// GRS code.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        twists: Vec<Twist<F>>,
    ) -> Result<Self, Error> {
        let length = domain.len();
        if multipliers.len() != length {
            return Err(Error::MultiplierCount {
                expected: length,
                got: multipliers.len(),
            });
        }
        if dimension == 0 || dimension >= length {
            return Err(Error::InvalidDimension { dimension, length });
        }
        for (index, multiplier) in multipliers.iter().enumerate() {
            if multiplier.is_zero() {
                return Err(Error::ZeroMultiplier { index });
            }
        }

        let max_offset = length - dimension;
        let mut pseudo_dimension = dimension;
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(twists.len());
        for (index, twist) in twists.iter().enumerate() {
            let offset = twist.offset();
            let hook = twist.hook();
            if offset < 1 || offset > max_offset {
                return Err(Error::TwistOffset {
                    offset,
                    max: max_offset,
                });
            }
            if hook >= dimension {
                return Err(Error::TwistHook { hook, dimension });
            }
            if twist.coefficient().is_zero() {
                return Err(Error::ZeroTwistCoefficient { index });
            }
            pairs.push((offset, hook));
            pseudo_dimension = pseudo_dimension.max(dimension + offset);
        }
        pairs.sort_unstable();
        for window in pairs.windows(2) {
            if window[0] == window[1] {
                return Err(Error::DuplicateTwist {
                    offset: window[0].0,
                    hook: window[0].1,
                });
            }
        }

        Ok(Self {
            domain,
            multipliers,
            dimension,
            pseudo_dimension,
            twists,
        })
    }

    /// Code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Pseudo-dimension `k' = k + max_j t_j` (equal to `k` without twists).
    #[must_use]
    pub const fn pseudo_dimension(&self) -> usize {
        self.pseudo_dimension
    }

    /// The twist set `T`.
    #[must_use]
    pub fn twists(&self) -> &[Twist<F>] {
        &self.twists
    }

    /// The evaluation domain `α`.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// The column multipliers `v`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// Encode a `k`-symbol message into an `n`-symbol codeword.
    ///
    /// `message` must have exactly `k` symbols and `codeword` exactly `n`.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        let dimension = self.dimension;
        let length = self.length();
        if message.len() != dimension {
            return Err(Error::MessageLength {
                expected: dimension,
                got: message.len(),
            });
        }
        if codeword.len() != length {
            return Err(Error::CodewordLength {
                expected: length,
                got: codeword.len(),
            });
        }

        let mut coefficients = vec![F::Elem::ZERO; self.pseudo_dimension];
        coefficients[..dimension].copy_from_slice(message);
        for twist in &self.twists {
            let destination = dimension - 1 + twist.offset();
            let contribution = twist.coefficient().mul(message[twist.hook()]);
            coefficients[destination] = coefficients[destination].add(contribution);
        }

        let polynomial = Polynomial::<F>::from_coefficients(&coefficients)?;
        let values = polynomial.evaluate_many(self.domain.points())?;
        for (slot, (value, multiplier)) in codeword
            .iter_mut()
            .zip(values.iter().zip(self.multipliers.iter()))
        {
            *slot = multiplier.mul(*value);
        }
        Ok(())
    }

    /// Build a list decoder for a chosen decoding radius.
    ///
    /// The Guruswami–Sudan multiplicity is searched within `parameter_limits`;
    /// root extraction is bounded by `root_limits`.
    pub fn list_decoder(
        &self,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<TgrsDecoder<F>, Error> {
        TgrsDecoder::new(self, target_radius, parameter_limits, root_limits)
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n-k)/2⌋`.
    ///
    /// Combine with [`TgrsDecoder::unique_decode`]. Whether the radius is
    /// feasible depends on the pseudo-dimension; an infeasible geometry surfaces
    /// as [`Error::Configuration`].
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<TgrsDecoder<F>, Error> {
        let radius = (self.length() - self.dimension) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}
