//! Guruswami–Sudan list and unique decoding for twisted GRS codes.

use alloc::vec;
use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, GsParameters, GsPlan, ParameterLimits, Polynomial,
};

use crate::code::TgrsCode;
use crate::error::Error;
use crate::outcome::UniqueDecode;

/// One twist contribution grouped by its destination degree.
struct HookTerm<F: ButterflyKernels> {
    hook: usize,
    coefficient: F::Elem,
}

/// Reusable working storage for repeated decodes sharing one decoder.
pub struct TgrsScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    ambient: Vec<Polynomial<F>>,
}

impl<F: ButterflyKernels> TgrsScratch<F> {
    /// Construct empty scratch storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decode: DecodeScratch::new(),
            normalized: Vec::new(),
            ambient: Vec::new(),
        }
    }
}

impl<F: ButterflyKernels> Default for TgrsScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated twisted GRS decoder bound to one decoding radius.
///
/// Holds the ambient GRS [`GsPlan`] together with the inverse column
/// multipliers and the twist constraints needed to normalize the received word
/// and filter candidates.
pub struct TgrsDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    inverse_multipliers: Vec<F::Elem>,
    dimension: usize,
    pseudo_dimension: usize,
    twists_by_destination: Vec<Vec<HookTerm<F>>>,
    target_radius: usize,
}

impl<F: ButterflyKernels> TgrsDecoder<F> {
    pub(crate) fn new(
        code: &TgrsCode<F>,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, Error> {
        let length = code.length();
        let parameters = GsParameters::search::<F>(
            length,
            code.pseudo_dimension - 1,
            target_radius,
            parameter_limits,
        )?;
        let plan = GsPlan::new(parameters, code.domain.clone(), root_limits)?;

        let inverse_multipliers = code.multipliers.iter().map(|m| m.inv()).collect();

        let dimension = code.dimension;
        let pseudo_dimension = code.pseudo_dimension;
        let mut twists_by_destination: Vec<Vec<HookTerm<F>>> =
            (0..pseudo_dimension - dimension).map(|_| Vec::new()).collect();
        for twist in &code.twists {
            let destination = dimension - 1 + twist.offset();
            twists_by_destination[destination - dimension].push(HookTerm {
                hook: twist.hook(),
                coefficient: twist.coefficient(),
            });
        }

        Ok(Self {
            plan,
            inverse_multipliers,
            dimension,
            pseudo_dimension,
            twists_by_destination,
            target_radius,
        })
    }

    /// The decoding radius this decoder was built for.
    #[must_use]
    pub const fn target_radius(&self) -> usize {
        self.target_radius
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Pseudo-dimension `k'` of the ambient GRS code.
    #[must_use]
    pub const fn pseudo_dimension(&self) -> usize {
        self.pseudo_dimension
    }

    /// Code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.inverse_multipliers.len()
    }

    /// List decode a received word into caller-owned output.
    ///
    /// `output` is cleared and filled with the message polynomials (degree
    /// `< k`) of every codeword within the decoding radius, in the ambient
    /// decoder's deterministic order. Returns the number of candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut TgrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        output.clear();
        let length = self.length();
        if received.len() != length {
            return Err(Error::ReceivedLength {
                expected: length,
                got: received.len(),
            });
        }

        scratch.normalized.clear();
        scratch.normalized.reserve(length);
        for (symbol, inverse) in received.iter().zip(self.inverse_multipliers.iter()) {
            scratch.normalized.push(symbol.mul(*inverse));
        }

        self.plan
            .decode_into(&scratch.normalized, &mut scratch.decode, &mut scratch.ambient)?;

        for candidate in &scratch.ambient {
            if self.satisfies_twists(candidate) {
                output.push(self.message_polynomial(candidate)?);
            }
        }
        Ok(output.len())
    }

    /// Uniquely decode a received word.
    ///
    /// Returns [`UniqueDecode::Message`] when exactly one codeword lies within
    /// the decoding radius, and [`UniqueDecode::NoCandidate`] or
    /// [`UniqueDecode::Ambiguous`] otherwise.
    pub fn unique_decode(
        &self,
        received: &[F::Elem],
        scratch: &mut TgrsScratch<F>,
    ) -> Result<UniqueDecode<F>, Error> {
        let mut output = Vec::new();
        let count = self.list_decode_into(received, scratch, &mut output)?;
        Ok(match count {
            0 => UniqueDecode::NoCandidate,
            1 => UniqueDecode::Message(output.pop().unwrap_or_else(Polynomial::zero)),
            _ => UniqueDecode::Ambiguous,
        })
    }

    /// Whether an ambient candidate obeys every twist coefficient constraint.
    fn satisfies_twists(&self, candidate: &Polynomial<F>) -> bool {
        for (index, terms) in self.twists_by_destination.iter().enumerate() {
            let destination = self.dimension + index;
            let mut required = F::Elem::ZERO;
            for term in terms {
                required = required.add(term.coefficient.mul(candidate.coefficient(term.hook)));
            }
            if candidate.coefficient(destination) != required {
                return false;
            }
        }
        true
    }

    /// Extract the degree-`< k` message polynomial from an ambient candidate.
    fn message_polynomial(&self, candidate: &Polynomial<F>) -> Result<Polynomial<F>, Error> {
        let mut coefficients = vec![F::Elem::ZERO; self.dimension];
        for (degree, slot) in coefficients.iter_mut().enumerate() {
            *slot = candidate.coefficient(degree);
        }
        Ok(Polynomial::from_coefficients(&coefficients)?)
    }
}
