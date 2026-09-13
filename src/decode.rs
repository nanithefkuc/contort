//! Guruswami–Sudan list and unique decoding for twisted GRS codes.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{DecodeScratch, GsParameters, GsPlan, ParameterLimits};
use poly_ring::{AlekhnovichLimits, Polynomial};

use crate::code::TgrsCode;
use crate::error::Error;
use crate::outcome::UniqueDecode;

/// One twist contribution grouped by its destination degree.
#[derive(Clone, Copy)]
struct HookTerm<F: ButterflyKernels> {
    hook: usize,
    coefficient: F::Elem,
}

/// Reusable working storage for repeated decodes sharing one decoder.
pub struct TgrsScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    ambient: Vec<Polynomial<F>>,
    filtered: Vec<Polynomial<F>>,
    message: Vec<F::Elem>,
}

impl<F: ButterflyKernels> TgrsScratch<F> {
    /// Construct empty scratch storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decode: DecodeScratch::new(),
            normalized: Vec::new(),
            ambient: Vec::new(),
            filtered: Vec::new(),
            message: Vec::new(),
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
/// and filter candidates. The constraints are stored in flat compressed-row
/// form: `constraint_terms[constraint_offsets[j]..constraint_offsets[j + 1]]`
/// are the hook terms whose destination degree is `k + j`.
pub struct TgrsDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    inverse_multipliers: Vec<F::Elem>,
    dimension: usize,
    pseudo_dimension: usize,
    constraint_offsets: Vec<usize>,
    constraint_terms: Vec<HookTerm<F>>,
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
        let destinations = pseudo_dimension - dimension;

        // Count terms per destination, then prefix-sum into flat offsets.
        let mut counts = alloc::vec![0usize; destinations];
        for twist in &code.twists {
            let destination = dimension - 1 + twist.offset();
            counts[destination - dimension] += 1;
        }
        let mut constraint_offsets = alloc::vec![0usize; destinations + 1];
        for j in 0..destinations {
            constraint_offsets[j + 1] = constraint_offsets[j] + counts[j];
        }
        let mut fill = constraint_offsets.clone();
        let mut constraint_terms = alloc::vec![
            HookTerm { hook: 0, coefficient: F::Elem::ZERO };
            constraint_offsets[destinations]
        ];
        for twist in &code.twists {
            let slot = twist.offset() - 1;
            constraint_terms[fill[slot]] = HookTerm {
                hook: twist.hook(),
                coefficient: twist.coefficient(),
            };
            fill[slot] += 1;
        }

        Ok(Self {
            plan,
            inverse_multipliers,
            dimension,
            pseudo_dimension,
            constraint_offsets,
            constraint_terms,
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

    /// Reserve every reusable buffer for this decoder's maximum geometry.
    ///
    /// After this call a warmed decode over `scratch` performs no internal heap
    /// allocation. The caller-owned `output` of [`list_decode_into`] must be
    /// warmed separately (its capacity is the caller's; a single decode of a
    /// worst-case word warms it).
    ///
    /// [`list_decode_into`]: Self::list_decode_into
    pub fn prepare_scratch(&self, scratch: &mut TgrsScratch<F>) -> Result<(), Error> {
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.ambient)?;
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.filtered)?;
        scratch.normalized.reserve(self.length());
        scratch.message.reserve(self.dimension);
        Ok(())
    }

    /// List decode a received word into caller-owned output.
    ///
    /// `output` retains and overwrites its existing polynomial storage, then is
    /// truncated to the candidate count, so a warmed decode does not reallocate
    /// it. On return `output` holds the message polynomials (degree `< k`) of
    /// every codeword within the decoding radius, in the ambient decoder's
    /// deterministic order. Returns the number of candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut TgrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length() {
            return Err(Error::ReceivedLength {
                expected: self.length(),
                got: received.len(),
            });
        }

        self.normalize_into(received, scratch);
        self.plan.decode_into(
            &scratch.normalized,
            &mut scratch.decode,
            &mut scratch.ambient,
        )?;

        let ambient = core::mem::take(&mut scratch.ambient);
        let mut count = 0;
        for candidate in &ambient {
            if self.satisfies_twists(candidate) {
                self.write_message(&mut scratch.message, output, count, candidate)?;
                count += 1;
            }
        }
        scratch.ambient = ambient;
        output.truncate(count);
        Ok(count)
    }

    /// Uniquely decode a received word.
    ///
    /// Returns [`UniqueDecode::Message`] when exactly one codeword lies within
    /// the decoding radius, and [`UniqueDecode::NoCandidate`] or
    /// [`UniqueDecode::Ambiguous`] otherwise. Filtering reuses scratch-owned
    /// storage; the `Ambiguous` and `NoCandidate` outcomes allocate nothing
    /// once warmed.
    pub fn unique_decode(
        &self,
        received: &[F::Elem],
        scratch: &mut TgrsScratch<F>,
    ) -> Result<UniqueDecode<F>, Error> {
        let mut filtered = core::mem::take(&mut scratch.filtered);
        let count = self.list_decode_into(received, scratch, &mut filtered)?;
        let outcome = match count {
            0 => UniqueDecode::NoCandidate,
            1 => UniqueDecode::Message(filtered[0].clone()),
            _ => UniqueDecode::Ambiguous,
        };
        scratch.filtered = filtered;
        Ok(outcome)
    }

    /// Normalize the received word by the inverse column multipliers.
    fn normalize_into(&self, received: &[F::Elem], scratch: &mut TgrsScratch<F>) {
        scratch.normalized.clear();
        scratch.normalized.reserve(received.len());
        for (symbol, inverse) in received.iter().zip(self.inverse_multipliers.iter()) {
            scratch.normalized.push(symbol.mul(*inverse));
        }
    }

    /// Whether an ambient candidate obeys every twist coefficient constraint.
    fn satisfies_twists(&self, candidate: &Polynomial<F>) -> bool {
        for j in 0..self.constraint_offsets.len() - 1 {
            let destination = self.dimension + j;
            let mut required = F::Elem::ZERO;
            for term in
                &self.constraint_terms[self.constraint_offsets[j]..self.constraint_offsets[j + 1]]
            {
                required = required.add(term.coefficient.mul(candidate.coefficient(term.hook)));
            }
            if candidate.coefficient(destination) != required {
                return false;
            }
        }
        true
    }

    /// Write the degree-`< k` message of `candidate` into `output[index]`,
    /// reusing retained storage where possible.
    ///
    /// The warmed path overwrites an existing output polynomial in place with
    /// [`set_coefficient`](Polynomial::set_coefficient) and truncates it to `k`
    /// coefficients, reusing its buffer. Only while the output vector is still
    /// growing does it build a fresh polynomial.
    fn write_message(
        &self,
        message: &mut Vec<F::Elem>,
        output: &mut Vec<Polynomial<F>>,
        index: usize,
        candidate: &Polynomial<F>,
    ) -> Result<(), Error> {
        if index < output.len() {
            let polynomial = &mut output[index];
            for degree in 0..self.dimension {
                polynomial.set_coefficient(degree, candidate.coefficient(degree))?;
            }
            polynomial.truncate(self.dimension);
        } else {
            message.clear();
            for degree in 0..self.dimension {
                message.push(candidate.coefficient(degree));
            }
            output.push(Polynomial::from_coefficients(message)?);
        }
        Ok(())
    }
}
