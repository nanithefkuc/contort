//! Extended generalized Reed–Solomon codes and their Guruswami–Sudan decoder.
//!
//! An extended GRS code appends `L` extra coordinates to an ordinary GRS code.
//! The first `n_base` coordinates evaluate a message polynomial
//! `f ∈ F_q[x]_{<k}` as a GRS code over `α = (α_1, …, α_{n_base})` scaled by the
//! base column multipliers; each extra coordinate `n_base + j` is a linear
//! functional `λ_j` of the message coefficients, scaled by its own multiplier:
//! `v_{n_base+j} · Σ_{d<k} λ_j[d] · f_d`.
//!
//! Puncturing all `L` extended coordinates yields the GRS code
//! `C_GRS(α, v', k)`, so decoding runs Guruswami–Sudan on that base code and
//! combines each candidate's punctured distance with the re-encoded mismatch on
//! the extended coordinates. This is the generalized Roth–Lempel reduction with
//! `L` exceptional coordinates instead of one.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits};
use poly_ring::{AlekhnovichLimits, Polynomial};

use crate::error::Error;
use crate::eval::horner;
use crate::outcome::UniqueDecode;

/// An extended generalized Reed–Solomon code `C(α, v, k, {λ_j})`.
///
/// The base evaluation domain `α` holds `n_base` points; the code length is
/// `n = n_base + L`, where `L` is the number of extension functionals.
/// `multipliers` has `n` entries — one per codeword coordinate, base and
/// extended alike. Each functional `λ_j` is a length-`k` coefficient vector
/// evaluated as `λ_j(f) = Σ_{d<k} λ_j[d] · f_d`. An empty functional list gives
/// an ordinary GRS code.
#[derive(Debug)]
pub struct ExtendedGrsCode<F: ButterflyKernels> {
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    functionals: Vec<Vec<F::Elem>>,
}

impl<F: ButterflyKernels> ExtendedGrsCode<F> {
    /// Build an extended GRS code from a base domain of `n_base` points, `n`
    /// nonzero column multipliers, dimension `k`, and `L` extension
    /// functionals.
    ///
    /// The code length is `n = n_base + L`. The dimension must satisfy
    /// `1 <= k <= n_base`, and every functional must be a length-`k` coefficient
    /// vector. An empty functional list is permitted and yields an ordinary GRS
    /// code.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        functionals: Vec<Vec<F::Elem>>,
    ) -> Result<Self, Error> {
        let base_length = domain.len();
        let length = base_length + functionals.len();
        if multipliers.len() != length {
            return Err(Error::MultiplierCount {
                expected: length,
                got: multipliers.len(),
            });
        }
        for (index, multiplier) in multipliers.iter().enumerate() {
            if multiplier.is_zero() {
                return Err(Error::ZeroMultiplier { index });
            }
        }
        if dimension < 1 || dimension > base_length {
            return Err(Error::InvalidDimension {
                dimension,
                length: base_length,
            });
        }
        for functional in &functionals {
            if functional.len() != dimension {
                return Err(Error::FunctionalLength {
                    expected: dimension,
                    got: functional.len(),
                });
            }
        }
        Ok(Self {
            domain,
            multipliers,
            dimension,
            functionals,
        })
    }

    /// Build the projective extension: a single functional `e_{k-1}`, the
    /// evaluation-at-infinity coordinate that reads the leading coefficient
    /// `f_{k-1}`. The resulting `[n_base + 1, k]` code is MDS.
    pub fn projective(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
    ) -> Result<Self, Error> {
        if dimension < 1 {
            return Err(Error::InvalidDimension {
                dimension,
                length: domain.len(),
            });
        }
        let mut functional = alloc::vec![F::Elem::ZERO; dimension];
        functional[dimension - 1] = F::Elem::ONE;
        Self::new(domain, multipliers, dimension, alloc::vec![functional])
    }

    /// Code length `n = n_base + L`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len() + self.functionals.len()
    }

    /// Base code length `n_base` (the number of evaluation points).
    #[must_use]
    pub fn base_length(&self) -> usize {
        self.domain.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// The column multipliers `v` (length `n`).
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// The base evaluation points `α` (length `n_base`).
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// The extension functionals `λ_0, …, λ_{L-1}`, each a length-`k` vector.
    #[must_use]
    pub fn functionals(&self) -> &[Vec<F::Elem>] {
        &self.functionals
    }

    /// Encode a `k`-symbol message into an `n`-symbol codeword.
    ///
    /// The first `n_base` coordinates are `v_i · f(α_i)` by Horner evaluation;
    /// coordinate `n_base + j` is `v_{n_base+j} · Σ_{d<k} λ_j[d] · f_d`. No heap
    /// allocation occurs.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        let dimension = self.dimension;
        let base_length = self.domain.len();
        let length = base_length + self.functionals.len();
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

        for (slot, (point, multiplier)) in codeword[..base_length]
            .iter_mut()
            .zip(self.domain.points().iter().zip(self.multipliers.iter()))
        {
            *slot = multiplier.mul(horner::<F>(message, *point));
        }

        for (j, functional) in self.functionals.iter().enumerate() {
            let mut value = F::Elem::ZERO;
            for (coefficient, symbol) in functional.iter().zip(message.iter()) {
                value = value.add(coefficient.mul(*symbol));
            }
            codeword[base_length + j] = self.multipliers[base_length + j].mul(value);
        }
        Ok(())
    }

    /// Build a list decoder for a chosen decoding radius.
    ///
    /// The Guruswami–Sudan multiplicity is searched within `parameter_limits`;
    /// root extraction is bounded by `root_limits`. The radius must be feasible
    /// for the base `[n_base, k]` GRS code.
    pub fn list_decoder(
        &self,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<ExtendedGrsDecoder<F>, Error> {
        ExtendedGrsDecoder::new(self, target_radius, parameter_limits, root_limits)
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n-k)/2⌋`.
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<ExtendedGrsDecoder<F>, Error> {
        let radius = (self.length() - self.dimension) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}

/// Reusable working storage for repeated extended-GRS decodes.
pub struct ExtendedGrsScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    candidates: Vec<Polynomial<F>>,
    distances: Vec<usize>,
    filtered: Vec<Polynomial<F>>,
}

impl<F: ButterflyKernels> ExtendedGrsScratch<F> {
    /// Construct empty scratch storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decode: DecodeScratch::new(),
            normalized: Vec::new(),
            candidates: Vec::new(),
            distances: Vec::new(),
            filtered: Vec::new(),
        }
    }
}

impl<F: ButterflyKernels> Default for ExtendedGrsScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated extended-GRS decoder bound to one decoding radius.
///
/// Holds the base GRS [`GsPlan`], the inverse multipliers for the `n_base` base
/// coordinates, and the `(functional, multiplier)` pairs for the `L` extended
/// coordinates needed to combine the punctured distance with the re-encoded
/// extended-coordinate mismatches.
pub struct ExtendedGrsDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    inverse_multipliers: Vec<F::Elem>,
    extensions: Vec<(Vec<F::Elem>, F::Elem)>,
    dimension: usize,
    length: usize,
    target_radius: usize,
}

impl<F: ButterflyKernels> ExtendedGrsDecoder<F> {
    pub(crate) fn new(
        code: &ExtendedGrsCode<F>,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, Error> {
        let base_length = code.base_length();
        let parameters = GsParameters::search::<F>(
            base_length,
            code.dimension - 1,
            target_radius,
            parameter_limits,
        )?;
        let plan = GsPlan::new(parameters, code.domain.clone(), root_limits)?;
        let inverse_multipliers = code.multipliers[..base_length]
            .iter()
            .map(|m| m.inv())
            .collect();
        let extensions = code
            .functionals
            .iter()
            .enumerate()
            .map(|(j, functional)| (functional.clone(), code.multipliers[base_length + j]))
            .collect();

        Ok(Self {
            plan,
            inverse_multipliers,
            extensions,
            dimension: code.dimension,
            length: code.length(),
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

    /// Code length `n`.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    /// Reserve every reusable buffer for this decoder's maximum geometry.
    ///
    /// After this call a warmed decode over `scratch` performs no internal heap
    /// allocation. The caller-owned `output` of [`list_decode_into`] is warmed
    /// by a single worst-case decode.
    ///
    /// [`list_decode_into`]: Self::list_decode_into
    pub fn prepare_scratch(&self, scratch: &mut ExtendedGrsScratch<F>) -> Result<(), Error> {
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.candidates)?;
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.filtered)?;
        scratch
            .normalized
            .reserve(self.length - self.extensions.len());
        scratch.distances.reserve(self.plan.parameters().y_degree());
        Ok(())
    }

    /// List decode a received word into caller-owned output.
    ///
    /// Runs the scored Guruswami–Sudan decode on the base `[n_base, k]` GRS
    /// code, which returns each candidate together with its exact punctured
    /// Hamming distance. The full distance is that punctured distance plus the
    /// number of extended coordinates whose re-encoded value disagrees with the
    /// received symbol, so no candidate is re-evaluated on the base code.
    /// `output` retains and overwrites its storage, then is truncated to the
    /// candidate count, so a warmed decode does not reallocate it. Returns the
    /// number of candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut ExtendedGrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
                got: received.len(),
            });
        }
        let base_length = self.length - self.extensions.len();

        scratch.normalized.clear();
        scratch.normalized.reserve(base_length);
        for (symbol, inverse) in received[..base_length]
            .iter()
            .zip(self.inverse_multipliers.iter())
        {
            scratch.normalized.push(symbol.mul(*inverse));
        }

        self.plan.decode_scored_into(
            &scratch.normalized,
            &mut scratch.decode,
            &mut scratch.candidates,
            &mut scratch.distances,
        )?;

        let mut count = 0;
        for (candidate, &punctured_distance) in
            scratch.candidates.iter().zip(scratch.distances.iter())
        {
            let mut extended_distance = 0;
            for (j, (functional, multiplier)) in self.extensions.iter().enumerate() {
                let mut value = F::Elem::ZERO;
                for (d, coefficient) in functional.iter().enumerate() {
                    value = value.add(coefficient.mul(candidate.coefficient(d)));
                }
                if multiplier.mul(value) != received[base_length + j] {
                    extended_distance += 1;
                }
            }
            if punctured_distance + extended_distance <= self.target_radius {
                write_candidate(output, count, candidate);
                count += 1;
            }
        }
        output.truncate(count);
        Ok(count)
    }

    /// Uniquely decode a received word.
    ///
    /// Returns [`UniqueDecode::Message`] when exactly one codeword lies within
    /// the decoding radius, and [`UniqueDecode::NoCandidate`] or
    /// [`UniqueDecode::Ambiguous`] otherwise. Filtering reuses scratch-owned
    /// storage; `Ambiguous` and `NoCandidate` allocate nothing once warmed.
    pub fn unique_decode(
        &self,
        received: &[F::Elem],
        scratch: &mut ExtendedGrsScratch<F>,
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
}

/// Write `candidate` into `output[index]`, reusing retained storage.
fn write_candidate<F: ButterflyKernels>(
    output: &mut Vec<Polynomial<F>>,
    index: usize,
    candidate: &Polynomial<F>,
) {
    if index < output.len() {
        output[index].clone_from(candidate);
    } else {
        output.push(candidate.clone());
    }
}
