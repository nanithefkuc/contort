//! Punctured generalized Reed–Solomon codes and their Guruswami–Sudan decoder.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

use crate::error::Error;
use crate::eval::horner;
use crate::outcome::UniqueDecode;

/// A punctured generalized Reed–Solomon code `C_punc(α, v, k, S)`.
///
/// The base code is the GRS code `C_GRS(α, v, k)` on the `n`-point evaluation
/// domain `α` with `n` nonzero column multipliers `v` and dimension `k`. The
/// puncture set `S ⊆ {0, …, n-1}` deletes coordinates: a codeword keeps only
/// the surviving coordinates in ascending original index order, so for each
/// `i ∉ S` the surviving symbol is `v_i · f(α_i)`. The effective length is
/// `n' = n - |S|`.
///
/// Puncturing a GRS code yields exactly the GRS code `C_GRS(α_S, v_S, k)` on
/// the surviving points with the surviving multipliers, so decoding runs
/// Guruswami–Sudan on that subdomain with no candidate filter — every ambient
/// candidate's low-`k` coefficients are a valid message.
///
/// `domain` holds the `n` base evaluation points; `punctures` is stored sorted
/// and deduplicated.
#[derive(Debug)]
pub struct PuncturedGrsCode<F: ButterflyKernels> {
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    punctures: Vec<usize>,
}

impl<F: ButterflyKernels> PuncturedGrsCode<F> {
    /// Build a punctured GRS code from `n` evaluation points, `n` nonzero
    /// column multipliers, dimension `k`, and a puncture set `S`.
    ///
    /// The dimension must satisfy `1 <= k < n`. Every puncture index must lie
    /// in `0..n` and appear at most once, and the surviving count `n - |S|`
    /// must be at least `k`. The puncture set is sorted and deduplicated before
    /// being stored.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        punctures: Vec<usize>,
    ) -> Result<Self, Error> {
        let length = domain.len();
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
        if dimension < 1 || dimension >= length {
            return Err(Error::InvalidDimension { dimension, length });
        }

        let mut punctures = punctures;
        punctures.sort_unstable();
        let mut previous: Option<usize> = None;
        for &index in &punctures {
            if index >= length {
                return Err(Error::PunctureIndex { index, length });
            }
            if previous == Some(index) {
                return Err(Error::DuplicatePuncture { index });
            }
            previous = Some(index);
        }

        let remaining = length - punctures.len();
        if remaining < dimension {
            return Err(Error::PunctureLength {
                remaining,
                dimension,
            });
        }

        Ok(Self {
            domain,
            multipliers,
            dimension,
            punctures,
        })
    }

    /// Effective code length `n' = n - |S|` after puncturing.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len() - self.punctures.len()
    }

    /// Base code length `n` before puncturing.
    #[must_use]
    pub fn base_length(&self) -> usize {
        self.domain.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// The base column multipliers `v`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// The base evaluation points `α`.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// The puncture set `S`, sorted ascending and deduplicated.
    #[must_use]
    pub fn punctures(&self) -> &[usize] {
        &self.punctures
    }

    /// Whether base coordinate `index` belongs to the puncture set.
    fn is_punctured(&self, index: usize) -> bool {
        self.punctures.binary_search(&index).is_ok()
    }

    /// Encode a `k`-symbol message into an `n'`-symbol punctured codeword.
    ///
    /// Surviving coordinates are written in ascending original index order:
    /// for each `i ∉ S` the next output slot receives `v_i · f(α_i)`.
    /// Allocation-free.
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

        let points = self.domain.points();
        let mut puncture_cursor = 0;
        let mut slot = 0;
        for (i, &point) in points.iter().enumerate() {
            if puncture_cursor < self.punctures.len() && self.punctures[puncture_cursor] == i {
                puncture_cursor += 1;
                continue;
            }
            codeword[slot] = self.multipliers[i].mul(horner::<F>(message, point));
            slot += 1;
        }
        Ok(())
    }

    /// Whether the puncture set is aligned to blocks of size `fold`.
    ///
    /// Returns `true` iff `fold > 0`, `fold` divides the base length `n`, and
    /// every block `[b·fold, (b+1)·fold)` is either entirely punctured or
    /// entirely surviving. This is the composed-fold capability flag: a
    /// block-aligned puncture commutes with an `fold`-fold, so the punctured
    /// code can itself be folded on the surviving blocks.
    #[must_use]
    pub fn is_block_aligned(&self, fold: usize) -> bool {
        let length = self.domain.len();
        if fold == 0 || !length.is_multiple_of(fold) {
            return false;
        }
        for block in 0..length / fold {
            let start = block * fold;
            let first = self.is_punctured(start);
            for index in start + 1..start + fold {
                if self.is_punctured(index) != first {
                    return false;
                }
            }
        }
        true
    }

    /// Build a list decoder for a chosen decoding radius.
    ///
    /// The Guruswami–Sudan multiplicity is searched within `parameter_limits`;
    /// root extraction is bounded by `root_limits`. The radius must be feasible
    /// for the punctured `[n', k]` GRS code.
    pub fn list_decoder(
        &self,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<PuncturedGrsDecoder<F>, Error> {
        PuncturedGrsDecoder::new(self, target_radius, parameter_limits, root_limits)
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n'-k)/2⌋`.
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<PuncturedGrsDecoder<F>, Error> {
        let radius = (self.length() - self.dimension) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}

/// Reusable working storage for repeated punctured GRS decodes.
pub struct PuncturedGrsScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    ambient: Vec<Polynomial<F>>,
    filtered: Vec<Polynomial<F>>,
    message: Vec<F::Elem>,
}

impl<F: ButterflyKernels> PuncturedGrsScratch<F> {
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

impl<F: ButterflyKernels> Default for PuncturedGrsScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated punctured GRS decoder bound to one decoding radius.
///
/// Holds the surviving-subdomain GRS [`GsPlan`] together with the inverse
/// multipliers for the surviving coordinates. The punctured code is exactly
/// the GRS code on the surviving points, so decoding runs Guruswami–Sudan on
/// that plan with no candidate filter.
pub struct PuncturedGrsDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    inverse_multipliers: Vec<F::Elem>,
    dimension: usize,
    length: usize,
    target_radius: usize,
}

impl<F: ButterflyKernels> PuncturedGrsDecoder<F> {
    pub(crate) fn new(
        code: &PuncturedGrsCode<F>,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, Error> {
        let length = code.length();
        let points = code.domain.points();
        let mut surviving_points = Vec::with_capacity(length);
        let mut inverse_multipliers = Vec::with_capacity(length);
        let mut puncture_cursor = 0;
        for (i, &point) in points.iter().enumerate() {
            if puncture_cursor < code.punctures.len() && code.punctures[puncture_cursor] == i {
                puncture_cursor += 1;
                continue;
            }
            surviving_points.push(point);
            inverse_multipliers.push(code.multipliers[i].inv());
        }

        let parameters =
            GsParameters::search::<F>(length, code.dimension - 1, target_radius, parameter_limits)?;
        let subdomain = EvaluationDomain::arbitrary(surviving_points)?;
        let plan = GsPlan::new(parameters, subdomain, root_limits)?;

        Ok(Self {
            plan,
            inverse_multipliers,
            dimension: code.dimension,
            length,
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

    /// Effective code length `n'`.
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
    pub fn prepare_scratch(&self, scratch: &mut PuncturedGrsScratch<F>) -> Result<(), Error> {
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.ambient)?;
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.filtered)?;
        scratch.normalized.reserve(self.length);
        scratch.message.reserve(self.dimension);
        Ok(())
    }

    /// List decode a received word into caller-owned output.
    ///
    /// Runs the Guruswami–Sudan decode on the surviving-subdomain `[n', k]` GRS
    /// code. Because puncturing a GRS code is again GRS, every ambient
    /// candidate is a valid message, so no filter is applied: each candidate's
    /// degree-`< k` message is written to `output`. `output` retains and
    /// overwrites its storage, then is truncated to the candidate count, so a
    /// warmed decode does not reallocate it. Returns the number of candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut PuncturedGrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
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
            self.write_message(&mut scratch.message, output, count, candidate)?;
            count += 1;
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
    /// storage; `Ambiguous` and `NoCandidate` allocate nothing once warmed.
    pub fn unique_decode(
        &self,
        received: &[F::Elem],
        scratch: &mut PuncturedGrsScratch<F>,
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

    /// Normalize the received word by the inverse surviving multipliers.
    fn normalize_into(&self, received: &[F::Elem], scratch: &mut PuncturedGrsScratch<F>) {
        scratch.normalized.clear();
        scratch.normalized.reserve(received.len());
        for (symbol, inverse) in received.iter().zip(self.inverse_multipliers.iter()) {
            scratch.normalized.push(symbol.mul(*inverse));
        }
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
