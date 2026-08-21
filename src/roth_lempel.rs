//! The Roth–Lempel code and its Guruswami–Sudan decoder.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{
    AlekhnovichLimits, DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits,
    Polynomial,
};

use crate::error::Error;
use crate::outcome::UniqueDecode;

/// A Roth–Lempel code `C_RL(α, v, k, δ)`.
///
/// The first `n-1` coordinates evaluate a message polynomial `f ∈ F_q[x]_{<k}`
/// as an ordinary GRS code over `α = (α_1, …, α_{n-1})`; the exceptional last
/// coordinate is `v_n · (f_{k-2} + δ · f_{k-1})`. Puncturing the last coordinate
/// yields the GRS code `C_GRS(α, v', k)` (Lemma 7, Zhu–Jin), so decoding runs
/// Guruswami–Sudan on that punctured code and combines each candidate's
/// punctured distance with the one-symbol exceptional check.
///
/// `domain` holds the `n-1` evaluation points; the code length is
/// `n = domain.len() + 1`. `multipliers` has `n` entries, one per codeword
/// coordinate including the exceptional last. The message space is the full
/// `F_q[x]_{<k}`, so unlike twisted GRS there is no coefficient filter — only a
/// re-encoded Hamming-distance check.
#[derive(Debug)]
pub struct RothLempelCode<F: ButterflyKernels> {
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    twist: F::Elem,
}

impl<F: ButterflyKernels> RothLempelCode<F> {
    /// Build a Roth–Lempel code from `n-1` evaluation points, `n` nonzero
    /// column multipliers, dimension `k`, and the twist `δ`.
    ///
    /// The dimension must satisfy `2 <= k < n` (the exceptional coordinate reads
    /// coefficients `f_{k-2}` and `f_{k-1}`). Any `δ` is accepted; whether the
    /// resulting code is non-GRS depends on `δ` and is not enforced here.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        twist: F::Elem,
    ) -> Result<Self, Error> {
        let length = domain.len() + 1;
        if multipliers.len() != length {
            return Err(Error::MultiplierCount {
                expected: length,
                got: multipliers.len(),
            });
        }
        if dimension < 2 {
            return Err(Error::MinimumDimension {
                dimension,
                minimum: 2,
            });
        }
        if dimension >= length {
            return Err(Error::InvalidDimension { dimension, length });
        }
        for (index, multiplier) in multipliers.iter().enumerate() {
            if multiplier.is_zero() {
                return Err(Error::ZeroMultiplier { index });
            }
        }
        Ok(Self {
            domain,
            multipliers,
            dimension,
            twist,
        })
    }

    /// Code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len() + 1
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// The twist `δ`.
    #[must_use]
    pub fn twist(&self) -> F::Elem {
        self.twist
    }

    /// The `n-1` evaluation points `α`.
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
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        let dimension = self.dimension;
        let length = self.length();
        let punctured = self.domain.len();
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

        let polynomial = Polynomial::<F>::from_coefficients(message)?;
        let values = polynomial.evaluate_many(self.domain.points())?;
        for (slot, (value, multiplier)) in codeword[..punctured]
            .iter_mut()
            .zip(values.iter().zip(self.multipliers.iter()))
        {
            *slot = multiplier.mul(*value);
        }

        let exceptional = message[dimension - 2].add(self.twist.mul(message[dimension - 1]));
        codeword[punctured] = self.multipliers[punctured].mul(exceptional);
        Ok(())
    }

    /// Build a list decoder for a chosen decoding radius.
    ///
    /// The Guruswami–Sudan multiplicity is searched within `parameter_limits`;
    /// root extraction is bounded by `root_limits`. The radius must be feasible
    /// for the punctured `[n-1, k]` GRS code.
    pub fn list_decoder(
        &self,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<RothLempelDecoder<F>, Error> {
        RothLempelDecoder::new(self, target_radius, parameter_limits, root_limits)
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n-k)/2⌋`.
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<RothLempelDecoder<F>, Error> {
        let radius = (self.length() - self.dimension) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}

/// Reusable working storage for repeated Roth–Lempel decodes.
pub struct RothLempelScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    candidates: Vec<Polynomial<F>>,
    distances: Vec<usize>,
    filtered: Vec<Polynomial<F>>,
}

impl<F: ButterflyKernels> RothLempelScratch<F> {
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

impl<F: ButterflyKernels> Default for RothLempelScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated Roth–Lempel decoder bound to one decoding radius.
///
/// Holds the punctured GRS [`GsPlan`], the inverse multipliers for the `n-1`
/// punctured coordinates, and the exceptional-coordinate data needed to combine
/// the punctured distance with the one-symbol exceptional check.
pub struct RothLempelDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    exceptional_multiplier: F::Elem,
    inverse_multipliers: Vec<F::Elem>,
    dimension: usize,
    twist: F::Elem,
    length: usize,
    target_radius: usize,
}

impl<F: ButterflyKernels> RothLempelDecoder<F> {
    pub(crate) fn new(
        code: &RothLempelCode<F>,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, Error> {
        let punctured = code.domain.len();
        let parameters = GsParameters::search::<F>(
            punctured,
            code.dimension - 1,
            target_radius,
            parameter_limits,
        )?;
        let plan = GsPlan::new(parameters, code.domain.clone(), root_limits)?;
        let inverse_multipliers = code.multipliers[..punctured]
            .iter()
            .map(|m| m.inv())
            .collect();

        Ok(Self {
            plan,
            exceptional_multiplier: code.multipliers[punctured],
            inverse_multipliers,
            dimension: code.dimension,
            twist: code.twist,
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
    pub fn prepare_scratch(&self, scratch: &mut RothLempelScratch<F>) -> Result<(), Error> {
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.candidates)?;
        self.plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.filtered)?;
        scratch.normalized.reserve(self.length - 1);
        scratch.distances.reserve(self.plan.parameters().y_degree());
        Ok(())
    }

    /// List decode a received word into caller-owned output.
    ///
    /// Runs the scored Guruswami–Sudan decode on the punctured `[n-1, k]` GRS
    /// code, which returns each candidate together with its exact punctured
    /// Hamming distance. The full Roth–Lempel distance is that punctured
    /// distance plus the one-symbol exceptional mismatch, so no candidate is
    /// ever re-evaluated. `output` retains and overwrites its storage, then is
    /// truncated to the candidate count, so a warmed decode does not reallocate
    /// it. Returns the number of candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut RothLempelScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
                got: received.len(),
            });
        }
        let punctured = self.length - 1;

        scratch.normalized.clear();
        scratch.normalized.reserve(punctured);
        for (symbol, inverse) in received[..punctured]
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
            let exceptional = candidate
                .coefficient(self.dimension - 2)
                .add(self.twist.mul(candidate.coefficient(self.dimension - 1)));
            let mismatch =
                usize::from(self.exceptional_multiplier.mul(exceptional) != received[punctured]);
            if punctured_distance + mismatch <= self.target_radius {
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
        scratch: &mut RothLempelScratch<F>,
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
