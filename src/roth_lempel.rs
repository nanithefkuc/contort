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
/// Guruswami–Sudan on that punctured code, re-encodes each candidate to a full
/// Roth–Lempel codeword, and keeps those within the decoding radius.
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
    pub fn encode_into(
        &self,
        message: &[F::Elem],
        codeword: &mut [F::Elem],
    ) -> Result<(), Error> {
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
}

impl<F: ButterflyKernels> RothLempelScratch<F> {
    /// Construct empty scratch storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decode: DecodeScratch::new(),
            normalized: Vec::new(),
            candidates: Vec::new(),
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
/// punctured coordinates, and everything needed to re-encode a candidate to a
/// full `n`-symbol Roth–Lempel codeword for the distance check.
pub struct RothLempelDecoder<F: ButterflyKernels> {
    plan: GsPlan<F>,
    multipliers: Vec<F::Elem>,
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
        let inverse_multipliers = code.multipliers[..punctured].iter().map(|m| m.inv()).collect();

        Ok(Self {
            plan,
            multipliers: code.multipliers.clone(),
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

    /// List decode a received word into caller-owned output.
    ///
    /// `output` is cleared and filled with the message polynomials (degree
    /// `< k`) of every Roth–Lempel codeword within the decoding radius, in the
    /// punctured decoder's deterministic order. Returns the number of
    /// candidates.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut RothLempelScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        output.clear();
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
                got: received.len(),
            });
        }
        let punctured = self.length - 1;

        scratch.normalized.clear();
        scratch.normalized.reserve(punctured);
        for (symbol, inverse) in received[..punctured].iter().zip(self.inverse_multipliers.iter()) {
            scratch.normalized.push(symbol.mul(*inverse));
        }

        self.plan
            .decode_into(&scratch.normalized, &mut scratch.decode, &mut scratch.candidates)?;

        for candidate in &scratch.candidates {
            if self.full_distance(candidate, received)? <= self.target_radius {
                output.push(candidate.clone());
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
        scratch: &mut RothLempelScratch<F>,
    ) -> Result<UniqueDecode<F>, Error> {
        let mut output = Vec::new();
        let count = self.list_decode_into(received, scratch, &mut output)?;
        Ok(match count {
            0 => UniqueDecode::NoCandidate,
            1 => UniqueDecode::Message(output.pop().unwrap_or_else(Polynomial::zero)),
            _ => UniqueDecode::Ambiguous,
        })
    }

    /// Full Hamming distance between the received word and the Roth–Lempel
    /// codeword of `candidate`, including the exceptional last coordinate.
    fn full_distance(
        &self,
        candidate: &Polynomial<F>,
        received: &[F::Elem],
    ) -> Result<usize, Error> {
        let punctured = self.length - 1;
        let values = candidate.evaluate_many(self.plan.domain().points())?;
        let mut distance = 0;
        for ((value, multiplier), symbol) in values
            .iter()
            .zip(self.multipliers.iter())
            .zip(received[..punctured].iter())
        {
            if multiplier.mul(*value) != *symbol {
                distance += 1;
            }
        }
        let exceptional = candidate
            .coefficient(self.dimension - 2)
            .add(self.twist.mul(candidate.coefficient(self.dimension - 1)));
        if self.multipliers[punctured].mul(exceptional) != received[punctured] {
            distance += 1;
        }
        Ok(distance)
    }
}
