//! The Roth–Lempel code and its Guruswami–Sudan decoder.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{EvaluationDomain, ParameterLimits};
use poly_ring::{AlekhnovichLimits, Polynomial};

use crate::error::Error;
use crate::extend::{ExtendedGrsCode, ExtendedGrsDecoder, ExtendedGrsScratch};
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
///
/// This is the single-functional extended GRS code with the exceptional
/// functional `λ = e_{k-2} + δ · e_{k-1}`; it is a thin view over
/// [`ExtendedGrsCode`].
#[derive(Debug)]
pub struct RothLempelCode<F: ButterflyKernels> {
    inner: ExtendedGrsCode<F>,
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

        let mut functional = alloc::vec![F::Elem::ZERO; dimension];
        functional[dimension - 2] = F::Elem::ONE;
        functional[dimension - 1] = twist;
        let inner = ExtendedGrsCode::new(domain, multipliers, dimension, alloc::vec![functional])?;
        Ok(Self { inner, twist })
    }

    /// Code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.inner.length()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The twist `δ`.
    #[must_use]
    pub fn twist(&self) -> F::Elem {
        self.twist
    }

    /// The `n-1` evaluation points `α`.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        self.inner.domain()
    }

    /// The column multipliers `v`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        self.inner.multipliers()
    }

    /// Encode a `k`-symbol message into an `n`-symbol codeword.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        self.inner.encode_into(message, codeword)
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
        let inner = self
            .inner
            .list_decoder(target_radius, parameter_limits, root_limits)?;
        Ok(RothLempelDecoder { inner })
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n-k)/2⌋`.
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<RothLempelDecoder<F>, Error> {
        let radius = (self.length() - self.dimension()) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}

/// Reusable working storage for repeated Roth–Lempel decodes.
///
/// This is exactly the extended-GRS scratch; Roth–Lempel is the
/// single-functional extended code.
pub type RothLempelScratch<F> = ExtendedGrsScratch<F>;

/// A validated Roth–Lempel decoder bound to one decoding radius.
///
/// A thin view over [`ExtendedGrsDecoder`]: it wraps the punctured GRS
/// [`GsPlan`], the inverse multipliers for the `n-1` punctured coordinates, and
/// the exceptional-coordinate data needed to combine the punctured distance
/// with the one-symbol exceptional check.
///
/// [`GsPlan`]: gs_engine::GsPlan
pub struct RothLempelDecoder<F: ButterflyKernels> {
    inner: ExtendedGrsDecoder<F>,
}

impl<F: ButterflyKernels> RothLempelDecoder<F> {
    /// The decoding radius this decoder was built for.
    #[must_use]
    pub const fn target_radius(&self) -> usize {
        self.inner.target_radius()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Code length `n`.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.inner.length()
    }

    /// Reserve every reusable buffer for this decoder's maximum geometry.
    ///
    /// After this call a warmed decode over `scratch` performs no internal heap
    /// allocation. The caller-owned `output` of [`list_decode_into`] is warmed
    /// by a single worst-case decode.
    ///
    /// [`list_decode_into`]: Self::list_decode_into
    pub fn prepare_scratch(&self, scratch: &mut RothLempelScratch<F>) -> Result<(), Error> {
        self.inner.prepare_scratch(scratch)
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
        self.inner.list_decode_into(received, scratch, output)
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
        self.inner.unique_decode(received, scratch)
    }
}
