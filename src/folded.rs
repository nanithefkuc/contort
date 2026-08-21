//! Folded Reed–Solomon construction: multiplicative-orbit geometry, encoder,
//! and the block-Hamming metric.
//!
//! An `m`-folded code evaluates a degree-`< k` polynomial over the consecutive
//! multiplicative orbit `γ^0, γ^1, …, γ^{n-1}` and groups every `m` consecutive
//! scalar symbols into one alphabet symbol of `F_q^m`. The scalar codeword is
//! therefore an ordinary GRS codeword over the orbit; folding is a
//! reinterpretation of its metric, from scalar Hamming to block Hamming. This
//! module owns that construction and metric only — the capacity list decoder is
//! an upstream `gs-engine`/`gfm` capability (see the roadmap), so no decoder
//! lives here.
//!
//! The canonical wire layout is block-major, component-minor: block `i` is the
//! slice `[i·m, (i+1)·m)`, which is exactly orbit order.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;

use crate::error::Error;
use crate::eval::horner;

/// A folded Reed–Solomon code over a multiplicative orbit.
///
/// The code length `n` is the number of column multipliers; the fold `m`
/// divides `n` into `N = n/m` blocks. A message `(f_0, …, f_{k-1})` maps to the
/// scalar codeword `(v_i · f(γ^i))_{i}` in block-major order.
#[derive(Debug)]
pub struct FoldedRsCode<F: ButterflyKernels> {
    orbit: Vec<F::Elem>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    fold: usize,
}

impl<F: ButterflyKernels> FoldedRsCode<F> {
    /// Build an `m`-folded code from a generator `γ`, `n` nonzero column
    /// multipliers, dimension `k`, and fold `m`.
    ///
    /// # Errors
    ///
    /// - [`Error::InsufficientOrbit`] if `γ` is zero or its order is below `n`
    ///   (the first `n` powers are not distinct);
    /// - [`Error::ZeroMultiplier`] for a zero multiplier;
    /// - [`Error::InvalidDimension`] unless `1 ≤ k < n`;
    /// - [`Error::FoldParameter`] unless `m ≥ 1` and `m` divides `n`.
    pub fn new(
        generator: F::Elem,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        fold: usize,
    ) -> Result<Self, Error> {
        let length = multipliers.len();
        if generator.is_zero() {
            return Err(Error::InsufficientOrbit { length });
        }
        let mut orbit = Vec::with_capacity(length);
        let mut power = F::Elem::ONE;
        for index in 0..length {
            if index > 0 && power == F::Elem::ONE {
                // The orbit returned to γ^0 before reaching n distinct powers.
                return Err(Error::InsufficientOrbit { length });
            }
            orbit.push(power);
            power = power.mul(generator);
        }
        if dimension == 0 || dimension >= length {
            return Err(Error::InvalidDimension { dimension, length });
        }
        for (index, multiplier) in multipliers.iter().enumerate() {
            if multiplier.is_zero() {
                return Err(Error::ZeroMultiplier { index });
            }
        }
        if fold == 0 || length % fold != 0 {
            return Err(Error::FoldParameter { fold, length });
        }
        Ok(Self {
            orbit,
            multipliers,
            dimension,
            fold,
        })
    }

    /// Code length `n` (scalar symbols).
    #[must_use]
    pub fn length(&self) -> usize {
        self.multipliers.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Fold `m` (scalar symbols per block).
    #[must_use]
    pub const fn fold(&self) -> usize {
        self.fold
    }

    /// Block length `N = n/m`.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.length() / self.fold
    }

    /// The multiplicative orbit `γ^0, …, γ^{n-1}`.
    #[must_use]
    pub fn orbit(&self) -> &[F::Elem] {
        &self.orbit
    }

    /// The column multipliers `v`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// Encode a `k`-symbol message into the `n`-symbol block-major codeword.
    ///
    /// Allocation-free: the message is evaluated at each orbit point by Horner
    /// and scaled into the caller's `codeword` slice.
    ///
    /// # Errors
    ///
    /// [`Error::MessageLength`] or [`Error::CodewordLength`] on a slice whose
    /// length is not `k` or `n`.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        if message.len() != self.dimension {
            return Err(Error::MessageLength {
                expected: self.dimension,
                got: message.len(),
            });
        }
        if codeword.len() != self.length() {
            return Err(Error::CodewordLength {
                expected: self.length(),
                got: codeword.len(),
            });
        }
        for ((slot, point), multiplier) in codeword
            .iter_mut()
            .zip(self.orbit.iter())
            .zip(self.multipliers.iter())
        {
            *slot = multiplier.mul(horner::<F>(message, *point));
        }
        Ok(())
    }

    /// Block `i` of a block-major word: the slice `[i·m, (i+1)·m)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N` or `word` is shorter than `(i+1)·m`.
    #[must_use]
    pub fn block<'w>(&self, word: &'w [F::Elem], index: usize) -> &'w [F::Elem] {
        let start = index * self.fold;
        &word[start..start + self.fold]
    }

    /// Folded (block-Hamming) distance between two block-major words: the number
    /// of `m`-symbol blocks in which they differ, no matter how many scalar
    /// components inside a block differ.
    #[must_use]
    pub fn block_distance(&self, a: &[F::Elem], b: &[F::Elem]) -> usize {
        a.chunks_exact(self.fold)
            .zip(b.chunks_exact(self.fold))
            .filter(|(x, y)| x != y)
            .count()
    }
}
