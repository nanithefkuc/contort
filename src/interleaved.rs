//! Homogeneous interleaved Reed–Solomon construction: column-major layout,
//! encoder, and the column-Hamming metric.
//!
//! An `ℓ`-interleaved code stacks `ℓ` independent `[n, k]` GRS codewords over a
//! shared evaluation domain and groups corresponding coordinates so one channel
//! symbol is a column in `F_q^ℓ`. This module owns the construction, the
//! column-major wire layout, and the column metric only. The collaborative
//! decoder — its common-locator solve and its random / semi-adversarial error
//! model — is an upstream `gs-engine`/`gfm` capability (see the roadmap), so no
//! decoder lives here.
//!
//! Canonical layout: column-major, so each channel symbol is contiguous —
//! `[f_0(α_0), …, f_{ℓ-1}(α_0), f_0(α_1), …, f_{ℓ-1}(α_1), …]`.

use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::EvaluationDomain;

use crate::error::Error;
use crate::eval::horner;

/// A homogeneous interleaved Reed–Solomon code of order `ℓ`.
///
/// All rows share the evaluation domain `α`, the column multipliers `v`, and
/// the dimension `k`. A batch of `ℓ` messages, each `k` symbols, maps to the
/// `ℓ·n`-symbol column-major codeword `(v_i · f_r(α_i))` with `r` fastest.
#[derive(Debug)]
pub struct InterleavedRsCode<F: ButterflyKernels> {
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    order: usize,
}

impl<F: ButterflyKernels> InterleavedRsCode<F> {
    /// Build an order-`ℓ` interleaved code over an evaluation domain and `n`
    /// nonzero column multipliers.
    ///
    /// # Errors
    ///
    /// - [`Error::MultiplierCount`] if `v` is not `n` long;
    /// - [`Error::ZeroMultiplier`] for a zero multiplier;
    /// - [`Error::InvalidDimension`] unless `1 ≤ k < n`;
    /// - [`Error::ZeroInterleave`] if `ℓ = 0`.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        order: usize,
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
        if order == 0 {
            return Err(Error::ZeroInterleave);
        }
        Ok(Self {
            domain,
            multipliers,
            dimension,
            order,
        })
    }

    /// Code length `n` (columns).
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Interleaving order `ℓ` (rows).
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
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

    /// Encode `ℓ` row messages into the `ℓ·n`-symbol column-major codeword.
    ///
    /// `messages` is the row-major message block of `ℓ·k` symbols (row `r` is
    /// `messages[r·k .. (r+1)·k]`); `codeword` is `ℓ·n` symbols column-major.
    /// Allocation-free.
    ///
    /// # Errors
    ///
    /// [`Error::MessageCount`] / [`Error::CodewordLength`] on a mis-sized batch.
    pub fn encode_into(&self, messages: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        let k = self.dimension;
        let n = self.length();
        let ell = self.order;
        if messages.len() != ell * k {
            return Err(Error::MessageCount {
                expected: ell,
                got: if k == 0 {
                    messages.len()
                } else {
                    messages.len() / k
                },
            });
        }
        if codeword.len() != ell * n {
            return Err(Error::CodewordLength {
                expected: ell * n,
                got: codeword.len(),
            });
        }
        let points = self.domain.points();
        for (column, (&point, multiplier)) in points.iter().zip(self.multipliers.iter()).enumerate()
        {
            for row in 0..ell {
                let message = &messages[row * k..row * k + k];
                codeword[column * ell + row] = multiplier.mul(horner::<F>(message, point));
            }
        }
        Ok(())
    }

    /// Column `i` of a column-major word: the slice `[i·ℓ, (i+1)·ℓ)`.
    ///
    /// # Panics
    ///
    /// Panics if `word` is shorter than `(i+1)·ℓ`.
    #[must_use]
    pub fn column<'w>(&self, word: &'w [F::Elem], index: usize) -> &'w [F::Elem] {
        let start = index * self.order;
        &word[start..start + self.order]
    }

    /// Interleaved (column-Hamming) distance between two column-major words: the
    /// number of `ℓ`-symbol columns in which they differ.
    #[must_use]
    pub fn column_distance(&self, a: &[F::Elem], b: &[F::Elem]) -> usize {
        a.chunks_exact(self.order)
            .zip(b.chunks_exact(self.order))
            .filter(|(x, y)| x != y)
            .count()
    }
}
