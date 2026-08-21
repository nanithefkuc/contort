//! Shared allocation-free polynomial evaluation.
//!
//! Encoders in this crate evaluate a message polynomial at fixed domain points
//! and scale by column multipliers. Horner evaluation over the caller's message
//! slice needs no intermediate `Polynomial` and no heap, which is what the
//! folded and interleaved encoders rely on for a zero-allocation steady state.

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;

/// Evaluate the polynomial with the given low-to-high `coefficients` at `point`
/// by Horner's method.
#[must_use]
pub(crate) fn horner<F: ButterflyKernels>(coefficients: &[F::Elem], point: F::Elem) -> F::Elem {
    let mut acc = F::Elem::ZERO;
    for &coefficient in coefficients.iter().rev() {
        acc = acc.mul(point).add(coefficient);
    }
    acc
}
