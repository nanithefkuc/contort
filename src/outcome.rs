//! Shared unique-decode outcome for the deformed Reed–Solomon families.

use butterfly_fft::core::kernel::ButterflyKernels;
use gs_engine::Polynomial;

/// Result of a unique decode.
pub enum UniqueDecode<F: ButterflyKernels> {
    /// Exactly one message polynomial (degree `< k`) matched.
    Message(Polynomial<F>),
    /// No codeword lay within the decoding radius.
    NoCandidate,
    /// More than one codeword lay within the decoding radius.
    Ambiguous,
}

impl<F: ButterflyKernels> UniqueDecode<F> {
    /// The decoded message polynomial when the decode was unambiguous.
    #[must_use]
    pub const fn message(&self) -> Option<&Polynomial<F>> {
        match self {
            Self::Message(polynomial) => Some(polynomial),
            _ => None,
        }
    }

    /// Whether the decode produced exactly one message.
    #[must_use]
    pub const fn is_unique(&self) -> bool {
        matches!(self, Self::Message(_))
    }
}
