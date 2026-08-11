//! A single twist applied to a generalized Reed–Solomon code.

use core::fmt;

use fgf::kernel::FieldKernels;

/// One twist `(t, h, η)` of a twisted generalized Reed–Solomon code.
///
/// A twist maps the free message coefficient `f_h` (the *hook*) onto the
/// higher monomial `x^{k-1+t}` scaled by `η`. In the twisted polynomial space
/// every codeword polynomial carries `η · f_h` at degree `k-1+t` in addition to
/// its free coefficients `f_0, …, f_{k-1}`.
///
/// The offset `t` and hook `h` are validated against a concrete code by
/// [`TgrsCode::new`](crate::TgrsCode::new); `Twist` itself only stores the
/// triple.
pub struct Twist<F: FieldKernels> {
    offset: usize,
    hook: usize,
    coefficient: F::Elem,
}

impl<F: FieldKernels> Twist<F> {
    /// Construct a twist from its degree offset `t`, hook `h`, and coefficient
    /// `η`.
    #[must_use]
    pub const fn new(offset: usize, hook: usize, coefficient: F::Elem) -> Self {
        Self {
            offset,
            hook,
            coefficient,
        }
    }

    /// The degree offset `t`, placing the twist term at degree `k-1+t`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// The hook `h`, the message coefficient this twist copies.
    #[must_use]
    pub const fn hook(&self) -> usize {
        self.hook
    }

    /// The twist coefficient `η`.
    #[must_use]
    pub fn coefficient(&self) -> F::Elem {
        self.coefficient
    }
}

impl<F: FieldKernels> Clone for Twist<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldKernels> Copy for Twist<F> {}

impl<F: FieldKernels> PartialEq for Twist<F> {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.hook == other.hook && self.coefficient == other.coefficient
    }
}

impl<F: FieldKernels> Eq for Twist<F> {}

impl<F: FieldKernels> fmt::Debug for Twist<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Twist")
            .field("offset", &self.offset)
            .field("hook", &self.hook)
            .field("coefficient", &self.coefficient)
            .finish()
    }
}
