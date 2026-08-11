//! Deformed Reed–Solomon codes: twisted, folded, and interleaved generalized
//! Reed–Solomon and the Roth–Lempel family.
//!
//! A deformed code starts from an ordinary Reed–Solomon evaluation code and
//! applies a structural deformation — a twist, a fold, or an interleave — to
//! obtain a new (potentially non-GRS) code. `contort` owns the deformation
//! itself: the code construction, the reduction of a received word to a
//! Guruswami–Sudan interpolation problem, and the admissibility filtering of
//! the returned message polynomials. The list-decoding machinery — parameter
//! search, interpolation, and root extraction — comes from [`gs_engine`];
//! finite-field arithmetic comes from [`fgf`].
//!
//! The implemented families are twisted GRS ([`TgrsCode`]) — decoded on the ambient
//! GRS code of pseudo-dimension `k'` with a twist-coefficient filter — and
//! Roth–Lempel ([`RothLempelCode`]) — decoded by puncturing the exceptional
//! coordinate, running Guruswami–Sudan on the resulting GRS code, and
//! re-encoding candidates to check the full Hamming distance (Zhu–Jin).

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

mod code;
mod decode;
mod error;
mod outcome;
mod roth_lempel;
mod twist;

pub use code::TgrsCode;
pub use decode::{TgrsDecoder, TgrsScratch};
pub use error::Error;
pub use outcome::UniqueDecode;
pub use roth_lempel::{RothLempelCode, RothLempelDecoder, RothLempelScratch};
pub use twist::Twist;

pub use gs_engine::{AlekhnovichLimits, EvaluationDomain, ParameterLimits, Polynomial};
