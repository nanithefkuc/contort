//! Deformed Reed–Solomon codes: twisted, folded, interleaved, punctured,
//! Möbius-transformed, and extended (Roth–Lempel) generalized Reed–Solomon.
//!
//! A deformed code starts from an ordinary Reed–Solomon evaluation code and
//! applies a structural deformation — a twist, a fold, an interleave, a domain
//! edit (puncture, Möbius relabelling, or extension) — to obtain a new
//! (potentially non-GRS) code. `contort` owns the deformation itself: the code
//! construction, the reduction of a received word to a Guruswami–Sudan
//! interpolation problem, and the admissibility filtering of the returned
//! message polynomials. The list-decoding machinery — parameter search,
//! interpolation, and root extraction — comes from [`gs_engine`]; finite-field
//! arithmetic comes from [`fgf`].
//!
//! The decode-bearing families are twisted GRS ([`TgrsCode`], ambient GS at
//! pseudo-dimension `k'` with a twist-coefficient filter), punctured GRS
//! ([`PuncturedGrsCode`], GS on the surviving subdomain), Möbius-transformed GRS
//! ([`MobiusGrsCode`], GS on the moved points or the renormalized domain), and
//! extended GRS ([`ExtendedGrsCode`], puncture-then-GS-then-re-encode), of which
//! Roth–Lempel ([`RothLempelCode`]) is the classical single-functional instance
//! (Zhu–Jin). The folded ([`FoldedRsCode`]) and interleaved
//! ([`InterleavedRsCode`]) constructions are encoders and metrics only; their
//! capacity and collaborative decoders are future upstream capabilities.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

mod code;
mod decode;
#[cfg(feature = "internals")]
mod descriptor;
mod error;
mod eval;
mod extend;
mod folded;
mod interleaved;
mod mobius;
mod outcome;
mod puncture;
mod roth_lempel;
mod twist;

pub use code::TgrsCode;
pub use decode::{TgrsDecoder, TgrsScratch};
#[cfg(feature = "internals")]
pub use descriptor::{
    BaseCode, BaseGeometry, CanonicalDescriptor, DESCRIPTOR_VERSION, DecoderCapability,
    ExtendCoord, TransformOp, TransformWord,
};
pub use error::Error;
pub use extend::{ExtendedGrsCode, ExtendedGrsDecoder, ExtendedGrsScratch};
pub use folded::FoldedRsCode;
pub use interleaved::InterleavedRsCode;
pub use mobius::{MobiusGrsCode, MobiusGrsDecoder, MobiusGrsScratch, MobiusMap};
pub use outcome::UniqueDecode;
pub use puncture::{PuncturedGrsCode, PuncturedGrsDecoder, PuncturedGrsScratch};
pub use roth_lempel::{RothLempelCode, RothLempelDecoder, RothLempelScratch};
pub use twist::Twist;

pub use gs_engine::{AlekhnovichLimits, EvaluationDomain, ParameterLimits, Polynomial};
