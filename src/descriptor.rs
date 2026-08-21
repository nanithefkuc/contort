//! Canonical transform descriptor for the six deformation generators.
//!
//! A deformed code is specified as a base GRS code plus an ordered *word* over
//! six generators — twist, Möbius, puncture, extend, fold, interleave. The word
//! is never the object a peer needs to transmit: it reduces to a
//! [`CanonicalDescriptor`] of constant slot count, and each axis satisfies a
//! closure law that nets any run of same-axis operations to a representation no
//! larger than the run:
//!
//! - **L1 (domain position)** — Möbius maps compose in `PGL(2, F)`; `r` maps net
//!   to one scale-normalized element (four field elements become three).
//! - **L2 (domain membership)** — punctures union idempotently and a puncture on
//!   an extended coordinate annihilates that extension.
//! - **L3 (message space)** — twists merge by destination degree; equal `(t, h)`
//!   pairs sum their `η` and annihilate at zero (in characteristic two, two
//!   equal twists cancel).
//! - **L4 (grouping)** — folds multiply (`F_{m₂} ∘ F_{m₁} = F_{m₁m₂}`), as do
//!   interleaves.
//!
//! The three axes act on disjoint data and commute, so the reduction is a fold
//! of the word into the descriptor state. Serializing the descriptor is never
//! larger than serializing any generating word (per axis), which is the
//! transmission-compressibility property; the descriptor and word encode
//! identically, so the reduction is lossless.
//!
//! This module is unstable, gated behind the `internals` feature, and describes
//! code families — it is not itself a public code family.

use alloc::vec;
use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{ConfigError, EvaluationDomain};

use crate::code::TgrsCode;

use crate::error::Error;
use crate::eval::horner;
use crate::mobius::MobiusMap;
use crate::twist::Twist;

/// Serialization format version. Bumping it is a wire-format break.
pub const DESCRIPTOR_VERSION: u8 = 1;

const TAG_TWIST: u8 = 1;
const TAG_MOBIUS: u8 = 2;
const TAG_PUNCTURE: u8 = 3;
const TAG_EXTEND: u8 = 4;
const TAG_FOLD: u8 = 5;
const TAG_INTERLEAVE: u8 = 6;

/// The decode capability a descriptor can honestly offer after normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderCapability {
    /// Ambient Guruswami–Sudan decoding at the given pseudo-dimension `k'`.
    AmbientGs {
        /// The pseudo-dimension `k'` the ambient decoder runs at.
        pseudo_dimension: usize,
    },
    /// Folded-list decoding: a genuine multiplicative orbit survives, the fold
    /// divides the length, no Möbius broke the orbit, and no puncture split a
    /// block.
    FoldedList {
        /// The pseudo-dimension `k'` the folded-list decoder runs at.
        pseudo_dimension: usize,
        /// The surviving fold `m`.
        fold: usize,
    },
    /// Collaborative (interleaved) decoding at the given order `ℓ`.
    Collaborative {
        /// The pseudo-dimension `k'` the collaborative decoder runs at.
        pseudo_dimension: usize,
        /// The interleaving order `ℓ`.
        order: usize,
    },
}

/// An appended extension coordinate: a linear functional on the `k` message
/// coefficients, scaled by a nonzero multiplier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtendCoord<F: ButterflyKernels> {
    functional: Vec<F::Elem>,
    multiplier: F::Elem,
}

impl<F: ButterflyKernels> ExtendCoord<F> {
    /// Construct an extension coordinate from its functional and multiplier.
    #[must_use]
    pub fn new(functional: Vec<F::Elem>, multiplier: F::Elem) -> Self {
        Self {
            functional,
            multiplier,
        }
    }

    /// The linear functional's coefficient vector (length `k`).
    #[must_use]
    pub fn functional(&self) -> &[F::Elem] {
        &self.functional
    }

    /// The coordinate's column multiplier.
    #[must_use]
    pub fn multiplier(&self) -> F::Elem {
        self.multiplier
    }
}

/// Geometry promised by the base-code identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseGeometry {
    /// Distinct points with no multiplicative-orbit promise.
    Arbitrary,
    /// Consecutive powers `1, γ, …, γ^{n-1}` of a validated generator.
    MultiplicativeOrbit,
}

/// The base GRS code a transform word deforms.
#[derive(Clone, Debug)]
pub struct BaseCode<F: ButterflyKernels> {
    id: u64,
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    geometry: BaseGeometry,
}

impl<F: ButterflyKernels> BaseCode<F> {
    /// Construct an arbitrary-domain base code with identifier zero.
    ///
    /// # Errors
    ///
    /// [`Error::MultiplierCount`], [`Error::ZeroMultiplier`], or
    /// [`Error::InvalidDimension`] for an inconsistent base.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
    ) -> Result<Self, Error> {
        Self::with_id(0, domain, multipliers, dimension)
    }

    /// Construct an arbitrary-domain base code with a stable peer-visible ID.
    ///
    /// The ID names the separately agreed domain and multiplier table; it is
    /// serialized in both words and descriptors.
    pub fn with_id(
        id: u64,
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
    ) -> Result<Self, Error> {
        let length = domain.len();
        validate_base::<F>(&multipliers, dimension, length)?;
        Ok(Self {
            id,
            domain,
            multipliers,
            dimension,
            geometry: BaseGeometry::Arbitrary,
        })
    }

    /// Construct a multiplicative-orbit base `1, γ, …, γ^{n-1}`.
    ///
    /// # Errors
    ///
    /// [`Error::InsufficientOrbit`] if the generator repeats before `n`;
    /// otherwise the same shape errors as [`BaseCode::with_id`].
    pub fn multiplicative_orbit(
        id: u64,
        generator: F::Elem,
        multipliers: Vec<F::Elem>,
        dimension: usize,
    ) -> Result<Self, Error> {
        let length = multipliers.len();
        if generator.is_zero() {
            return Err(Error::InsufficientOrbit { length });
        }
        let mut points = Vec::with_capacity(length);
        let mut power = F::Elem::ONE;
        for index in 0..length {
            if index > 0 && power == F::Elem::ONE {
                return Err(Error::InsufficientOrbit { length });
            }
            points.push(power);
            power = power.mul(generator);
        }
        let domain = EvaluationDomain::arbitrary(points)?;
        validate_base::<F>(&multipliers, dimension, length)?;
        Ok(Self {
            id,
            domain,
            multipliers,
            dimension,
            geometry: BaseGeometry::MultiplicativeOrbit,
        })
    }

    /// Stable base-code identifier serialized on the wire.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// The base domain's validated geometry.
    #[must_use]
    pub const fn geometry(&self) -> BaseGeometry {
        self.geometry
    }

    /// The base evaluation domain `α`.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// The base column multipliers `v`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// The base code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len()
    }

    /// The message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// One generator in a transform word.
#[derive(Clone, Debug)]
pub enum TransformOp<F: ButterflyKernels> {
    /// A message-space twist.
    Twist(Twist<F>),
    /// A domain-position Möbius relabelling.
    Mobius(MobiusMap<F>),
    /// A domain-membership puncture of one coordinate (base index `< n`, or an
    /// extended index `>= n` that annihilates the matching extension).
    Puncture(usize),
    /// A domain-membership extension by a linear functional and multiplier.
    Extend(ExtendCoord<F>),
    /// A grouping fold by `m`.
    Fold(usize),
    /// A grouping interleave by `ℓ`.
    Interleave(usize),
}

/// A base code plus an ordered list of transform generators.
#[derive(Clone, Debug)]
pub struct TransformWord<F: ButterflyKernels> {
    base: BaseCode<F>,
    ops: Vec<TransformOp<F>>,
}

impl<F: ButterflyKernels> TransformWord<F> {
    /// Start an empty word over a base code (the ordinary GRS code).
    #[must_use]
    pub fn new(base: BaseCode<F>) -> Self {
        Self {
            base,
            ops: Vec::new(),
        }
    }

    /// Append one generator.
    pub fn push(&mut self, op: TransformOp<F>) -> &mut Self {
        self.ops.push(op);
        self
    }

    /// The base code.
    #[must_use]
    pub const fn base(&self) -> &BaseCode<F> {
        &self.base
    }

    /// The generator sequence.
    #[must_use]
    pub fn ops(&self) -> &[TransformOp<F>] {
        &self.ops
    }

    /// Reduce the word to its canonical descriptor by the closure laws.
    ///
    /// Allocation happens only here, at build time; the resulting descriptor
    /// encodes without further pipeline bookkeeping.
    pub fn normalize(&self) -> Result<CanonicalDescriptor<F>, Error> {
        let k = self.base.dimension;
        let base_length = self.base.length();
        let max_offset = base_length - k;

        let mut twists: Vec<(usize, usize, F::Elem)> = Vec::new();
        let mut mobius = MobiusMap::<F>::identity();
        let mut punctures: Vec<usize> = Vec::new();
        let mut extends: Vec<ExtendCoord<F>> = Vec::new();
        let mut fold = 1usize;
        let mut interleave = 1usize;

        for op in &self.ops {
            match op {
                TransformOp::Twist(twist) => {
                    let offset = twist.offset();
                    let hook = twist.hook();
                    if offset < 1 || offset > max_offset {
                        return Err(Error::TwistOffset {
                            offset,
                            max: max_offset,
                        });
                    }
                    if hook >= k {
                        return Err(Error::TwistHook { hook, dimension: k });
                    }
                    if twist.coefficient().is_zero() {
                        return Err(Error::ZeroTwistCoefficient {
                            index: twists.len(),
                        });
                    }
                    merge_twist::<F>(&mut twists, offset, hook, twist.coefficient());
                }
                TransformOp::Mobius(map) => {
                    if map.determinant().is_zero() {
                        return Err(Error::MobiusDelta);
                    }
                    mobius = map.compose(&mobius);
                }
                TransformOp::Puncture(index) => {
                    if *index < base_length {
                        if !punctures.contains(index) {
                            punctures.push(*index);
                        }
                    } else {
                        // Extension indices are interpreted in the *current*
                        // scalar word. Removing one shifts later extensions,
                        // exactly as sequential puncturing does.
                        let slot = index - base_length;
                        if slot >= extends.len() {
                            let current_length = base_length
                                .checked_add(extends.len())
                                .ok_or_else(|| overflow("transform word length"))?;
                            return Err(Error::PunctureIndex {
                                index: *index,
                                length: current_length,
                            });
                        }
                        // L2: a puncture annihilates the extension at this point.
                        extends.remove(slot);
                    }
                }
                TransformOp::Extend(coord) => {
                    if coord.functional.len() != k {
                        return Err(Error::FunctionalLength {
                            expected: k,
                            got: coord.functional.len(),
                        });
                    }
                    if coord.multiplier.is_zero() {
                        let index = base_length
                            .checked_add(extends.len())
                            .ok_or_else(|| overflow("extension coordinate index"))?;
                        return Err(Error::ZeroMultiplier { index });
                    }
                    extends.push(coord.clone());
                }
                TransformOp::Fold(m) => {
                    if *m == 0 {
                        return Err(Error::FoldParameter {
                            fold: 0,
                            length: base_length,
                        });
                    }
                    fold = fold.checked_mul(*m).ok_or_else(|| overflow("net fold"))?;
                }
                TransformOp::Interleave(order) => {
                    if *order == 0 {
                        return Err(Error::ZeroInterleave);
                    }
                    interleave = interleave
                        .checked_mul(*order)
                        .ok_or_else(|| overflow("net interleave"))?;
                }
            }
        }

        twists.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        punctures.sort_unstable();
        let twist_ops: Vec<Twist<F>> = twists
            .iter()
            .map(|&(offset, hook, eta)| Twist::new(offset, hook, eta))
            .collect();
        let max_twist = twists
            .iter()
            .map(|&(offset, _, _)| offset)
            .max()
            .unwrap_or(0);
        let pseudo_dimension = k
            .checked_add(max_twist)
            .ok_or_else(|| overflow("descriptor pseudo-dimension"))?;

        let survivors = base_length - punctures.len();
        if survivors < k {
            return Err(Error::PunctureLength {
                remaining: survivors,
                dimension: k,
            });
        }
        let length = survivors
            .checked_add(extends.len())
            .ok_or_else(|| overflow("descriptor code length"))?;
        let mobius = mobius.normalized();
        let capability = derive_capability::<F>(
            pseudo_dimension,
            fold,
            interleave,
            &mobius,
            &punctures,
            base_length,
            &extends,
            length,
            self.base.geometry,
        );

        Ok(CanonicalDescriptor {
            base: self.base.clone(),
            twists: twist_ops,
            mobius,
            punctures,
            extends,
            fold,
            interleave,
            pseudo_dimension,
            capability,
        })
    }

    /// Encode a message by first reducing the word, then encoding from the
    /// descriptor.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        self.normalize()?.encode_into(message, codeword)
    }

    /// The serialized word bytes: version, base-code ID, dimension, then one
    /// tagged record per generator in word order.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = vec![DESCRIPTOR_VERSION];
        buffer.extend_from_slice(&self.base.id.to_le_bytes());
        push_usize(&mut buffer, self.base.dimension);
        for op in &self.ops {
            match op {
                TransformOp::Twist(twist) => {
                    buffer.push(TAG_TWIST);
                    push_usize(&mut buffer, twist.offset());
                    push_usize(&mut buffer, twist.hook());
                    push_elem::<F>(&mut buffer, twist.coefficient());
                }
                TransformOp::Mobius(map) => {
                    buffer.push(TAG_MOBIUS);
                    push_elem::<F>(&mut buffer, map.a());
                    push_elem::<F>(&mut buffer, map.b());
                    push_elem::<F>(&mut buffer, map.c());
                    push_elem::<F>(&mut buffer, map.d());
                }
                TransformOp::Puncture(index) => {
                    buffer.push(TAG_PUNCTURE);
                    push_usize(&mut buffer, *index);
                }
                TransformOp::Extend(coord) => {
                    buffer.push(TAG_EXTEND);
                    push_usize(&mut buffer, coord.functional.len());
                    for &value in &coord.functional {
                        push_elem::<F>(&mut buffer, value);
                    }
                    push_elem::<F>(&mut buffer, coord.multiplier);
                }
                TransformOp::Fold(m) => {
                    buffer.push(TAG_FOLD);
                    push_usize(&mut buffer, *m);
                }
                TransformOp::Interleave(order) => {
                    buffer.push(TAG_INTERLEAVE);
                    push_usize(&mut buffer, *order);
                }
            }
        }
        buffer
    }

    /// The serialized word length in bytes.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.to_bytes().len()
    }
}

/// The reduced, axis-sorted state that fully determines a deformed code.
#[derive(Clone, Debug)]
pub struct CanonicalDescriptor<F: ButterflyKernels> {
    base: BaseCode<F>,
    twists: Vec<Twist<F>>,
    mobius: MobiusMap<F>,
    punctures: Vec<usize>,
    extends: Vec<ExtendCoord<F>>,
    fold: usize,
    interleave: usize,
    pseudo_dimension: usize,
    capability: DecoderCapability,
}

impl<F: ButterflyKernels> CanonicalDescriptor<F> {
    /// The base code.
    #[must_use]
    pub const fn base(&self) -> &BaseCode<F> {
        &self.base
    }

    /// The merged, destination-sorted twist program.
    #[must_use]
    pub fn twists(&self) -> &[Twist<F>] {
        &self.twists
    }

    /// The net Möbius map (identity when the word carried none).
    #[must_use]
    pub fn mobius(&self) -> &MobiusMap<F> {
        &self.mobius
    }

    /// The net puncture set, in ascending base-coordinate order.
    #[must_use]
    pub fn punctures(&self) -> &[usize] {
        &self.punctures
    }

    /// The surviving extension coordinates.
    #[must_use]
    pub fn extends(&self) -> &[ExtendCoord<F>] {
        &self.extends
    }

    /// The net fold `m`.
    #[must_use]
    pub const fn fold(&self) -> usize {
        self.fold
    }

    /// The net interleave order `ℓ`.
    #[must_use]
    pub const fn interleave(&self) -> usize {
        self.interleave
    }

    /// The pseudo-dimension `k'` the ambient decoder runs at.
    #[must_use]
    pub const fn pseudo_dimension(&self) -> usize {
        self.pseudo_dimension
    }

    /// The honest decode capability after normalization.
    #[must_use]
    pub const fn capability(&self) -> DecoderCapability {
        self.capability
    }

    /// The flat codeword length: surviving base coordinates plus extensions.
    #[must_use]
    pub fn length(&self) -> usize {
        (self.base.length() - self.punctures.len()) + self.extends.len()
    }

    /// The message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.base.dimension
    }

    /// Rebuild the ambient TGRS code for descriptor states whose full reduction
    /// is exactly one GRS/TGRS plan: no puncture, extension, fold, or interleave.
    ///
    /// The net Möbius is absorbed into the moved domain and the merged twist
    /// program is retained. Returns `Ok(None)` when domain membership or
    /// grouping requires a different concrete adapter.
    pub fn ambient_tgrs_code(&self) -> Result<Option<TgrsCode<F>>, Error> {
        if !self.punctures.is_empty()
            || !self.extends.is_empty()
            || self.fold != 1
            || self.interleave != 1
        {
            return Ok(None);
        }
        let mut moved = Vec::with_capacity(self.base.length());
        for (index, &point) in self.base.domain.points().iter().enumerate() {
            moved.push(
                self.mobius
                    .apply(point)
                    .ok_or(Error::MobiusPole { index })?,
            );
        }
        let domain = EvaluationDomain::arbitrary(moved)?;
        Ok(Some(TgrsCode::new(
            domain,
            self.base.multipliers.clone(),
            self.base.dimension,
            self.twists.clone(),
        )?))
    }

    /// Encode a `k`-symbol message into the flat one-row codeword.
    ///
    /// The codeword lists the surviving base coordinates in ascending index
    /// order — each the twisted message polynomial evaluated at the
    /// Möbius-moved point and scaled by the base multiplier — followed by the
    /// extension coordinates. A fold only reshapes this flat row. Interleave
    /// records how multiple independently encoded rows are transposed; this
    /// single-row method leaves those row bytes unchanged.
    ///
    /// Allocation-free: twist terms are accumulated directly at each point,
    /// so no ambient coefficient vector is materialized.
    pub fn encode_into(&self, message: &[F::Elem], codeword: &mut [F::Elem]) -> Result<(), Error> {
        let k = self.base.dimension;
        if message.len() != k {
            return Err(Error::MessageLength {
                expected: k,
                got: message.len(),
            });
        }
        let length = self.length();
        if codeword.len() != length {
            return Err(Error::CodewordLength {
                expected: length,
                got: codeword.len(),
            });
        }

        let points = self.base.domain.points();
        let mut write = 0;
        for (index, (&point, &multiplier)) in
            points.iter().zip(self.base.multipliers.iter()).enumerate()
        {
            if self.punctures.binary_search(&index).is_ok() {
                continue;
            }
            let moved = self
                .mobius
                .apply(point)
                .ok_or(Error::MobiusPole { index })?;
            let mut value = horner::<F>(message, moved);
            for twist in &self.twists {
                let destination = k - 1 + twist.offset();
                let exponent = u64::try_from(destination)
                    .map_err(|_| overflow("twist evaluation exponent"))?;
                let contribution = twist
                    .coefficient()
                    .mul(message[twist.hook()])
                    .mul(moved.pow(exponent));
                value = value.add(contribution);
            }
            codeword[write] = multiplier.mul(value);
            write += 1;
        }

        for coord in &self.extends {
            let mut accumulator = F::Elem::ZERO;
            for (&lambda, &symbol) in coord.functional.iter().zip(message.iter()) {
                accumulator = accumulator.add(lambda.mul(symbol));
            }
            codeword[write] = coord.multiplier.mul(accumulator);
            write += 1;
        }
        Ok(())
    }

    /// The serialized descriptor bytes in canonical axis order.
    ///
    /// Records appear as version, base-code ID, dimension, merged twists, the
    /// scale-normalized net Möbius (present only when non-identity, storing its
    /// pivot position plus three coefficients), net punctures, extensions, and
    /// fold/interleave numbers (present only when greater than one). Changing
    /// this encoding is a format break.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = vec![DESCRIPTOR_VERSION];
        buffer.extend_from_slice(&self.base.id.to_le_bytes());
        push_usize(&mut buffer, self.base.dimension);

        for twist in &self.twists {
            buffer.push(TAG_TWIST);
            push_usize(&mut buffer, twist.offset());
            push_usize(&mut buffer, twist.hook());
            push_elem::<F>(&mut buffer, twist.coefficient());
        }

        if !self.mobius.is_identity() {
            let normalized = self.mobius.normalized();
            let coefficients = [
                normalized.a(),
                normalized.b(),
                normalized.c(),
                normalized.d(),
            ];
            let pivot = coefficients
                .iter()
                .position(|value| !value.is_zero())
                .unwrap_or(0);
            // The high nibble carries the pivot position, so scale normalization
            // really saves one field element even for one-byte fields.
            buffer.push(TAG_MOBIUS | ((pivot as u8) << 4));
            for (position, &value) in coefficients.iter().enumerate() {
                if position != pivot {
                    push_elem::<F>(&mut buffer, value);
                }
            }
        }

        for &index in &self.punctures {
            buffer.push(TAG_PUNCTURE);
            push_usize(&mut buffer, index);
        }

        for coord in &self.extends {
            buffer.push(TAG_EXTEND);
            push_usize(&mut buffer, coord.functional.len());
            for &value in &coord.functional {
                push_elem::<F>(&mut buffer, value);
            }
            push_elem::<F>(&mut buffer, coord.multiplier);
        }

        if self.fold > 1 {
            buffer.push(TAG_FOLD);
            push_usize(&mut buffer, self.fold);
        }
        if self.interleave > 1 {
            buffer.push(TAG_INTERLEAVE);
            push_usize(&mut buffer, self.interleave);
        }
        buffer
    }

    /// The serialized descriptor length in bytes.
    #[must_use]
    pub fn serialized_len(&self) -> usize {
        self.to_bytes().len()
    }
}

/// Merge a twist into the destination-grouped program (L3): equal `(offset,
/// hook)` pairs sum their coefficients and annihilate at zero.
fn merge_twist<F: ButterflyKernels>(
    twists: &mut Vec<(usize, usize, F::Elem)>,
    offset: usize,
    hook: usize,
    coefficient: F::Elem,
) {
    if let Some(position) = twists
        .iter()
        .position(|&(o, h, _)| o == offset && h == hook)
    {
        let combined = twists[position].2.add(coefficient);
        if combined.is_zero() {
            twists.remove(position);
        } else {
            twists[position].2 = combined;
        }
    } else {
        twists.push((offset, hook, coefficient));
    }
}

/// Derive the honest decode capability from the reduced state.
#[allow(clippy::too_many_arguments)]
fn derive_capability<F: ButterflyKernels>(
    pseudo_dimension: usize,
    fold: usize,
    interleave: usize,
    mobius: &MobiusMap<F>,
    punctures: &[usize],
    base_length: usize,
    extends: &[ExtendCoord<F>],
    length: usize,
    geometry: BaseGeometry,
) -> DecoderCapability {
    if interleave > 1 {
        return DecoderCapability::Collaborative {
            pseudo_dimension,
            order: interleave,
        };
    }
    if fold > 1 {
        let foldable = geometry == BaseGeometry::MultiplicativeOrbit
            && length.is_multiple_of(fold)
            && mobius.is_orbit_preserving()
            && extends.is_empty()
            && blocks_aligned(punctures, base_length, fold);
        if foldable {
            return DecoderCapability::FoldedList {
                pseudo_dimension,
                fold,
            };
        }
    }
    DecoderCapability::AmbientGs { pseudo_dimension }
}

/// Whether the puncture set deletes whole `fold`-blocks of the base domain, so
/// the surviving consecutive orbit stays foldable.
fn blocks_aligned(punctures: &[usize], base_length: usize, fold: usize) -> bool {
    if fold == 0 || !base_length.is_multiple_of(fold) {
        return false;
    }
    for block in 0..base_length / fold {
        let start = block * fold;
        let first = punctures.binary_search(&start).is_ok();
        for index in start + 1..start + fold {
            if punctures.binary_search(&index).is_ok() != first {
                return false;
            }
        }
    }
    true
}

fn push_usize(buffer: &mut Vec<u8>, mut value: usize) {
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buffer.push(low);
            break;
        }
        buffer.push(low | 0x80);
    }
}

fn validate_base<F: ButterflyKernels>(
    multipliers: &[F::Elem],
    dimension: usize,
    length: usize,
) -> Result<(), Error> {
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
    Ok(())
}

fn overflow(context: &'static str) -> Error {
    ConfigError::GeometryOverflow { context }.into()
}

fn push_elem<F: ButterflyKernels>(buffer: &mut Vec<u8>, value: F::Elem) {
    let start = buffer.len();
    buffer.resize(start + F::BYTES, 0);
    F::write(&mut buffer[start..], value);
}
