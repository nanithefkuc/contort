//! Möbius-transformed generalized Reed–Solomon codes.
//!
//! A Möbius map `φ(x) = (a·x + b)/(c·x + d)` with `Δ = a·d − b·c ≠ 0` relabels
//! the evaluation points of a GRS code projectively. The transformed code
//!
//! `C_φ = { (w_i · f(φ(α_i)))_i : deg f < k }`
//!
//! is again GRS — on the moved points `β_i = φ(α_i)` with the unchanged
//! multipliers `w` — so it decodes through the ambient Guruswami–Sudan engine
//! with no new machinery. Two equivalent decode routes fall out of the closure
//! identity `g(X) = (c·X + d)^{k−1} f(φ(X))` (`deg g < k`):
//!
//! 1. **moved points** — build the plan directly on `β` and decode; candidates
//!    are the message polynomials `f`.
//! 2. **normalized multipliers** — decode on the original domain `α` with
//!    multipliers `w_i·(c·α_i + d)^{−(k−1)}`, recovering `g`, then pull `g` back
//!    to `f` through the inverse map.
//!
//! Both routes return the same list; [`MobiusGrsDecoder`] exposes each and the
//! differential is an integration invariant.
//!
//! v1 rejects a pole on the domain ([`Error::MobiusPole`]); the projective
//! `f(∞)` coordinate it would create is an [`crate::ExtendedGrsCode`] extension,
//! not a Möbius edit.

use alloc::vec;
use alloc::vec::Vec;

use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use gs_engine::{DecodeScratch, EvaluationDomain, GsParameters, GsPlan, ParameterLimits};
use poly_ring::{AlekhnovichLimits, Polynomial};

use crate::error::Error;
use crate::eval::horner;
use crate::outcome::UniqueDecode;

/// A projective-linear map `φ(x) = (a·x + b)/(c·x + d)` on `P¹(F)`.
///
/// Stored as its `2×2` coefficient matrix `[[a, b], [c, d]]`; composition is
/// matrix multiplication and the group law of `PGL(2, F)`.
pub struct MobiusMap<F: ButterflyKernels> {
    a: F::Elem,
    b: F::Elem,
    c: F::Elem,
    d: F::Elem,
}

impl<F: ButterflyKernels> MobiusMap<F> {
    /// Construct a map from its four coefficients `(a, b, c, d)`.
    #[must_use]
    pub const fn new(a: F::Elem, b: F::Elem, c: F::Elem, d: F::Elem) -> Self {
        Self { a, b, c, d }
    }

    /// The coefficient `a`.
    #[must_use]
    pub fn a(&self) -> F::Elem {
        self.a
    }

    /// The coefficient `b`.
    #[must_use]
    pub fn b(&self) -> F::Elem {
        self.b
    }

    /// The coefficient `c`.
    #[must_use]
    pub fn c(&self) -> F::Elem {
        self.c
    }

    /// The coefficient `d`.
    #[must_use]
    pub fn d(&self) -> F::Elem {
        self.d
    }

    /// The determinant `Δ = a·d − b·c` (in characteristic two, `a·d + b·c`).
    ///
    /// The map is a bijection of `P¹(F)` exactly when `Δ ≠ 0`.
    #[must_use]
    pub fn determinant(&self) -> F::Elem {
        self.a.mul(self.d).sub(self.b.mul(self.c))
    }

    /// Evaluate `φ(x)`, returning `None` at the pole `x = −d/c`.
    #[must_use]
    pub fn apply(&self, x: F::Elem) -> Option<F::Elem> {
        let denominator = self.c.mul(x).add(self.d);
        if denominator.is_zero() {
            None
        } else {
            Some(self.a.mul(x).add(self.b).div(denominator))
        }
    }

    /// The composition `self ∘ inner`, i.e. `x ↦ self(inner(x))`.
    ///
    /// Equal to the matrix product `self · inner`; this is the `PGL(2, F)`
    /// group operation used to net a run of Möbius transforms into one.
    #[must_use]
    pub fn compose(&self, inner: &Self) -> Self {
        Self {
            a: self.a.mul(inner.a).add(self.b.mul(inner.c)),
            b: self.a.mul(inner.b).add(self.b.mul(inner.d)),
            c: self.c.mul(inner.a).add(self.d.mul(inner.c)),
            d: self.c.mul(inner.b).add(self.d.mul(inner.d)),
        }
    }

    /// The adjugate `[[d, −b], [−c, a]]`, representing `φ⁻¹` up to the scalar
    /// `Δ`.
    #[must_use]
    pub fn adjugate(&self) -> Self {
        // Characteristic two: −b = b and −c = c, so the adjugate of
        // [[a, b], [c, d]] is [[d, b], [c, a]].
        Self {
            a: self.d,
            b: self.b,
            c: self.c,
            d: self.a,
        }
    }

    /// Whether `φ` preserves a multiplicative orbit's geometric-progression
    /// structure: the scalings `x ↦ a·x` (`b = c = 0`) and the inversions
    /// `x ↦ b/(c·x)` (`a = d = 0`), the normalizer of the domain's
    /// multiplicative subgroup in `PGL(2, F)`.
    ///
    /// A map failing this test downgrades a composed fold's decode capability
    /// from folded-list to ambient GS (see the composability notes).
    #[must_use]
    pub fn is_orbit_preserving(&self) -> bool {
        (self.b.is_zero() && self.c.is_zero()) || (self.a.is_zero() && self.d.is_zero())
    }

    /// The unique map sending three distinct points `(p0, p1, p2)` to
    /// `(0, 1, ∞)`.
    ///
    /// `PGL(2, F)` is sharply 3-transitive on `P¹(F)`, so this is the canonical
    /// representative used to drop a net Möbius transform's slot cost to zero
    /// over a plain (untwisted) message space. The three points must be
    /// distinct or the resulting map is singular.
    #[must_use]
    pub fn from_three_points(p0: F::Elem, p1: F::Elem, p2: F::Elem) -> Self {
        // φ(x) = ((x − p0)(p1 − p2)) / ((x − p2)(p1 − p0)).
        let a = p1.sub(p2);
        let c = p1.sub(p0);
        Self {
            a,
            b: p0.mul(a),
            c,
            d: p2.mul(c),
        }
    }

    /// The identity map `x ↦ x` (matrix `[[1, 0], [0, 1]]`).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: F::Elem::ONE,
            b: F::Elem::ZERO,
            c: F::Elem::ZERO,
            d: F::Elem::ONE,
        }
    }

    /// The canonical projective representative: every coefficient divided by
    /// the first nonzero of `(a, b, c, d)`, so that leading entry becomes `1`.
    ///
    /// Scaling the matrix does not change the map, so this picks one member of
    /// each `PGL(2, F)` class — the scale-normalized form the descriptor stores.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let pivot = [self.a, self.b, self.c, self.d]
            .into_iter()
            .find(|value| !value.is_zero())
            .unwrap_or(F::Elem::ONE);
        let inverse = pivot.inv();
        Self {
            a: self.a.mul(inverse),
            b: self.b.mul(inverse),
            c: self.c.mul(inverse),
            d: self.d.mul(inverse),
        }
    }

    /// Whether `φ` is the identity of `PGL(2, F)` (a nonzero scalar multiple of
    /// the identity matrix).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.normalized() == Self::identity()
    }
}

impl<F: ButterflyKernels> Clone for MobiusMap<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: ButterflyKernels> Copy for MobiusMap<F> {}

impl<F: ButterflyKernels> core::fmt::Debug for MobiusMap<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MobiusMap")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("c", &self.c)
            .field("d", &self.d)
            .finish()
    }
}

impl<F: ButterflyKernels> PartialEq for MobiusMap<F> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c && self.d == other.d
    }
}

impl<F: ButterflyKernels> Eq for MobiusMap<F> {}

/// A Möbius-transformed generalized Reed–Solomon code.
///
/// Built from a base domain `α`, nonzero multipliers `w`, dimension `k`, and a
/// nonsingular map `φ`. The moved points `β_i = φ(α_i)` are precomputed at
/// construction; encoding evaluates the message polynomial there.
#[derive(Debug)]
pub struct MobiusGrsCode<F: ButterflyKernels> {
    domain: EvaluationDomain<F>,
    multipliers: Vec<F::Elem>,
    dimension: usize,
    map: MobiusMap<F>,
    moved: Vec<F::Elem>,
}

impl<F: ButterflyKernels> MobiusGrsCode<F> {
    /// Build a Möbius-transformed GRS code.
    ///
    /// # Errors
    ///
    /// - [`Error::MultiplierCount`] / [`Error::ZeroMultiplier`] for a bad
    ///   multiplier vector;
    /// - [`Error::InvalidDimension`] unless `1 ≤ k < n`;
    /// - [`Error::MobiusDelta`] if `Δ = 0`;
    /// - [`Error::MobiusPole`] if the pole `−d/c` lands on a domain point.
    pub fn new(
        domain: EvaluationDomain<F>,
        multipliers: Vec<F::Elem>,
        dimension: usize,
        map: MobiusMap<F>,
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
        if map.determinant().is_zero() {
            return Err(Error::MobiusDelta);
        }
        let mut moved = Vec::with_capacity(length);
        for (index, &point) in domain.points().iter().enumerate() {
            match map.apply(point) {
                Some(image) => moved.push(image),
                None => return Err(Error::MobiusPole { index }),
            }
        }
        Ok(Self {
            domain,
            multipliers,
            dimension,
            map,
            moved,
        })
    }

    /// Code length `n`.
    #[must_use]
    pub fn length(&self) -> usize {
        self.domain.len()
    }

    /// Message dimension `k`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// The base evaluation domain `α`.
    #[must_use]
    pub const fn domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }

    /// The column multipliers `w`.
    #[must_use]
    pub fn multipliers(&self) -> &[F::Elem] {
        &self.multipliers
    }

    /// The Möbius map `φ`.
    #[must_use]
    pub fn map(&self) -> &MobiusMap<F> {
        &self.map
    }

    /// The moved evaluation points `β_i = φ(α_i)`.
    #[must_use]
    pub fn moved_points(&self) -> &[F::Elem] {
        &self.moved
    }

    /// Whether `φ` preserves multiplicative-orbit structure (fold capability).
    #[must_use]
    pub fn is_orbit_preserving(&self) -> bool {
        self.map.is_orbit_preserving()
    }

    /// Encode a `k`-symbol message into an `n`-symbol codeword.
    ///
    /// Allocation-free: the message is evaluated at each moved point by Horner
    /// and scaled by the column multiplier.
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
            .zip(self.moved.iter())
            .zip(self.multipliers.iter())
        {
            *slot = multiplier.mul(horner::<F>(message, *point));
        }
        Ok(())
    }

    /// Build a list decoder for a chosen decoding radius.
    pub fn list_decoder(
        &self,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<MobiusGrsDecoder<F>, Error> {
        MobiusGrsDecoder::new(self, target_radius, parameter_limits, root_limits)
    }

    /// Build a decoder at the MDS unique-decoding radius `⌊(n−k)/2⌋`.
    pub fn unique_decoder(
        &self,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<MobiusGrsDecoder<F>, Error> {
        let radius = (self.length() - self.dimension) / 2;
        self.list_decoder(radius, parameter_limits, root_limits)
    }
}

/// Reusable working storage for repeated Möbius decodes.
pub struct MobiusGrsScratch<F: ButterflyKernels> {
    decode: DecodeScratch<F>,
    normalized: Vec<F::Elem>,
    ambient: Vec<Polynomial<F>>,
    filtered: Vec<Polynomial<F>>,
    message: Vec<F::Elem>,
}

impl<F: ButterflyKernels> MobiusGrsScratch<F> {
    /// Construct empty scratch storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decode: DecodeScratch::new(),
            normalized: Vec::new(),
            ambient: Vec::new(),
            filtered: Vec::new(),
            message: Vec::new(),
        }
    }
}

impl<F: ButterflyKernels> Default for MobiusGrsScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated Möbius decoder bound to one decoding radius.
///
/// Carries both decode plans: [`list_decode_moved_into`] runs the plan on the
/// moved points `β`, [`list_decode_normalized_into`] runs the plan on the base
/// domain `α` with the multipliers renormalized and pulls each recovered `g`
/// back to `f`. [`list_decode_into`] uses the moved-point route.
///
/// [`list_decode_moved_into`]: Self::list_decode_moved_into
/// [`list_decode_normalized_into`]: Self::list_decode_normalized_into
/// [`list_decode_into`]: Self::list_decode_into
pub struct MobiusGrsDecoder<F: ButterflyKernels> {
    moved_plan: GsPlan<F>,
    base_plan: GsPlan<F>,
    /// Inverse column multipliers `w_i⁻¹` for the moved-point route.
    inverse_multipliers: Vec<F::Elem>,
    /// Per-coordinate factor `(c·α_i + d)^{k−1}·w_i⁻¹` mapping a received
    /// symbol to `g(α_i)` for the normalized route.
    base_normalizers: Vec<F::Elem>,
    /// Fixed pullback basis: row `i` is the length-`k` coefficient vector of
    /// `N^i · D^{k−1−i}` with `N = d·Y + b`, `D = c·Y + a`.
    pullback_basis: Vec<Vec<F::Elem>>,
    /// The scalar `Δ^{−(k−1)}` applied after the pullback basis product.
    pullback_scalar: F::Elem,
    dimension: usize,
    length: usize,
    target_radius: usize,
}

impl<F: ButterflyKernels> MobiusGrsDecoder<F> {
    pub(crate) fn new(
        code: &MobiusGrsCode<F>,
        target_radius: usize,
        parameter_limits: ParameterLimits,
        root_limits: AlekhnovichLimits,
    ) -> Result<Self, Error> {
        let length = code.length();
        let dimension = code.dimension;
        let parameters =
            GsParameters::search::<F>(length, dimension - 1, target_radius, parameter_limits)?;

        let moved_domain = EvaluationDomain::arbitrary(code.moved.clone())?;
        let moved_plan = GsPlan::new(parameters, moved_domain, root_limits)?;
        let base_plan = GsPlan::new(parameters, code.domain.clone(), root_limits)?;

        let inverse_multipliers: Vec<F::Elem> = code.multipliers.iter().map(|m| m.inv()).collect();

        let map = code.map;
        let exponent = (dimension - 1) as u64;
        let base_normalizers: Vec<F::Elem> = code
            .domain
            .points()
            .iter()
            .zip(code.multipliers.iter())
            .map(|(&point, multiplier)| {
                let denominator = map.c().mul(point).add(map.d());
                denominator.pow(exponent).mul(multiplier.inv())
            })
            .collect();

        let pullback_basis = pullback_basis::<F>(&map, dimension);
        let pullback_scalar = map.determinant().pow(exponent).inv();

        Ok(Self {
            moved_plan,
            base_plan,
            inverse_multipliers,
            base_normalizers,
            pullback_basis,
            pullback_scalar,
            dimension,
            length,
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

    /// Reserve every reusable buffer for this decoder's maximum geometry.
    pub fn prepare_scratch(&self, scratch: &mut MobiusGrsScratch<F>) -> Result<(), Error> {
        self.moved_plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.ambient)?;
        self.moved_plan
            .prepare_scratch(&mut scratch.decode, &mut scratch.filtered)?;
        scratch.normalized.reserve(self.length);
        scratch.message.reserve(self.dimension);
        Ok(())
    }

    /// List decode into caller-owned output using the moved-point route.
    ///
    /// This is the default route: the transformed code is `GRS(β, w)`, so the
    /// ambient candidates already are the message polynomials `f`.
    pub fn list_decode_into(
        &self,
        received: &[F::Elem],
        scratch: &mut MobiusGrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        self.list_decode_moved_into(received, scratch, output)
    }

    /// List decode via the moved-point plan on `β`.
    pub fn list_decode_moved_into(
        &self,
        received: &[F::Elem],
        scratch: &mut MobiusGrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
                got: received.len(),
            });
        }
        scratch.normalized.clear();
        scratch.normalized.reserve(self.length);
        for (symbol, inverse) in received.iter().zip(self.inverse_multipliers.iter()) {
            scratch.normalized.push(symbol.mul(*inverse));
        }
        self.moved_plan.decode_into(
            &scratch.normalized,
            &mut scratch.decode,
            &mut scratch.ambient,
        )?;
        let ambient = core::mem::take(&mut scratch.ambient);
        let mut count = 0;
        for candidate in &ambient {
            write_low_degree::<F>(
                &mut scratch.message,
                output,
                count,
                candidate,
                self.dimension,
            )?;
            count += 1;
        }
        scratch.ambient = ambient;
        output.truncate(count);
        Ok(count)
    }

    /// List decode via the normalized-multiplier plan on `α`, pulling each
    /// recovered `g` back to the message `f`.
    pub fn list_decode_normalized_into(
        &self,
        received: &[F::Elem],
        scratch: &mut MobiusGrsScratch<F>,
        output: &mut Vec<Polynomial<F>>,
    ) -> Result<usize, Error> {
        if received.len() != self.length {
            return Err(Error::ReceivedLength {
                expected: self.length,
                got: received.len(),
            });
        }
        scratch.normalized.clear();
        scratch.normalized.reserve(self.length);
        for (symbol, normalizer) in received.iter().zip(self.base_normalizers.iter()) {
            scratch.normalized.push(symbol.mul(*normalizer));
        }
        self.base_plan.decode_into(
            &scratch.normalized,
            &mut scratch.decode,
            &mut scratch.ambient,
        )?;
        let ambient = core::mem::take(&mut scratch.ambient);
        let mut count = 0;
        for candidate in &ambient {
            self.pull_back(&mut scratch.message, output, count, candidate)?;
            count += 1;
        }
        scratch.ambient = ambient;
        output.truncate(count);
        Ok(count)
    }

    /// Uniquely decode via the moved-point route.
    pub fn unique_decode(
        &self,
        received: &[F::Elem],
        scratch: &mut MobiusGrsScratch<F>,
    ) -> Result<UniqueDecode<F>, Error> {
        let mut filtered = core::mem::take(&mut scratch.filtered);
        let count = self.list_decode_into(received, scratch, &mut filtered)?;
        let outcome = match count {
            0 => UniqueDecode::NoCandidate,
            1 => UniqueDecode::Message(filtered[0].clone()),
            _ => UniqueDecode::Ambiguous,
        };
        scratch.filtered = filtered;
        Ok(outcome)
    }

    /// Convert a recovered `g` (degree `< k`) into the message `f` via
    /// `f = Δ^{−(k−1)}·Σ_i g_i·(N^i·D^{k−1−i})`, writing into `output[index]`.
    fn pull_back(
        &self,
        message: &mut Vec<F::Elem>,
        output: &mut Vec<Polynomial<F>>,
        index: usize,
        candidate: &Polynomial<F>,
    ) -> Result<(), Error> {
        message.clear();
        for degree in 0..self.dimension {
            let mut accumulator = F::Elem::ZERO;
            for source in 0..self.dimension {
                accumulator = accumulator.add(
                    candidate
                        .coefficient(source)
                        .mul(self.pullback_basis[source][degree]),
                );
            }
            message.push(self.pullback_scalar.mul(accumulator));
        }
        write_message_coefficients::<F>(output, index, message)
    }
}

/// Precompute the pullback basis rows `N^i · D^{k−1−i}` for `i` in `0..k`,
/// with `N(Y) = d·Y + b` and `D(Y) = c·Y + a` (the adjugate of `φ`). Each row
/// is a length-`k` coefficient vector.
fn pullback_basis<F: ButterflyKernels>(map: &MobiusMap<F>, dimension: usize) -> Vec<Vec<F::Elem>> {
    let adjugate = map.adjugate();
    // N = adjugate numerator = a'·Y + b' = d·Y + b ; D = c'·Y + d' = c·Y + a.
    let numerator = [adjugate.b(), adjugate.a()];
    let denominator = [adjugate.d(), adjugate.c()];

    let mut numerator_powers = vec![vec![F::Elem::ONE]; 1];
    let mut denominator_powers = vec![vec![F::Elem::ONE]; 1];
    for _ in 1..dimension {
        let next_numerator = mul_linear::<F>(numerator_powers.last().unwrap(), numerator);
        numerator_powers.push(next_numerator);
        let next_denominator = mul_linear::<F>(denominator_powers.last().unwrap(), denominator);
        denominator_powers.push(next_denominator);
    }

    let mut basis = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let mut row = convolve::<F>(&numerator_powers[i], &denominator_powers[dimension - 1 - i]);
        row.resize(dimension, F::Elem::ZERO);
        basis.push(row);
    }
    basis
}

/// Multiply a polynomial by the linear factor `linear[0] + linear[1]·X`.
fn mul_linear<F: ButterflyKernels>(poly: &[F::Elem], linear: [F::Elem; 2]) -> Vec<F::Elem> {
    let mut result = vec![F::Elem::ZERO; poly.len() + 1];
    for (degree, &coefficient) in poly.iter().enumerate() {
        result[degree] = result[degree].add(coefficient.mul(linear[0]));
        result[degree + 1] = result[degree + 1].add(coefficient.mul(linear[1]));
    }
    result
}

/// Convolve two coefficient vectors (polynomial product).
fn convolve<F: ButterflyKernels>(a: &[F::Elem], b: &[F::Elem]) -> Vec<F::Elem> {
    let mut result = vec![F::Elem::ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            result[i + j] = result[i + j].add(x.mul(y));
        }
    }
    result
}

/// Write the degree-`< dimension` prefix of `candidate` into `output[index]`,
/// reusing retained storage where possible (mirrors `TgrsDecoder`).
fn write_low_degree<F: ButterflyKernels>(
    message: &mut Vec<F::Elem>,
    output: &mut Vec<Polynomial<F>>,
    index: usize,
    candidate: &Polynomial<F>,
    dimension: usize,
) -> Result<(), Error> {
    if index < output.len() {
        let polynomial = &mut output[index];
        for degree in 0..dimension {
            polynomial.set_coefficient(degree, candidate.coefficient(degree))?;
        }
        polynomial.truncate(dimension);
    } else {
        message.clear();
        for degree in 0..dimension {
            message.push(candidate.coefficient(degree));
        }
        output.push(Polynomial::from_coefficients(message)?);
    }
    Ok(())
}

/// Write the coefficient slice `message` into `output[index]`, reusing storage.
fn write_message_coefficients<F: ButterflyKernels>(
    output: &mut Vec<Polynomial<F>>,
    index: usize,
    message: &[F::Elem],
) -> Result<(), Error> {
    if index < output.len() {
        let polynomial = &mut output[index];
        for (degree, &value) in message.iter().enumerate() {
            polynomial.set_coefficient(degree, value)?;
        }
        polynomial.truncate(message.len());
    } else {
        output.push(Polynomial::from_coefficients(message)?);
    }
    Ok(())
}
