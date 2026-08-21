//! Steady-state allocation behaviour of the decode path.
//!
//! Zero-allocation steady-state decoding is a crate invariant: once a decoder's
//! scratch and the caller's output vector are warm, a decode reuses every
//! buffer. This measures the result directly with a counting global allocator
//! rather than trusting the claim, and it does so on a deliberately
//! *multi-candidate* received word so the retained-overwrite path is exercised
//! for more than one output slot. The parameters are a low-rate `[16, 2]` code
//! over `GF(2^4)`, whose Guruswami–Sudan list radius exceeds the unique-decoding
//! radius, so a word can sit inside the ball of two codewords at once.
//!
//! The allocator counter is process-wide. Each `tests/*.rs` file is its own
//! test binary (its own process and global allocator), and this file holds a
//! single `#[test]`, so there is no cross-test contamination to guard against.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use contort::{
    AlekhnovichLimits, EvaluationDomain, ExtendedGrsCode, ExtendedGrsScratch, FoldedRsCode,
    InterleavedRsCode, MobiusGrsCode, MobiusGrsScratch, MobiusMap, ParameterLimits, Polynomial,
    PuncturedGrsCode, PuncturedGrsScratch, RothLempelCode, RothLempelScratch, TgrsCode,
    TgrsScratch, Twist, UniqueDecode,
};
#[cfg(feature = "internals")]
use contort::{BaseCode, ExtendCoord, TransformOp, TransformWord};
use fgf::Gf16;
use fgf::gf16::Elem;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn e(value: u8) -> Elem {
    Elem(u16::from(value))
}

fn parameter_limits() -> ParameterLimits {
    ParameterLimits::new(8, 16, usize::MAX, usize::MAX)
}

fn root_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128)
}

fn ramp_multipliers(n: usize) -> Vec<Elem> {
    (0..n).map(|i| e((i + 1) as u8)).collect()
}

/// A received word midway between two codewords: on coordinates where they
/// disagree, keep the first codeword on the lower half of the differing
/// positions and take the second on the upper half, so the word sits at
/// distance `⌊d/2⌋` and `⌈d/2⌉` from them. With `d <= 2·tau` this places the
/// word inside both balls and forces at least two candidates.
fn midpoint(a: &[Elem], b: &[Elem]) -> Vec<Elem> {
    let differing: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    let split = differing.len() / 2;
    let mut received = a.to_vec();
    for &i in &differing[split..] {
        received[i] = b[i];
    }
    received
}

/// Measure allocations while running `body` once.
fn measured<T>(body: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let value = body();
    COUNTING.store(false, Ordering::Relaxed);
    (value, ALLOCS.load(Ordering::Relaxed) - before)
}

const N: usize = 16;
const TAU: usize = 8;

#[test]
fn warm_decode_does_not_allocate() {
    tgrs_steady_state();
    roth_lempel_steady_state();
    folded_encode_is_allocation_free();
    interleaved_encode_is_allocation_free();
    puncture_steady_state();
    extend_steady_state();
    mobius_steady_state();
    #[cfg(feature = "internals")]
    descriptor_encode_is_allocation_free();
}

fn folded_encode_is_allocation_free() {
    let code = FoldedRsCode::<Gf16>::new(e(2), ramp_multipliers(6), 2, 2).unwrap();
    let message = [e(5), e(7)];
    let mut codeword = vec![Elem::ZERO; 6];
    code.encode_into(&message, &mut codeword).unwrap();
    let ((), allocs) = measured(|| code.encode_into(&message, &mut codeword).unwrap());
    assert_eq!(allocs, 0, "warm folded encode allocated {allocs} times");
}

fn interleaved_encode_is_allocation_free() {
    let points: Vec<Elem> = (0..5u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let code = InterleavedRsCode::new(domain, ramp_multipliers(5), 2, 2).unwrap();
    let messages = [e(5), e(7), e(7), e(5)];
    let mut codeword = vec![Elem::ZERO; 2 * 5];
    code.encode_into(&messages, &mut codeword).unwrap();
    let ((), allocs) = measured(|| code.encode_into(&messages, &mut codeword).unwrap());
    assert_eq!(
        allocs, 0,
        "warm interleaved encode allocated {allocs} times"
    );
}

fn tgrs_steady_state() {
    let points: Vec<Elem> = (0..N as u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let code = TgrsCode::new(
        domain,
        ramp_multipliers(N),
        2,
        vec![Twist::<Gf16>::new(1, 0, e(3))],
    )
    .unwrap();

    let decoder = code
        .list_decoder(TAU, parameter_limits(), root_limits())
        .unwrap();

    let mut c0 = vec![Elem::ZERO; N];
    let mut c1 = vec![Elem::ZERO; N];
    code.encode_into(&[e(1), e(0)], &mut c0).unwrap();
    code.encode_into(&[e(0), e(1)], &mut c1).unwrap();
    let received = midpoint(&c0, &c1);

    let mut scratch = TgrsScratch::<Gf16>::new();
    decoder.prepare_scratch(&mut scratch).unwrap();
    let mut output: Vec<Polynomial<Gf16>> = Vec::new();

    let candidates = decoder
        .list_decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    assert!(
        candidates >= 2,
        "expected a multi-candidate list, got {candidates}"
    );
    let _ = decoder.unique_decode(&received, &mut scratch).unwrap();

    let (count, allocs) = measured(|| {
        decoder
            .list_decode_into(&received, &mut scratch, &mut output)
            .unwrap()
    });
    assert_eq!(count, candidates);
    assert_eq!(
        allocs, 0,
        "warm TGRS list_decode_into allocated {allocs} times"
    );

    let (outcome, allocs) = measured(|| decoder.unique_decode(&received, &mut scratch).unwrap());
    assert!(matches!(outcome, UniqueDecode::Ambiguous));
    assert_eq!(
        allocs, 0,
        "warm TGRS unique_decode allocated {allocs} times"
    );
}

fn roth_lempel_steady_state() {
    let points: Vec<Elem> = (1..N as u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let code = RothLempelCode::new(domain, ramp_multipliers(N), 2, e(3)).unwrap();

    let decoder = code
        .list_decoder(TAU, parameter_limits(), root_limits())
        .unwrap();

    let mut c0 = vec![Elem::ZERO; N];
    let mut c1 = vec![Elem::ZERO; N];
    code.encode_into(&[e(1), e(0)], &mut c0).unwrap();
    code.encode_into(&[e(0), e(1)], &mut c1).unwrap();
    let received = midpoint(&c0, &c1);

    let mut scratch = RothLempelScratch::<Gf16>::new();
    decoder.prepare_scratch(&mut scratch).unwrap();
    let mut output: Vec<Polynomial<Gf16>> = Vec::new();

    let candidates = decoder
        .list_decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    assert!(
        candidates >= 2,
        "expected a multi-candidate list, got {candidates}"
    );
    let _ = decoder.unique_decode(&received, &mut scratch).unwrap();

    let (count, allocs) = measured(|| {
        decoder
            .list_decode_into(&received, &mut scratch, &mut output)
            .unwrap()
    });
    assert_eq!(count, candidates);
    assert_eq!(
        allocs, 0,
        "warm Roth–Lempel list_decode_into allocated {allocs} times"
    );

    let (outcome, allocs) = measured(|| decoder.unique_decode(&received, &mut scratch).unwrap());
    assert!(matches!(outcome, UniqueDecode::Ambiguous));
    assert_eq!(
        allocs, 0,
        "warm Roth–Lempel unique_decode allocated {allocs} times"
    );
}

fn puncture_steady_state() {
    let points: Vec<Elem> = (0..16u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let code = PuncturedGrsCode::new(domain, ramp_multipliers(N), 2, vec![8]).unwrap();
    let effective = code.length();
    let decoder = code
        .list_decoder(TAU, parameter_limits(), root_limits())
        .unwrap();

    let mut c0 = vec![Elem::ZERO; effective];
    let mut c1 = vec![Elem::ZERO; effective];
    code.encode_into(&[e(1), e(0)], &mut c0).unwrap();
    code.encode_into(&[e(0), e(1)], &mut c1).unwrap();
    let received = midpoint(&c0, &c1);

    let mut codeword = vec![Elem::ZERO; effective];
    code.encode_into(&[e(4), e(9)], &mut codeword).unwrap();
    let ((), allocs) = measured(|| code.encode_into(&[e(4), e(9)], &mut codeword).unwrap());
    assert_eq!(allocs, 0, "warm punctured encode allocated {allocs} times");

    let mut scratch = PuncturedGrsScratch::<Gf16>::new();
    decoder.prepare_scratch(&mut scratch).unwrap();
    let mut output: Vec<Polynomial<Gf16>> = Vec::new();
    let candidates = decoder
        .list_decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    assert!(
        candidates >= 2,
        "expected a multi-candidate list, got {candidates}"
    );
    let _ = decoder.unique_decode(&received, &mut scratch).unwrap();

    let (count, allocs) = measured(|| {
        decoder
            .list_decode_into(&received, &mut scratch, &mut output)
            .unwrap()
    });
    assert_eq!(count, candidates);
    assert_eq!(
        allocs, 0,
        "warm punctured list_decode_into allocated {allocs} times"
    );
}

fn extend_steady_state() {
    let points: Vec<Elem> = (0..(N - 1) as u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let code = ExtendedGrsCode::projective(domain, ramp_multipliers(N), 2).unwrap();
    let n = code.length();
    let decoder = code
        .list_decoder(TAU, parameter_limits(), root_limits())
        .unwrap();

    let mut c0 = vec![Elem::ZERO; n];
    let mut c1 = vec![Elem::ZERO; n];
    code.encode_into(&[e(1), e(0)], &mut c0).unwrap();
    code.encode_into(&[e(0), e(1)], &mut c1).unwrap();
    let received = midpoint(&c0, &c1);

    let mut codeword = vec![Elem::ZERO; n];
    code.encode_into(&[e(11), e(29)], &mut codeword).unwrap();
    let ((), allocs) = measured(|| code.encode_into(&[e(11), e(29)], &mut codeword).unwrap());
    assert_eq!(allocs, 0, "warm extended encode allocated {allocs} times");

    let mut scratch = ExtendedGrsScratch::<Gf16>::new();
    decoder.prepare_scratch(&mut scratch).unwrap();
    let mut output: Vec<Polynomial<Gf16>> = Vec::new();
    let candidates = decoder
        .list_decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    assert!(
        candidates >= 2,
        "expected a multi-candidate list, got {candidates}"
    );
    let _ = decoder.unique_decode(&received, &mut scratch).unwrap();

    let (count, allocs) = measured(|| {
        decoder
            .list_decode_into(&received, &mut scratch, &mut output)
            .unwrap()
    });
    assert_eq!(count, candidates);
    assert_eq!(
        allocs, 0,
        "warm extended list_decode_into allocated {allocs} times"
    );
}

fn mobius_steady_state() {
    let points: Vec<Elem> = (0..16u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    // Affine map (c = 0) — no pole, no domain exclusion.
    let map = MobiusMap::new(e(2), e(1), e(0), e(1));
    let code = MobiusGrsCode::new(domain, ramp_multipliers(N), 2, map).unwrap();
    let n = code.length();
    let decoder = code
        .list_decoder(TAU, parameter_limits(), root_limits())
        .unwrap();

    let mut c0 = vec![Elem::ZERO; n];
    let mut c1 = vec![Elem::ZERO; n];
    code.encode_into(&[e(1), e(0)], &mut c0).unwrap();
    code.encode_into(&[e(0), e(1)], &mut c1).unwrap();
    let received = midpoint(&c0, &c1);

    let mut codeword = vec![Elem::ZERO; n];
    code.encode_into(&[e(5), e(9)], &mut codeword).unwrap();
    let ((), allocs) = measured(|| code.encode_into(&[e(5), e(9)], &mut codeword).unwrap());
    assert_eq!(allocs, 0, "warm Möbius encode allocated {allocs} times");

    let mut scratch = MobiusGrsScratch::<Gf16>::new();
    decoder.prepare_scratch(&mut scratch).unwrap();
    let mut output: Vec<Polynomial<Gf16>> = Vec::new();
    let candidates = decoder
        .list_decode_into(&received, &mut scratch, &mut output)
        .unwrap();
    assert!(
        candidates >= 2,
        "expected a multi-candidate list, got {candidates}"
    );
    let _ = decoder.unique_decode(&received, &mut scratch).unwrap();

    let (count, allocs) = measured(|| {
        decoder
            .list_decode_into(&received, &mut scratch, &mut output)
            .unwrap()
    });
    assert_eq!(count, candidates);
    assert_eq!(
        allocs, 0,
        "warm Möbius list_decode_into allocated {allocs} times"
    );
}

#[cfg(feature = "internals")]
fn descriptor_encode_is_allocation_free() {
    let points: Vec<Elem> = (0..N as u8).map(e).collect();
    let domain = EvaluationDomain::<Gf16>::arbitrary(points).unwrap();
    let base = BaseCode::with_id(91, domain, ramp_multipliers(N), 2).unwrap();
    let mut word = TransformWord::new(base);
    word.push(TransformOp::Twist(Twist::new(1, 0, e(3))))
        .push(TransformOp::Mobius(MobiusMap::new(
            e(2),
            e(1),
            Elem::ZERO,
            Elem::ONE,
        )))
        .push(TransformOp::Puncture(7))
        .push(TransformOp::Extend(ExtendCoord::new(
            vec![Elem::ONE, e(5)],
            e(9),
        )));
    let descriptor = word.normalize().unwrap();
    let message = [e(11), e(29)];
    let mut codeword = vec![Elem::ZERO; descriptor.length()];

    descriptor.encode_into(&message, &mut codeword).unwrap();
    let ((), allocs) = measured(|| descriptor.encode_into(&message, &mut codeword).unwrap());
    assert_eq!(
        allocs, 0,
        "normalized descriptor encode allocated {allocs} times"
    );
}
