//! Steady-state list- and unique-decode throughput for the twisted GRS and
//! Roth–Lempel families.
//!
//! Every case warms its decoder scratch and the caller output vector before
//! timing, so the measured loop is the allocation-free steady state proven by
//! `tests/zero_alloc.rs`: normalization, the ambient Guruswami–Sudan decode,
//! and the family's filter / exceptional check. The `BenchmarkId` carries the
//! field and the code shape. Numbers land in `BENCHMARKS.md`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use poly_ring::{AlekhnovichLimits, Polynomial};

use contort::{RothLempelCode, RothLempelScratch, TgrsCode, TgrsScratch, Twist};
use gs_engine::{EvaluationDomain, ParameterLimits};

fn parameter_limits() -> ParameterLimits {
    ParameterLimits::new(8, 16, usize::MAX, usize::MAX)
}

fn root_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128)
}

fn bench_gf8(criterion: &mut Criterion) {
    use fgf::Gf8B;
    use fgf::gf8b::Elem;
    let e = |v: u8| Elem::from_raw(v);
    let n = 8usize;
    let k = 2usize;
    let tau = 3usize;
    let multipliers: Vec<Elem> = (0..n).map(|i| e((i + 1) as u8)).collect();

    // TGRS
    let tgrs = TgrsCode::<Gf8B>::new(
        EvaluationDomain::arbitrary((0..n as u8).map(e).collect()).unwrap(),
        multipliers.clone(),
        k,
        vec![Twist::new(1, 0, e(3))],
    )
    .unwrap();
    let mut codeword = vec![Elem::ZERO; n];
    tgrs.encode_into(&[e(5), e(7)], &mut codeword).unwrap();
    codeword[0] = Elem::from_raw(codeword[0].to_raw() ^ 1);
    codeword[3] = Elem::from_raw(codeword[3].to_raw() ^ 1);
    let tgrs_decoder = tgrs
        .list_decoder(tau, parameter_limits(), root_limits())
        .unwrap();
    let mut tgrs_scratch = TgrsScratch::<Gf8B>::new();
    tgrs_decoder.prepare_scratch(&mut tgrs_scratch).unwrap();
    let mut tgrs_output: Vec<Polynomial<Gf8B>> = Vec::new();
    tgrs_decoder
        .list_decode_into(&codeword, &mut tgrs_scratch, &mut tgrs_output)
        .unwrap();

    // Roth–Lempel
    let rl = RothLempelCode::<Gf8B>::new(
        EvaluationDomain::arbitrary((1..n as u8).map(e).collect()).unwrap(),
        multipliers,
        k,
        e(3),
    )
    .unwrap();
    let mut rl_codeword = vec![Elem::ZERO; n];
    rl.encode_into(&[e(5), e(7)], &mut rl_codeword).unwrap();
    rl_codeword[0] = Elem::from_raw(rl_codeword[0].to_raw() ^ 1);
    rl_codeword[3] = Elem::from_raw(rl_codeword[3].to_raw() ^ 1);
    let rl_decoder = rl
        .list_decoder(tau, parameter_limits(), root_limits())
        .unwrap();
    let mut rl_scratch = RothLempelScratch::<Gf8B>::new();
    rl_decoder.prepare_scratch(&mut rl_scratch).unwrap();
    let mut rl_output: Vec<Polynomial<Gf8B>> = Vec::new();
    rl_decoder
        .list_decode_into(&rl_codeword, &mut rl_scratch, &mut rl_output)
        .unwrap();

    let mut group = criterion.benchmark_group("list-decode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| {
            tgrs_decoder
                .list_decode_into(black_box(&codeword), &mut tgrs_scratch, &mut tgrs_output)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| {
            rl_decoder
                .list_decode_into(black_box(&rl_codeword), &mut rl_scratch, &mut rl_output)
                .unwrap()
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("unique-decode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| {
            tgrs_decoder
                .unique_decode(black_box(&codeword), &mut tgrs_scratch)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| {
            rl_decoder
                .unique_decode(black_box(&rl_codeword), &mut rl_scratch)
                .unwrap()
        });
    });
    group.finish();
}

fn bench_gf16(criterion: &mut Criterion) {
    use fgf::Gf16;
    use fgf::gf16::Elem;
    let e = |v: u8| Elem::from_raw(u16::from(v));
    let n = 16usize;
    let k = 2usize;
    let tau = 6usize;
    let multipliers: Vec<Elem> = (0..n).map(|i| e((i + 1) as u8)).collect();

    let tgrs = TgrsCode::<Gf16>::new(
        EvaluationDomain::arbitrary((0..n as u8).map(e).collect()).unwrap(),
        multipliers.clone(),
        k,
        vec![Twist::new(1, 0, e(3))],
    )
    .unwrap();
    let mut codeword = vec![Elem::ZERO; n];
    tgrs.encode_into(&[e(5), e(7)], &mut codeword).unwrap();
    for slot in codeword[..4].iter_mut() {
        *slot = Elem::from_raw(slot.to_raw() ^ 1);
    }
    let tgrs_decoder = tgrs
        .list_decoder(tau, parameter_limits(), root_limits())
        .unwrap();
    let mut tgrs_scratch = TgrsScratch::<Gf16>::new();
    tgrs_decoder.prepare_scratch(&mut tgrs_scratch).unwrap();
    let mut tgrs_output: Vec<Polynomial<Gf16>> = Vec::new();
    tgrs_decoder
        .list_decode_into(&codeword, &mut tgrs_scratch, &mut tgrs_output)
        .unwrap();

    let rl = RothLempelCode::<Gf16>::new(
        EvaluationDomain::arbitrary((1..n as u8).map(e).collect()).unwrap(),
        multipliers,
        k,
        e(3),
    )
    .unwrap();
    let mut rl_codeword = vec![Elem::ZERO; n];
    rl.encode_into(&[e(5), e(7)], &mut rl_codeword).unwrap();
    for slot in rl_codeword[..4].iter_mut() {
        *slot = Elem::from_raw(slot.to_raw() ^ 1);
    }
    let rl_decoder = rl
        .list_decoder(tau, parameter_limits(), root_limits())
        .unwrap();
    let mut rl_scratch = RothLempelScratch::<Gf16>::new();
    rl_decoder.prepare_scratch(&mut rl_scratch).unwrap();
    let mut rl_output: Vec<Polynomial<Gf16>> = Vec::new();
    rl_decoder
        .list_decode_into(&rl_codeword, &mut rl_scratch, &mut rl_output)
        .unwrap();

    let mut group = criterion.benchmark_group("list-decode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| {
            tgrs_decoder
                .list_decode_into(black_box(&codeword), &mut tgrs_scratch, &mut tgrs_output)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| {
            rl_decoder
                .list_decode_into(black_box(&rl_codeword), &mut rl_scratch, &mut rl_output)
                .unwrap()
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("unique-decode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| {
            tgrs_decoder
                .unique_decode(black_box(&codeword), &mut tgrs_scratch)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| {
            rl_decoder
                .unique_decode(black_box(&rl_codeword), &mut rl_scratch)
                .unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_gf8, bench_gf16);
criterion_main!(benches);
