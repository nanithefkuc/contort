//! Steady-state encode throughput for the twisted GRS and Roth–Lempel families.
//!
//! Each case reuses its codeword buffer, matching the allocation-free steady
//! state that `tests/zero_alloc.rs` proves for decoding. The `BenchmarkId`
//! carries the field so a scalar `GF(2^8)` number is never confused with a
//! `GF(2^4)` one. Numbers land in `BENCHMARKS.md`, never in this file.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use contort::{EvaluationDomain, RothLempelCode, TgrsCode, Twist};

fn bench_gf8(criterion: &mut Criterion) {
    use fgf::Gf8;
    use fgf::gf8::Elem;
    let e = |v: u8| Elem(v);
    let n = 8usize;
    let k = 2usize;

    let points: Vec<Elem> = (0..n as u8).map(e).collect();
    let multipliers: Vec<Elem> = (0..n).map(|i| e((i + 1) as u8)).collect();

    let tgrs = TgrsCode::<Gf8>::new(
        EvaluationDomain::arbitrary(points.clone()).unwrap(),
        multipliers.clone(),
        k,
        vec![Twist::new(1, 0, e(3))],
    )
    .unwrap();
    let rl_points: Vec<Elem> = (1..n as u8).map(e).collect();
    let rl = RothLempelCode::<Gf8>::new(
        EvaluationDomain::arbitrary(rl_points).unwrap(),
        multipliers,
        k,
        e(3),
    )
    .unwrap();

    let message = [e(5), e(7)];
    let mut codeword = vec![Elem::ZERO; n];

    let mut group = criterion.benchmark_group("encode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| {
            tgrs.encode_into(black_box(&message), &mut codeword)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^8)"), &(), |b, ()| {
        b.iter(|| rl.encode_into(black_box(&message), &mut codeword).unwrap());
    });
    group.finish();
}

fn bench_gf16(criterion: &mut Criterion) {
    use fgf::Gf16;
    use fgf::gf16::Elem;
    let e = |v: u8| Elem(u16::from(v));
    let n = 16usize;
    let k = 2usize;

    let points: Vec<Elem> = (0..n as u8).map(e).collect();
    let multipliers: Vec<Elem> = (0..n).map(|i| e((i + 1) as u8)).collect();

    let tgrs = TgrsCode::<Gf16>::new(
        EvaluationDomain::arbitrary(points.clone()).unwrap(),
        multipliers.clone(),
        k,
        vec![Twist::new(1, 0, e(3))],
    )
    .unwrap();
    let rl_points: Vec<Elem> = (1..n as u8).map(e).collect();
    let rl = RothLempelCode::<Gf16>::new(
        EvaluationDomain::arbitrary(rl_points).unwrap(),
        multipliers,
        k,
        e(3),
    )
    .unwrap();

    let message = [e(5), e(7)];
    let mut codeword = vec![Elem::ZERO; n];

    let mut group = criterion.benchmark_group("encode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::new("tgrs", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| {
            tgrs.encode_into(black_box(&message), &mut codeword)
                .unwrap()
        });
    });
    group.bench_with_input(BenchmarkId::new("roth_lempel", "GF(2^4)"), &(), |b, ()| {
        b.iter(|| rl.encode_into(black_box(&message), &mut codeword).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_gf8, bench_gf16);
criterion_main!(benches);
