#![cfg(feature = "internals")]

//! Canonical transform descriptor fixtures: L1–L4 normalization, transmission
//! compression, encoder/decoder preservation, capability downgrades, and frozen
//! wire bytes.

use std::collections::HashSet;

use contort::{
    AlekhnovichLimits, BaseCode, BaseGeometry, DecoderCapability, Error, EvaluationDomain,
    ExtendCoord, ExtendedGrsCode, MobiusGrsCode, MobiusMap, ParameterLimits, Polynomial,
    PuncturedGrsCode, TgrsCode, TransformOp, TransformWord, Twist,
};
use fgf::field::Elem as _;
use fgf::gf8::Elem;
use fgf::gf16::Elem as Elem16;
use fgf::{Gf8, Gf16};
use gs_engine::ConfigError;

fn e(value: u8) -> Elem {
    Elem(value)
}

fn e16(value: u16) -> Elem16 {
    Elem16(value)
}

fn domain(n: usize) -> EvaluationDomain<Gf8> {
    EvaluationDomain::arbitrary((1..=n as u8).map(e).collect()).unwrap()
}

fn multipliers(n: usize) -> Vec<Elem> {
    (1..=n as u8).map(e).collect()
}

fn base(id: u64) -> BaseCode<Gf8> {
    BaseCode::with_id(id, domain(8), multipliers(8), 2).unwrap()
}

fn parameter_limits() -> ParameterLimits {
    ParameterLimits::new(8, 16, usize::MAX, usize::MAX)
}

fn root_limits() -> AlekhnovichLimits {
    AlekhnovichLimits::new(1_000_000, 100_000, usize::MAX, usize::MAX, 128)
}

fn decoded(candidates: &[Polynomial<Gf8>], k: usize) -> Vec<Vec<Elem>> {
    let mut values: Vec<Vec<Elem>> = candidates
        .iter()
        .map(|candidate| (0..k).map(|degree| candidate.coefficient(degree)).collect())
        .collect();
    values.sort();
    values
}

fn assert_compresses(word: &TransformWord<Gf8>, strict: bool) {
    let descriptor = word.normalize().unwrap();
    assert!(
        descriptor.serialized_len() <= word.serialized_len(),
        "descriptor {} exceeded word {}",
        descriptor.serialized_len(),
        word.serialized_len()
    );
    if strict {
        assert!(descriptor.serialized_len() < word.serialized_len());
    } else {
        assert_eq!(descriptor.serialized_len(), word.serialized_len());
    }
}

#[test]
fn frozen_corpus_compresses_by_l1_to_l4() {
    // Adversarial normal form: one already-canonical record on each non-Möbius
    // axis. No law can absorb anything, so serialization is equal-sized.
    let mut normal = TransformWord::new(base(11));
    normal
        .push(TransformOp::Twist(Twist::new(1, 0, e(2))))
        .push(TransformOp::Puncture(6))
        .push(TransformOp::Extend(ExtendCoord::new(
            vec![e(1), e(3)],
            e(7),
        )))
        .push(TransformOp::Fold(2))
        .push(TransformOp::Interleave(3));
    assert_compresses(&normal, false);

    // L1: a deep Möbius run becomes one scale-normalized PGL element.
    let mut mobius = TransformWord::new(base(12));
    mobius
        .push(TransformOp::Mobius(MobiusMap::new(e(2), e(1), e(0), e(1))))
        .push(TransformOp::Mobius(MobiusMap::new(e(1), e(0), e(1), e(9))))
        .push(TransformOp::Mobius(MobiusMap::new(e(3), e(4), e(0), e(1))));
    assert_compresses(&mobius, true);

    // L3: collided twist pairs annihilate in characteristic two.
    let mut twists = TransformWord::new(base(13));
    twists
        .push(TransformOp::Twist(Twist::new(1, 0, e(5))))
        .push(TransformOp::Twist(Twist::new(1, 0, e(5))));
    assert_compresses(&twists, true);

    // L2: duplicate punctures union; puncturing an extension annihilates it.
    let mut punctures = TransformWord::new(base(14));
    punctures
        .push(TransformOp::Puncture(4))
        .push(TransformOp::Puncture(4))
        .push(TransformOp::Extend(ExtendCoord::new(
            vec![e(1), e(0)],
            e(3),
        )))
        .push(TransformOp::Puncture(8));
    assert_compresses(&punctures, true);

    // L4: grouping runs become two numbers, independent of run depth.
    let mut grouping = TransformWord::new(base(15));
    grouping
        .push(TransformOp::Fold(2))
        .push(TransformOp::Fold(2))
        .push(TransformOp::Interleave(2))
        .push(TransformOp::Interleave(3));
    assert_compresses(&grouping, true);
    let descriptor = grouping.normalize().unwrap();
    assert_eq!(descriptor.fold(), 4);
    assert_eq!(descriptor.interleave(), 6);
}

#[test]
fn descriptor_encoder_matches_each_concrete_family() {
    let message = [e(11), e(29)];

    let twist = Twist::new(2, 1, e(3));
    let mut word = TransformWord::new(base(21));
    word.push(TransformOp::Twist(twist));
    let descriptor = word.normalize().unwrap();
    let concrete = TgrsCode::new(domain(8), multipliers(8), 2, vec![twist]).unwrap();
    let mut expected = vec![Elem::ZERO; concrete.length()];
    let mut actual = vec![Elem::ZERO; descriptor.length()];
    concrete.encode_into(&message, &mut expected).unwrap();
    descriptor.encode_into(&message, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let map = MobiusMap::new(e(2), e(1), e(0), e(1));
    let mut word = TransformWord::new(base(22));
    word.push(TransformOp::Mobius(map));
    let descriptor = word.normalize().unwrap();
    let concrete = MobiusGrsCode::new(domain(8), multipliers(8), 2, map).unwrap();
    concrete.encode_into(&message, &mut expected).unwrap();
    descriptor.encode_into(&message, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let mut word = TransformWord::new(base(23));
    word.push(TransformOp::Puncture(2))
        .push(TransformOp::Puncture(5));
    let descriptor = word.normalize().unwrap();
    let concrete = PuncturedGrsCode::new(domain(8), multipliers(8), 2, vec![2, 5]).unwrap();
    expected.resize(concrete.length(), Elem::ZERO);
    actual.resize(descriptor.length(), Elem::ZERO);
    concrete.encode_into(&message, &mut expected).unwrap();
    descriptor.encode_into(&message, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let functional = vec![e(1), e(7)];
    let mut word = TransformWord::new(base(24));
    word.push(TransformOp::Extend(ExtendCoord::new(
        functional.clone(),
        e(9),
    )));
    let descriptor = word.normalize().unwrap();
    let mut extended_multipliers = multipliers(8);
    extended_multipliers.push(e(9));
    let concrete =
        ExtendedGrsCode::new(domain(8), extended_multipliers, 2, vec![functional]).unwrap();
    expected.resize(concrete.length(), Elem::ZERO);
    actual.resize(descriptor.length(), Elem::ZERO);
    concrete.encode_into(&message, &mut expected).unwrap();
    descriptor.encode_into(&message, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn word_and_descriptor_encoders_match_composed_reference() {
    let map = MobiusMap::new(e(2), e(1), e(0), e(1));
    let functional = vec![e(1), e(4)];
    let twist = Twist::new(1, 0, e(3));
    let mut word = TransformWord::new(base(30));
    word.push(TransformOp::Twist(twist))
        .push(TransformOp::Mobius(map))
        .push(TransformOp::Puncture(2))
        .push(TransformOp::Extend(ExtendCoord::new(
            functional.clone(),
            e(9),
        )));
    let descriptor = word.normalize().unwrap();
    let message = [e(5), e(7)];
    let mut from_word = vec![Elem::ZERO; descriptor.length()];
    let mut from_descriptor = vec![Elem::ZERO; descriptor.length()];
    word.encode_into(&message, &mut from_word).unwrap();
    descriptor
        .encode_into(&message, &mut from_descriptor)
        .unwrap();

    // Independent direct interpreter: construct twisted coefficients, move
    // each point in word order, puncture coordinate 2, then append λ(message).
    let coefficients = [message[0], message[1], e(3).mul(message[0])];
    let mut reference = Vec::new();
    for (index, (&point, &multiplier)) in domain(8)
        .points()
        .iter()
        .zip(multipliers(8).iter())
        .enumerate()
    {
        if index == 2 {
            continue;
        }
        let x = map.apply(point).unwrap();
        let value = coefficients[2]
            .mul(x)
            .add(coefficients[1])
            .mul(x)
            .add(coefficients[0]);
        reference.push(multiplier.mul(value));
    }
    reference.push(
        e(9).mul(
            functional[0]
                .mul(message[0])
                .add(functional[1].mul(message[1])),
        ),
    );

    assert_eq!(from_word, reference);
    assert_eq!(from_descriptor, reference);
}

#[test]
fn word_and_descriptor_decoders_return_identical_lists() {
    let first = MobiusMap::new(e(2), e(1), e(0), e(1));
    let second = MobiusMap::new(e(1), e(0), e(1), e(200));
    let twists = [Twist::new(1, 0, e(3)), Twist::new(2, 1, e(7))];
    let mut word = TransformWord::new(base(40));
    word.push(TransformOp::Twist(twists[0]))
        .push(TransformOp::Mobius(first))
        .push(TransformOp::Twist(twists[1]))
        .push(TransformOp::Mobius(second));
    let descriptor = word.normalize().unwrap();
    let from_descriptor = descriptor.ambient_tgrs_code().unwrap().unwrap();

    // Independent word build: compose in application order and retain the
    // original twist set.
    let net = second.compose(&first).normalized();
    let moved: Vec<Elem> = domain(8)
        .points()
        .iter()
        .map(|&point| net.apply(point).unwrap())
        .collect();
    let from_word = TgrsCode::new(
        EvaluationDomain::arbitrary(moved).unwrap(),
        multipliers(8),
        2,
        twists.to_vec(),
    )
    .unwrap();

    let message = [e(17), e(91)];
    let mut received = vec![Elem::ZERO; 8];
    from_word.encode_into(&message, &mut received).unwrap();
    received[3] = received[3].add(e(1));

    let word_decoder = from_word
        .list_decoder(1, parameter_limits(), root_limits())
        .unwrap();
    let descriptor_decoder = from_descriptor
        .list_decoder(1, parameter_limits(), root_limits())
        .unwrap();
    let mut word_scratch = contort::TgrsScratch::new();
    let mut descriptor_scratch = contort::TgrsScratch::new();
    let mut word_out = Vec::new();
    let mut descriptor_out = Vec::new();
    word_decoder
        .list_decode_into(&received, &mut word_scratch, &mut word_out)
        .unwrap();
    descriptor_decoder
        .list_decode_into(&received, &mut descriptor_scratch, &mut descriptor_out)
        .unwrap();
    assert_eq!(decoded(&word_out, 2), decoded(&descriptor_out, 2));
}

#[test]
fn mobius_group_law_scale_and_three_transitivity() {
    let first = MobiusMap::<Gf8>::new(e(2), e(1), e(0), e(1));
    let second = MobiusMap::<Gf8>::new(e(1), e(0), e(1), e(200));
    let net = second.compose(&first);
    for point in (1..=8).map(e) {
        assert_eq!(net.apply(point), second.apply(first.apply(point).unwrap()));
    }

    let scale = e(9);
    let scaled = MobiusMap::<Gf8>::new(
        first.a().mul(scale),
        first.b().mul(scale),
        first.c().mul(scale),
        first.d().mul(scale),
    );
    assert_eq!(first.normalized(), scaled.normalized());

    let representative = MobiusMap::<Gf8>::from_three_points(e(1), e(2), e(3));
    assert_eq!(representative.apply(e(1)), Some(Elem::ZERO));
    assert_eq!(representative.apply(e(2)), Some(Elem::ONE));
    assert_eq!(representative.apply(e(3)), None);
    assert!(!representative.determinant().is_zero());
}

#[test]
fn twist_merge_annihilation_and_dimension_invariant() {
    let points: Vec<Elem16> = (0..8).map(e16).collect();
    let base = BaseCode::with_id(
        50,
        EvaluationDomain::<Gf16>::arbitrary(points).unwrap(),
        (1..=8).map(e16).collect(),
        2,
    )
    .unwrap();
    let mut word = TransformWord::new(base);
    word.push(TransformOp::Twist(Twist::new(1, 0, e16(2))))
        .push(TransformOp::Twist(Twist::new(1, 1, e16(3))))
        .push(TransformOp::Twist(Twist::new(1, 0, e16(2))));
    let descriptor = word.normalize().unwrap();
    assert_eq!(descriptor.twists(), &[Twist::new(1, 1, e16(3))]);
    assert_eq!(descriptor.pseudo_dimension(), 3);
    assert_eq!(descriptor.dimension(), 2);

    // Exhaustively prove the merged image still has q^k distinct codewords.
    let mut image = HashSet::new();
    let mut codeword = vec![Elem16::ZERO; descriptor.length()];
    for a in 0..16 {
        for b in 0..16 {
            descriptor
                .encode_into(&[e16(a), e16(b)], &mut codeword)
                .unwrap();
            image.insert(codeword.clone());
        }
    }
    assert_eq!(image.len(), 16 * 16);

    // Ambient/decreasing hooks are outside the supported graph language, so
    // they are rejected before any cycle can form.
    let mut invalid = TransformWord::new(descriptor.base().clone());
    invalid.push(TransformOp::Twist(Twist::new(1, 2, e16(1))));
    assert_eq!(
        invalid.normalize().unwrap_err(),
        Error::TwistHook {
            hook: 2,
            dimension: 2
        }
    );
}

#[test]
fn capability_flags_survive_normalization() {
    let orbit = || {
        BaseCode::<Gf16>::multiplicative_orbit(60, e16(2), (1..=8).map(e16).collect(), 2).unwrap()
    };

    let mut folded = TransformWord::new(orbit());
    folded.push(TransformOp::Fold(2));
    assert_eq!(
        folded.normalize().unwrap().capability(),
        DecoderCapability::FoldedList {
            pseudo_dimension: 2,
            fold: 2
        }
    );

    let mut scaled = TransformWord::new(orbit());
    scaled
        .push(TransformOp::Mobius(MobiusMap::new(
            e16(2),
            Elem16::ZERO,
            Elem16::ZERO,
            Elem16::ONE,
        )))
        .push(TransformOp::Fold(2));
    assert!(matches!(
        scaled.normalize().unwrap().capability(),
        DecoderCapability::FoldedList { .. }
    ));

    let mut general = TransformWord::new(orbit());
    general
        .push(TransformOp::Twist(Twist::new(2, 0, e16(3))))
        .push(TransformOp::Mobius(MobiusMap::new(
            Elem16::ONE,
            Elem16::ONE,
            Elem16::ZERO,
            Elem16::ONE,
        )))
        .push(TransformOp::Fold(2));
    assert_eq!(
        general.normalize().unwrap().capability(),
        DecoderCapability::AmbientGs {
            pseudo_dimension: 4
        }
    );

    let mut mid_block = TransformWord::new(orbit());
    mid_block
        .push(TransformOp::Puncture(1))
        .push(TransformOp::Fold(2));
    assert_eq!(
        mid_block.normalize().unwrap().capability(),
        DecoderCapability::AmbientGs {
            pseudo_dimension: 2
        }
    );

    let mut whole_block = TransformWord::new(orbit());
    whole_block
        .push(TransformOp::Puncture(0))
        .push(TransformOp::Puncture(1))
        .push(TransformOp::Fold(2));
    assert!(matches!(
        whole_block.normalize().unwrap().capability(),
        DecoderCapability::FoldedList { .. }
    ));

    let mut collaborative = TransformWord::new(orbit());
    collaborative.push(TransformOp::Interleave(3));
    assert_eq!(
        collaborative.normalize().unwrap().capability(),
        DecoderCapability::Collaborative {
            pseudo_dimension: 2,
            order: 3
        }
    );

    let arbitrary = base(61);
    assert_eq!(arbitrary.geometry(), BaseGeometry::Arbitrary);
    let mut not_orbit = TransformWord::new(arbitrary);
    not_orbit.push(TransformOp::Fold(2));
    assert!(matches!(
        not_orbit.normalize().unwrap().capability(),
        DecoderCapability::AmbientGs { .. }
    ));
}

#[test]
fn normalization_checks_grouping_overflow() {
    let mut word = TransformWord::new(base(70));
    word.push(TransformOp::Fold(usize::MAX))
        .push(TransformOp::Fold(2));
    assert_eq!(
        word.normalize().unwrap_err(),
        Error::Configuration(ConfigError::GeometryOverflow {
            context: "net fold"
        })
    );
}

#[test]
fn descriptor_bytes_are_versioned_and_frozen() {
    let mut word = TransformWord::new(base(0x0102_0304_0506_0708));
    word.push(TransformOp::Twist(Twist::new(1, 0, e(3))))
        .push(TransformOp::Mobius(MobiusMap::new(e(1), e(2), e(0), e(1))))
        .push(TransformOp::Puncture(5))
        .push(TransformOp::Extend(ExtendCoord::new(
            vec![e(1), e(0)],
            e(7),
        )))
        .push(TransformOp::Fold(2))
        .push(TransformOp::Interleave(3));
    let descriptor = word.normalize().unwrap();

    assert_eq!(
        descriptor.to_bytes(),
        vec![
            1, // version
            8, 7, 6, 5, 4, 3, 2, 1, // base ID (little endian)
            2, // k
            1, 1, 0, 3, // twist (tag, t, h, eta)
            2, 2, 0, 1, // Möbius (tag+pivot, b, c, d)
            3, 5, // puncture
            4, 2, 1, 0, 7, // extend
            5, 2, // fold
            6, 3, // interleave
        ]
    );
}
