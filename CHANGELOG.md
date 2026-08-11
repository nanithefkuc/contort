# Changelog

All notable changes to contort are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Renamed the crate from `twister` to `contort` and scoped it to the deformed
  Reed–Solomon family (twisted / folded / interleaved GRS, Roth–Lempel).
- Retargeted list decoding onto `gs-engine`; removed the self-contained
  pre-`gs-engine` field and Guruswami–Sudan prototype.
- Added the twisted GRS family: `TgrsCode` encoding and Guruswami–Sudan list
  and unique decoding, filtering ambient GRS candidates by the twist
  constraints (Zhu–Jin). Depends on `gs-engine`, `fgf`, and `butterfly-fft`.
- Added the Roth–Lempel family: `RothLempelCode` encoding and Guruswami–Sudan
  list and unique decoding, puncturing the exceptional coordinate to a GRS code
  and re-encoding candidates for a full Hamming-distance check (Zhu–Jin).
- Promoted the per-family error to a shared `Error` and shared the
  `UniqueDecode` outcome across families.
