<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->

# Public Task Validation Files

`public_task.zip` contains the eleven unchanged public files of the supplied
RowColInt8Quant task: the task text, README, case and operator definitions,
coverage and performance targets, input generator, independent golden,
accuracy verifier, benchmark and performance checker. It contains no solution,
historical results, environment-specific binaries or private dependencies.

Archive SHA-256:
`c780ab2cc344a9220d59eabf6345e0bde68f6c6a0034a312dfdd9fe80d20e772`.

The original `case.json` SHA-256 is
`f0e6fdbf77c27960762db89253044323244e7ab4b549d2bf4e4d4d6fe0d8be76`.
All eleven extracted files were checked byte-for-byte against the supplied
task used for the recorded `1a1f096506dcceeab0b697f59937224400747968`
validation. They are archived verbatim, without reformatting or changing
their acceptance criteria. The original external SHA256SUMS manifest had a
task-text filename mismatch and is not included as a new authoritative manifest.

Adding these reference inputs and documenting their retrieval does not change
the operator implementation or retest the historical results. The seven local
acceptance materials identify the measured implementation revision separately
from the commit distributing this reference archive. Both are on the existing
`feat/rowcol-int8-quant` branch; no additional feature branch is required.
