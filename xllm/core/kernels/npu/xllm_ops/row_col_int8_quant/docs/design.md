# RowColInt8Quant Design

## Contract

The input is one finite, contiguous FP16 ND matrix on the selected NPU device.
The C++ entry checks physical ND storage as well as logical contiguity, so an
internally blocked tensor cannot be misdescribed as ND to ACLNN.
The supported range is `1 <= M <= 4096`, `2 <= N <= 4096`, with even N.
There are no attributes. The four independent contiguous outputs are INT8
qRow/qCol of shape [M,N], FP32 rowAbsmax [M], and FP32 colAbsmax [N].
All computation and input/output data movement are on the NPU. Host code reads
only tensor metadata. It does not inspect values or change the input.

## Execution

The tiling grid consists of `ceil(M/64) * ceil(N/256)` tiles. Up to 48 AIV
cores process tiles cyclically. The ACLNN operator executes three phases in
one launch, using two hardware `SyncAll` barriers. Kernel direct calling uses
the same three phase bodies in three ordered launches on the current stream,
so that it does not depend on ACLNN-managed cross-core launch metadata.

1. Each tile reads its valid FP16 submatrix, pads its UB image with positive
   zero, converts to FP32 and takes absolute values. Data-block reduction
   followed by repeat reduction produces 64 row maxima. A binary tree over
   rows produces 256 column maxima. The tile stores these 320 FP32 values.
2. Cores separately reduce tile-local row statistics across column tiles and
   column statistics across row tiles. Each reduction gathers its partials
   with a strided DMA and reduces a zero-padded power-of-two UB array.
3. Each tile reads the original FP16 values again. Row factors are broadcast
   across columns, and column factors across rows. INT8 output conversion and
   two-dimensional stores are batched by tile.

The element address is `x[(tileRow*64+i)*N + tileCol*256+j]`. A tile's
statistics start at `workspace_user[tileIndex*320]`, with columns in [0,256)
and rows in [256,320). No atomics, full FP32 matrix, or full transpose is used.

## Numerical Handling

FP16 to FP32 conversion and the max operations are exact for finite FP16.
Absolute value and zero padding produce positive-zero absmax even for signed
zero input. Quantization computes a FP32 factor and FP32 product, clips to
[-127,127], and uses FP32-to-INT32 `CAST_RINT`. Converting those small integers
through FP16 to INT8 is exact. `DEQSCALE` is explicitly set to FP16 one because
the FP16-to-INT8 instruction consumes that register.

Zero denominators are replaced by the smallest positive FP16 value before
division. For zero axes the numerator values are all zero; their products
and INT8 outputs remain zero. Nonzero finite FP16 denominators are unchanged.
The vector division instruction may differ from CPU division near rounding
boundaries. The unchanged public verifier must check and record the permitted
half-integer-neighborhood differences, rather than using a global tolerance.

## Memory And Ordering

User workspace is exactly `4 * ceil(M/64) * ceil(N/256) * (64+256)` bytes.
The ACLNN workspace additionally reserves 16 MiB for CANN synchronization.
Direct calling uses the same allocation bound, keeping the user region at
the same offset. There is no persistent device workspace in the Python module.

Per-core UB buffers are 32 KiB FP16 input/conversion staging, 64 KiB FP32
values, 64 KiB FP32 scratch, 8 KiB factors/broadcast scratch, and 16 KiB INT8
output: 184 KiB total, within the target's 192 KiB UB. Conversion staging is
reused only after input values have been converted to the separate FP32 buffer.
Explicit cross-pipeline events order vector clears, DMA and vector reads,
and output DMA consumption before the output buffer is reused.

The C++ binding changes device only when necessary, with a restoring NPU
device guard, and uses the current PyTorch NPU stream. The allocator associates
fresh output/workspace allocations with that stream. The binding additionally
records input storage on the execution stream so input allocated elsewhere
cannot be recycled before queued work completes after early Python release.
Every call allocates four new outputs, including a prepared call. The prepared
call owns its input reference, but does not cache mutable outputs or workspaces.
The binding uses the registered NPU storage factory and caching allocator,
initializes ND storage metadata, and avoids a redundant operator dispatch for
empty outputs/workspace. It also uses a bounded thread-local
cache of at most 64 repeatable ACLNN execution plans. The cache key includes
shape, logical device and stream. Addresses are rebound on every call through
`aclSetInputTensorAddr` / `aclSetOutputTensorAddr`; cached descriptors do not own
tensor storage. No output or workspace is cached. The original two-stage
ACLNN API remains the formal execution path.

Reproduction uses `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`. Without it,
the tested default caching allocator sometimes returns oversized free blocks,
making active allocation peak exceed the task allowance despite identical
workspace requests. This configuration is local to the sourced project env.

## Validation Boundary

The original task package remains unchanged outside this project. Public
accuracy, fixed-protocol performance, engineering coverage and allocator
memory checks are separate gates. A public accuracy pass is not a full task
acceptance. Hardware/software deviations and incomplete community review,
repository publication or PR steps must remain visible in the delivery report.

Engineering validation uses inputs allocated on one stream and consumed on
another. A queued native matmul workload and an unfinished completion event
prove that the Python input reference is dropped before execution completes.
Allocation/write pressure on the allocation stream must not recycle that
storage while the consumer is pending. The two prepared streams submit calls
interleaved with both completion events still pending; retained-output stress
uses allocations matching every output's shape and dtype.
