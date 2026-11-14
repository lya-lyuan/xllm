#include "kernel_operator.h"
#include "row_col_int8_quant_tiling.h"

using namespace AscendC;
__aicore__ inline uint32_t MinCount(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

class RowColQuant {
 public:
  template <HardEvent event>
  __aicore__ inline void Fence() {
    auto id = pipe_.FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR qr,
                              GM_ADDR qc,
                              GM_ADDR r,
                              GM_ADDR c,
                              GM_ADDR workspace,
                              const RowColInt8QuantTilingData& t) {
    t_ = t;
    x_.SetGlobalBuffer((__gm__ half*)x);
    qr_.SetGlobalBuffer((__gm__ int8_t*)qr);
    qc_.SetGlobalBuffer((__gm__ int8_t*)qc);
    r_.SetGlobalBuffer((__gm__ float*)r);
    c_.SetGlobalBuffer((__gm__ float*)c);
    partial_.SetGlobalBuffer((__gm__ float*)GetUserWorkspace(workspace));
    pipe_.InitBuffer(input_, 64 * 256 * sizeof(half));
    pipe_.InitBuffer(values_, 64 * 256 * sizeof(float));
    pipe_.InitBuffer(scratch_, 64 * 256 * sizeof(float));
    pipe_.InitBuffer(aux_, 8192);
    pipe_.InitBuffer(out_, 64 * 256);
    // FP16-to-INT8 conversion consumes DEQSCALE, which is not reset between
    // tasks.
    SetDeqScale(half(1.0f));
    Fence<HardEvent::S_V>();
  }

  __aicore__ inline void LoadTile(uint32_t tile) {
    row_ = tile / t_.colTiles * 64;
    col_ = tile % t_.colTiles * 256;
    rows_ = MinCount(64, t_.m - row_);
    cols_ = MinCount(256, t_.n - col_);
    auto h = input_.Get<half>();
    Duplicate(h, half(0), 64 * 256);
    Fence<HardEvent::V_MTE2>();
    uint32_t padded = (cols_ + 15) / 16 * 16;
    DataCopyExtParams p{
        (uint16_t)rows_, cols_ * 2, (t_.n - cols_) * 2, (256 - padded) / 16, 0};
    DataCopyPad(h, x_[row_ * t_.n + col_], p, {true, 0, uint8_t(0), half(0)});
    // Zero-fill a fixed UB stride; DMA still reads only the valid input bytes.
    stride_ = 256;
    Fence<HardEvent::MTE2_V>();
    Cast(values_.Get<float>(), h, RoundMode::CAST_NONE, 64 * stride_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void Statistics(uint32_t tile) {
    LoadTile(tile);
    auto a = values_.Get<float>();
    auto s = scratch_.Get<float>();
    Abs(a, a, 64 * stride_);
    PipeBarrier<PIPE_V>();
    // Reduce each 8-float data block, then reduce the 32 values belonging to a
    // row.
    ReduceDataBlock<ReduceType::MAX>(s[512], a, 64, 128, 1, 1, 8);
    ReduceDataBlock<ReduceType::MAX>(s[1536], a[8192], 64, 128, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    ReduceRepeat<ReduceType::MAX>(
        s[256], s[512], 32, 64, 1, 1, 4, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    for (uint32_t span = 32 * 256; span >= 256; span /= 2) {
      Max(a, a, a[span], span);
      PipeBarrier<PIPE_V>();
    }
    DataCopy(s, a, 256);
    Fence<HardEvent::V_MTE3>();
    DataCopy(partial_[tile * 320], s, 320);
    Fence<HardEvent::MTE3_V>();
  }

  __aicore__ inline void FinalStatistics() {
    auto s = scratch_.Get<float>();
    for (uint32_t rt = GetBlockIdx(); rt < t_.rowTiles; rt += t_.blocks) {
      Duplicate(s, 0.0f, 16 * 64);
      Fence<HardEvent::V_MTE2>();
      DataCopyPad(s,
                  partial_[rt * t_.colTiles * 320 + 256],
                  {(uint16_t)t_.colTiles, 64 * 4, (320 - 64) * 4, 0, 0},
                  {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      for (uint32_t span = 8 * 64; span >= 64; span /= 2) {
        Max(s, s, s[span], span);
        PipeBarrier<PIPE_V>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(
          r_[rt * 64], s, {1, MinCount(64, t_.m - rt * 64) * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
    }
    for (uint32_t ct = GetBlockIdx(); ct < t_.colTiles; ct += t_.blocks) {
      Duplicate(s, 0.0f, 64 * 256);
      Fence<HardEvent::V_MTE2>();
      DataCopyPad(
          s,
          partial_[ct * 320],
          {(uint16_t)t_.rowTiles, 256 * 4, (t_.colTiles * 320 - 256) * 4, 0, 0},
          {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      for (uint32_t span = 32 * 256; span >= 256; span /= 2) {
        Max(s, s, s[span], span);
        PipeBarrier<PIPE_V>();
      }
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(
          c_[ct * 256], s, {1, MinCount(256, t_.n - ct * 256) * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
    }
  }

  __aicore__ inline void StoreTile(GlobalTensor<int8_t>& dst) {
    auto s = scratch_.Get<float>();
    auto ints = input_.Get<int32_t>();
    auto halves = input_.Get<half>()[8192];
    auto bytes = out_.Get<int8_t>();
    uint32_t count = rows_ * 256;
    Mins(s, s, 127.0f, count);
    PipeBarrier<PIPE_V>();
    Maxs(s, s, -127.0f, count);
    PipeBarrier<PIPE_V>();
    for (uint32_t offset = 0; offset < count; offset += 4096) {
      uint32_t chunk = MinCount(4096, count - offset);
      Cast(ints, s[offset], RoundMode::CAST_RINT, chunk);
      PipeBarrier<PIPE_V>();
      Cast(halves, ints, RoundMode::CAST_NONE, chunk);
      PipeBarrier<PIPE_V>();
      Cast(bytes[offset], halves, RoundMode::CAST_TRUNC, chunk);
      PipeBarrier<PIPE_V>();
    }
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(dst[row_ * t_.n + col_],
                bytes,
                {(uint16_t)rows_,
                 cols_,
                 (256 - (cols_ + 31) / 32 * 32) / 32,
                 t_.n - cols_,
                 0});
    Fence<HardEvent::MTE3_V>();
  }

  __aicore__ inline void Quantize(uint32_t tile) {
    LoadTile(tile);
    auto a = values_.Get<float>();
    auto s = scratch_.Get<float>();
    auto scale = aux_.Get<float>();
    DataCopyPad(scale, c_[col_], {1, cols_ * 4, 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(
        scale[256], r_[row_], {1, rows_ * 4, 0, 0, 0}, {false, 0, 0, 0});
    Fence<HardEvent::MTE2_V>();
    // The minimum positive FP16 value keeps zero axes safe without changing
    // finite nonzero scales.
    Maxs(scale, scale, 0x1p-24f, 320);
    Duplicate(scale[512], 127.0f, 320);
    PipeBarrier<PIPE_V>();
    Div(scale, scale[512], scale, 320);
    PipeBarrier<PIPE_V>();
    Brcb(scale[1024], scale[256], 8, {1, 8});
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < 4; ++j) {
      Mul(s[j * 64], a[j * 64], scale[1024], 64, rows_, {1, 1, 0, 32, 32, 1});
    }
    PipeBarrier<PIPE_V>();
    StoreTile(qr_);
    for (uint32_t j = 0; j < 4; ++j) {
      Mul(s[j * 64], a[j * 64], scale[j * 64], 64, rows_, {1, 1, 1, 32, 32, 0});
    }
    PipeBarrier<PIPE_V>();
    StoreTile(qc_);
  }

  __aicore__ inline void Process() {
    for (uint32_t tile = GetBlockIdx(); tile < t_.rowTiles * t_.colTiles;
         tile += t_.blocks)
      Statistics(tile);
    SyncAll();
    FinalStatistics();
    SyncAll();
    for (uint32_t tile = GetBlockIdx(); tile < t_.rowTiles * t_.colTiles;
         tile += t_.blocks)
      Quantize(tile);
  }
  __aicore__ inline void ProcessStage(uint32_t stage) {
    if (stage == 1) {
      FinalStatistics();
      return;
    }
    for (uint32_t tile = GetBlockIdx(); tile < t_.rowTiles * t_.colTiles;
         tile += t_.blocks) {
      if (stage == 0)
        Statistics(tile);
      else
        Quantize(tile);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, values_, scratch_, aux_, out_;
  GlobalTensor<half> x_;
  GlobalTensor<int8_t> qr_, qc_;
  GlobalTensor<float> r_, c_, partial_;
  RowColInt8QuantTilingData t_;
  uint32_t row_, col_, rows_, cols_, stride_;
};

#ifndef ROWCOL_DIRECT
extern "C" __global__ __aicore__ void row_col_int8_quant(GM_ADDR x,
                                                         GM_ADDR qRow,
                                                         GM_ADDR qCol,
                                                         GM_ADDR rowAbsmax,
                                                         GM_ADDR colAbsmax,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(RowColInt8QuantTilingData);
  GET_TILING_DATA(t, tiling);
  RowColQuant op;
  op.Init(x, qRow, qCol, rowAbsmax, colAbsmax, workspace, t);
  op.Process();
}
#else
extern "C" __global__ __aicore__ void row_col_direct(
    GM_ADDR x,
    GM_ADDR qr,
    GM_ADDR qc,
    GM_ADDR r,
    GM_ADDR c,
    GM_ADDR workspace,
    uint32_t stage,
    RowColInt8QuantTilingData t) {
  SetSysWorkspaceForce(workspace);
  RowColQuant op;
  op.Init(x, qr, qc, r, c, workspace, t);
  op.ProcessStage(stage);
}

extern "C" __attribute__((visibility("default"))) uint32_t
rowcol_launch(uint32_t blocks,
              void* stream,
              void* x,
              void* qr,
              void* qc,
              void* r,
              void* c,
              void* workspace,
              uint32_t m,
              uint32_t n) {
  RowColInt8QuantTilingData t{m, n, (m + 63) / 64, (n + 255) / 256, blocks};
  for (uint32_t stage = 0; stage < 3; ++stage) {
    auto status =
        row_col_direct<<<blocks, nullptr, stream> > >((GM_ADDR)x,
                                                      (GM_ADDR)qr,
                                                      (GM_ADDR)qc,
                                                      (GM_ADDR)r,
                                                      (GM_ADDR)c,
                                                      (GM_ADDR)workspace,
                                                      stage,
                                                      &t);
    if (status != 0) return status;
  }
  return 0;
}
#endif
