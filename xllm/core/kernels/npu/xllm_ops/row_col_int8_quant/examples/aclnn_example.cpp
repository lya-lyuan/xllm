#include <acl/acl.h>
#include <aclnn_row_col_int8_quant.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(int status, const char* what) {
  if (status != 0)
    throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
}

struct Buffer {
  void* ptr = nullptr;
  explicit Buffer(size_t bytes) {
    if (bytes)
      Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "malloc");
  }
  ~Buffer() {
    if (ptr) aclrtFree(ptr);
  }
  Buffer(const Buffer&) = delete;
};

struct TensorDesc {
  aclTensor* ptr;
  TensorDesc(void* data, std::vector<int64_t> shape, aclDataType type) {
    std::vector<int64_t> stride(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i)
      stride[i - 1] = stride[i] * shape[i];
    ptr = aclCreateTensor(shape.data(),
                          shape.size(),
                          type,
                          stride.data(),
                          0,
                          ACL_FORMAT_ND,
                          shape.data(),
                          shape.size(),
                          data);
    if (!ptr) throw std::runtime_error("aclCreateTensor failed");
  }
  ~TensorDesc() { aclDestroyTensor(ptr); }
};

static void Example(aclrtStream stream) {
  // Raw FP16 encodings of [-1, 2; 3, -4]. Five allocations cannot overlap.
  const std::array<uint16_t, 4> host{0xbc00, 0x4000, 0x4200, 0xc400};
  Buffer x(8), qr(4), qc(4), r(8), c(8);
  Check(aclrtMemcpy(x.ptr, 8, host.data(), 8, ACL_MEMCPY_HOST_TO_DEVICE),
        "copy input");
  TensorDesc xd(x.ptr, {2, 2}, ACL_FLOAT16), qrd(qr.ptr, {2, 2}, ACL_INT8),
      qcd(qc.ptr, {2, 2}, ACL_INT8), rd(r.ptr, {2}, ACL_FLOAT),
      cd(c.ptr, {2}, ACL_FLOAT);
  uint64_t workspaceBytes = 0;
  aclOpExecutor* executor = nullptr;
  Check(
      aclnnRowColInt8QuantGetWorkspaceSize(
          xd.ptr, qrd.ptr, qcd.ptr, rd.ptr, cd.ptr, &workspaceBytes, &executor),
      "get workspace");
  Buffer workspace(workspaceBytes);
  Check(aclnnRowColInt8Quant(workspace.ptr, workspaceBytes, executor, stream),
        "launch");
  Check(aclrtSynchronizeStream(stream), "synchronize");
  std::array<int8_t, 4> row{}, col{};
  std::array<float, 2> rows{}, cols{};
  Check(aclrtMemcpy(row.data(), 4, qr.ptr, 4, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy qRow");
  Check(aclrtMemcpy(col.data(), 4, qc.ptr, 4, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy qCol");
  Check(aclrtMemcpy(rows.data(), 8, r.ptr, 8, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy row absmax");
  Check(aclrtMemcpy(cols.data(), 8, c.ptr, 8, ACL_MEMCPY_DEVICE_TO_HOST),
        "copy col absmax");
  if (row != std::array<int8_t, 4>{-64, 127, 95, -127} ||
      col != std::array<int8_t, 4>{-42, 64, 127, -127} ||
      rows != std::array<float, 2>{2, 4} ||
      cols != std::array<float, 2>{3, 4}) {
    throw std::runtime_error("output mismatch");
  }
  std::cout << "ACLNN example passed; workspace_bytes=" << workspaceBytes
            << '\n';
}

int main() {
  aclrtStream stream = nullptr;
  bool initialized = false, device = false;
  int result = 0;
  try {
    Check(aclInit(nullptr), "aclInit");
    initialized = true;
    Check(aclrtSetDevice(0), "set device");
    device = true;
    Check(aclrtCreateStream(&stream), "create stream");
    Example(stream);
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << '\n';
    result = 1;
  }
  if (stream) aclrtDestroyStream(stream);
  if (device) aclrtResetDevice(0);
  if (initialized) aclFinalize();
  return result;
}
