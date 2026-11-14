#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>
#include <torch/extension.h>

#include <array>
#include <map>
#include <memory>
#include <tuple>

#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

using Tensor = at::Tensor;
using Outputs = std::array<Tensor, 4>;
using WorkspaceFn = aclnnStatus (*)(const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    uint64_t*,
                                    aclOpExecutor**);
using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

static void CheckInput(const Tensor& x) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "x must be on NPU");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
  TORCH_CHECK(x.dim() == 2, "x must have rank 2");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 4096, "M must be in [1,4096]");
  TORCH_CHECK(x.size(1) >= 2 && x.size(1) <= 4096 && x.size(1) % 2 == 0,
              "N must be even and in [2,4096]");
  const auto* storage = static_cast<const torch_npu::NPUStorageImpl*>(
      x.storage().unsafeGetStorageImpl());
  TORCH_CHECK(storage->npu_desc_.npu_format_ == ACL_FORMAT_ND,
              "x must use physical ND storage");
}

struct AclTensorDeleter {
  void operator()(aclTensor* p) const { aclDestroyTensor(p); }
};
using Descriptor = std::unique_ptr<aclTensor, AclTensorDeleter>;
static Descriptor Describe(const Tensor& x) {
  aclDataType dtype = x.scalar_type() == at::kHalf   ? ACL_FLOAT16
                      : x.scalar_type() == at::kChar ? ACL_INT8
                                                     : ACL_FLOAT;
  auto* ptr = aclCreateTensor(x.sizes().data(),
                              x.dim(),
                              dtype,
                              x.strides().data(),
                              0,
                              ACL_FORMAT_ND,
                              x.sizes().data(),
                              x.dim(),
                              x.data_ptr());
  TORCH_CHECK(ptr, "aclCreateTensor failed");
  return Descriptor(ptr);
}

static Tensor EmptyND(at::IntArrayRef sizes,
                      at::ScalarType dtype,
                      c10::Device device) {
  // Use the registered NPU storage factory and allocator, without a second
  // operator dispatch.
  auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  TORCH_CHECK(c10::GetStorageImplCreate(c10::DeviceType::PrivateUse1),
              "NPU storage factory is not registered");
  size_t bytes = c10::elementSize(dtype);
  for (auto dim : sizes) bytes *= dim;
  auto storage =
      c10::make_storage_impl(c10::StorageImpl::use_byte_size_t(),
                             c10::SymInt(static_cast<int64_t>(bytes)),
                             allocator->allocate(bytes),
                             allocator,
                             true,
                             device);
  auto* npuStorage = static_cast<torch_npu::NPUStorageImpl*>(storage.get());
  auto result = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(dtype));
  result.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  auto& desc = npuStorage->npu_desc_;
  desc.base_sizes_.assign(sizes.begin(), sizes.end());
  desc.storage_sizes_ = desc.base_sizes_;
  desc.base_strides_.assign(result.strides().begin(), result.strides().end());
  desc.origin_format_ = ACL_FORMAT_ND;
  desc.npu_format_ = ACL_FORMAT_ND;
  desc.data_type_ = result.dtype();
  return result;
}

static Outputs AllocateOutputs(const Tensor& x) {
  return {EmptyND(x.sizes(), at::kChar, x.device()),
          EmptyND(x.sizes(), at::kChar, x.device()),
          EmptyND({x.size(0)}, at::kFloat, x.device()),
          EmptyND({x.size(1)}, at::kFloat, x.device())};
}

struct ExecutionPlan {
  std::array<Descriptor, 5> descriptors;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  ~ExecutionPlan() {
    if (executor) aclDestroyAclOpExecutor(executor);
  }
};
using PlanKey = std::tuple<int64_t, int64_t, int, c10::StreamId>;
static thread_local std::map<PlanKey, std::unique_ptr<ExecutionPlan>> plans;

static Outputs Run(const Tensor& x) {
  CheckInput(x);
  c10_npu::OptionalNPUGuard guard;
  if (c10_npu::current_device() != x.device().index())
    guard.set_device(x.device());
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto y = AllocateOutputs(x);
  static void* lib = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(lib, "cannot load private libcust_opapi.so: ", dlerror());
  static auto wsFn = reinterpret_cast<WorkspaceFn>(
      dlsym(lib, "aclnnRowColInt8QuantGetWorkspaceSize"));
  static auto runFn =
      reinterpret_cast<ExecuteFn>(dlsym(lib, "aclnnRowColInt8Quant"));
  TORCH_CHECK(wsFn && runFn, "private ACLNN API symbols are missing");
  PlanKey key{x.size(0), x.size(1), x.device().index(), stream.id()};
  auto it = plans.find(key);
  if (it == plans.end()) {
    auto plan = std::make_unique<ExecutionPlan>();
    plan->descriptors = {Describe(x),
                         Describe(y[0]),
                         Describe(y[1]),
                         Describe(y[2]),
                         Describe(y[3])};
    auto& d = plan->descriptors;
    auto status = wsFn(d[0].get(),
                       d[1].get(),
                       d[2].get(),
                       d[3].get(),
                       d[4].get(),
                       &plan->workspaceSize,
                       &plan->executor);
    TORCH_CHECK(status == ACL_SUCCESS,
                "GetWorkspaceSize failed: ",
                status,
                " ",
                aclGetRecentErrMsg());
    TORCH_CHECK(aclSetAclOpExecutorRepeatable(plan->executor) == ACL_SUCCESS,
                "cannot make ACLNN executor repeatable");
    if (plans.size() >= 64) plans.erase(plans.begin());
    it = plans.emplace(key, std::move(plan)).first;
  }
  auto& plan = *it->second;
  TORCH_CHECK(aclSetInputTensorAddr(
                  plan.executor, 0, plan.descriptors[0].get(), x.data_ptr()) ==
                  ACL_SUCCESS,
              "cannot rebind input address");
  for (size_t i = 0; i < y.size(); ++i) {
    TORCH_CHECK(
        aclSetOutputTensorAddr(
            plan.executor, i, plan.descriptors[i + 1].get(), y[i].data_ptr()) ==
            ACL_SUCCESS,
        "cannot rebind output address");
  }
  const uint64_t bound = 16 * 1024 * 1024 + 4 * ((x.size(0) + 63) / 64) *
                                                ((x.size(1) + 255) / 256) * 320;
  TORCH_CHECK(plan.workspaceSize <= bound,
              "ACLNN workspace exceeds task bound: ",
              plan.workspaceSize);
  auto rawStream = stream.stream();
  auto workspace = EmptyND(
      {static_cast<int64_t>(plan.workspaceSize)}, at::kByte, x.device());
  // Protect allocations used on a stream other than their allocation stream.
  c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(), stream);
  // Outputs and workspace are allocated on this stream; only the input may
  // originate elsewhere.
  auto status =
      runFn(workspace.data_ptr(), plan.workspaceSize, plan.executor, rawStream);
  TORCH_CHECK(status == ACL_SUCCESS,
              "ACLNN launch failed: ",
              status,
              " ",
              aclGetRecentErrMsg());
  return y;
}

using DirectFn = uint32_t (*)(uint32_t,
                              void*,
                              void*,
                              void*,
                              void*,
                              void*,
                              void*,
                              void*,
                              uint32_t,
                              uint32_t);
static Outputs RunDirect(const Tensor& x) {
  CheckInput(x);
  c10_npu::OptionalNPUGuard guard;
  if (c10_npu::current_device() != x.device().index())
    guard.set_device(x.device());
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto y = AllocateOutputs(x);
  static void* lib = dlopen(ROWCOL_DIRECT_LIBRARY, RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(lib, "cannot load direct kernel library: ", dlerror());
  static auto launch = reinterpret_cast<DirectFn>(dlsym(lib, "rowcol_launch"));
  TORCH_CHECK(launch, "direct launch symbol missing");
  const uint32_t m = x.size(0), n = x.size(1);
  const uint32_t tiles = ((m + 63) / 64) * ((n + 255) / 256);
  const uint32_t blocks = std::min<uint32_t>(48, tiles);
  auto rawStream = stream.stream();
  auto workspace =
      EmptyND({16 * 1024 * 1024 + static_cast<int64_t>(4) * tiles * 320},
              at::kByte,
              x.device());
  c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(), stream);
  auto status = launch(blocks,
                       rawStream,
                       x.data_ptr(),
                       y[0].data_ptr(),
                       y[1].data_ptr(),
                       y[2].data_ptr(),
                       y[3].data_ptr(),
                       workspace.data_ptr(),
                       m,
                       n);
  TORCH_CHECK(status == 0, "direct kernel launch failed: ", status);
  return y;
}

static Tensor ValidateArguments(pybind11::handle inputs,
                                pybind11::handle attributes) {
  if (!PyDict_Check(attributes.ptr()))
    throw pybind11::type_error("attributes must be a dictionary");
  if (PyDict_Size(attributes.ptr()) != 0)
    throw pybind11::value_error("unknown attributes");
  if ((!PyList_Check(inputs.ptr()) && !PyTuple_Check(inputs.ptr())) ||
      PySequence_Size(inputs.ptr()) != 1) {
    throw pybind11::value_error("expected exactly one input tensor");
  }
  return pybind11::reinterpret_borrow<pybind11::sequence>(inputs)[0]
      .cast<Tensor>();
}

static pybind11::tuple AsTuple(const Outputs& outputs) {
  pybind11::tuple result(4);
  for (size_t i = 0; i < outputs.size(); ++i)
    result[i] = pybind11::cast(outputs[i]);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run",
      [](pybind11::handle inputs, pybind11::handle attributes) {
        return AsTuple(Run(ValidateArguments(inputs, attributes)));
      },
      pybind11::arg("inputs"),
      pybind11::arg("attributes"));
  m.def(
      "run_direct",
      [](pybind11::handle inputs, pybind11::handle attributes) {
        return AsTuple(RunDirect(ValidateArguments(inputs, attributes)));
      },
      pybind11::arg("inputs"),
      pybind11::arg("attributes"));
  m.def("prepare_input",
        [](pybind11::handle inputs, pybind11::handle attributes) {
          auto x = ValidateArguments(inputs, attributes);
          CheckInput(x);
          return x;
        });
  m.def("run_tensor", [](const Tensor& x) { return AsTuple(Run(x)); });
  m.def("validate", &CheckInput);
}
