/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <aclnn/acl_meta.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {
using Outputs = std::array<torch::Tensor, 4>;
using WorkspaceFn = aclnnStatus (*)(const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    const aclTensor*,
                                    uint64_t*,
                                    aclOpExecutor**);
using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using ExecutorFn = aclnnStatus (*)(aclOpExecutor*);
using BindAddressFn = aclnnStatus (*)(aclOpExecutor*,
                                      size_t,
                                      aclTensor*,
                                      void*);
using PlanKey = std::tuple<int64_t, int64_t, int32_t, int64_t>;
constexpr uint64_t kSystemWorkspaceBytes = 16 * 1024 * 1024;
constexpr size_t kMaximumPlans = 64;

void validate_input(const torch::Tensor& input) {
  // This is the untrusted tensor boundary; rejected metadata must be an error
  // before any output allocation or device submission.
  if (input.device().type() != c10::DeviceType::PrivateUse1 ||
      input.scalar_type() != torch::kFloat16 || input.dim() != 2 ||
      !input.is_contiguous()) {
    throw std::invalid_argument(
        "RowColInt8Quant expects a contiguous rank-2 FP16 NPU tensor");
  }
  const int64_t m = input.size(0);
  const int64_t n = input.size(1);
  if (m < 1 || m > 4096 || n < 2 || n > 4096 || n % 2 != 0) {
    throw std::invalid_argument(
        "RowColInt8Quant requires 1<=M<=4096 and even 2<=N<=4096");
  }
  const auto* storage = static_cast<const torch_npu::NPUStorageImpl*>(
      input.storage().unsafeGetStorageImpl());
  if (storage->npu_desc_.npu_format_ != ACL_FORMAT_ND) {
    throw std::invalid_argument("RowColInt8Quant requires physical ND storage");
  }
}

torch::Tensor empty_nd(torch::IntArrayRef sizes,
                       torch::ScalarType dtype,
                       torch::Device device) {
  // Avoid another operator dispatch while preserving the registered NPU
  // storage/allocator, stream ownership and active-allocation accounting.
  auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  CHECK(c10::GetStorageImplCreate(c10::DeviceType::PrivateUse1));
  size_t bytes = c10::elementSize(dtype);
  for (const int64_t size : sizes) {
    bytes *= size;
  }
  auto storage =
      c10::make_storage_impl(c10::StorageImpl::use_byte_size_t(),
                             c10::SymInt(static_cast<int64_t>(bytes)),
                             allocator->allocate(bytes),
                             allocator,
                             /*resizable=*/true,
                             device);
  auto* npu_storage = static_cast<torch_npu::NPUStorageImpl*>(storage.get());
  auto output = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(std::move(storage)),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      c10::scalarTypeToTypeMeta(dtype));
  output.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  auto& descriptor = npu_storage->npu_desc_;
  descriptor.base_sizes_.assign(sizes.begin(), sizes.end());
  descriptor.storage_sizes_ = descriptor.base_sizes_;
  descriptor.base_strides_.assign(output.strides().begin(),
                                  output.strides().end());
  descriptor.origin_format_ = ACL_FORMAT_ND;
  descriptor.npu_format_ = ACL_FORMAT_ND;
  descriptor.data_type_ = output.dtype();
  return output;
}

class TensorDeleter final {
 public:
  void operator()(aclTensor* tensor) const { aclnn::detail::release(tensor); }
};
using Descriptor = std::unique_ptr<aclTensor, TensorDeleter>;

class ExecutorDeleter final {
 public:
  void operator()(aclOpExecutor* executor) const {
    static const ExecutorFn destroy =
        aclnn::detail::get_op_api_func<ExecutorFn>("aclDestroyAclOpExecutor");
    CHECK(destroy != nullptr);
    destroy(executor);
  }
};

Descriptor describe(const torch::Tensor& tensor) {
  static const auto create_tensor =
      aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorFn>(
          "aclCreateTensor");
  CHECK(create_tensor != nullptr);
  const aclDataType dtype = tensor.scalar_type() == torch::kFloat16
                                ? ACL_FLOAT16
                            : tensor.scalar_type() == torch::kInt8 ? ACL_INT8
                                                                   : ACL_FLOAT;
  Descriptor descriptor(create_tensor(tensor.sizes().data(),
                                      tensor.dim(),
                                      dtype,
                                      tensor.strides().data(),
                                      /*offset=*/0,
                                      ACL_FORMAT_ND,
                                      tensor.sizes().data(),
                                      tensor.dim(),
                                      tensor.data_ptr()));
  CHECK(descriptor);
  return descriptor;
}

class ExecutionPlan final {
 public:
  ExecutionPlan(const torch::Tensor& input,
                const Outputs& outputs,
                WorkspaceFn workspace_fn)
      : descriptors_{describe(input),
                     describe(outputs[0]),
                     describe(outputs[1]),
                     describe(outputs[2]),
                     describe(outputs[3])} {
    aclOpExecutor* executor = nullptr;
    CHECK_EQ(workspace_fn(descriptors_[0].get(),
                          descriptors_[1].get(),
                          descriptors_[2].get(),
                          descriptors_[3].get(),
                          descriptors_[4].get(),
                          &workspace_size_,
                          &executor),
             ACL_SUCCESS);
    executor_.reset(executor);
    static const ExecutorFn make_repeatable =
        aclnn::detail::get_op_api_func<ExecutorFn>(
            "aclSetAclOpExecutorRepeatable");
    CHECK(make_repeatable != nullptr);
    CHECK_EQ(make_repeatable(executor_.get()), ACL_SUCCESS);
  }

  ExecutionPlan(const ExecutionPlan&) = delete;
  ExecutionPlan& operator=(const ExecutionPlan&) = delete;

  void bind(const torch::Tensor& input, const Outputs& outputs) {
    static const BindAddressFn bind_input =
        aclnn::detail::get_op_api_func<BindAddressFn>("aclSetInputTensorAddr");
    static const BindAddressFn bind_output =
        aclnn::detail::get_op_api_func<BindAddressFn>("aclSetOutputTensorAddr");
    CHECK(bind_input != nullptr && bind_output != nullptr);
    CHECK_EQ(bind_input(executor_.get(),
                        /*index=*/0,
                        descriptors_[0].get(),
                        input.data_ptr()),
             ACL_SUCCESS);
    for (size_t index = 0; index < outputs.size(); ++index) {
      CHECK_EQ(bind_output(executor_.get(),
                           index,
                           descriptors_[index + 1].get(),
                           outputs[index].data_ptr()),
               ACL_SUCCESS);
    }
  }

  uint64_t workspace_size() const { return workspace_size_; }
  aclOpExecutor* executor() const { return executor_.get(); }

 private:
  std::array<Descriptor, 5> descriptors_;
  std::unique_ptr<aclOpExecutor, ExecutorDeleter> executor_;
  uint64_t workspace_size_ = 0;
};

class PlanKeyHash final {
 public:
  size_t operator()(const PlanKey& key) const {
    size_t hash = 0;
    const std::array<int64_t, 4> parts{
        std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key)};
    for (const int64_t part : parts) {
      hash ^=
          std::hash<int64_t>{}(part) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

using PlanCache =
    std::unordered_map<PlanKey, std::unique_ptr<ExecutionPlan>, PlanKeyHash>;
thread_local PlanCache plans;
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
row_col_int8_quant(const torch::Tensor& input) {
  validate_input(input);
  c10_npu::OptionalNPUGuard guard;
  if (c10_npu::current_device() != input.device().index()) {
    guard.set_device(input.device());
  }
  const c10_npu::NPUStream stream =
      c10_npu::getCurrentNPUStream(input.device().index());
  static const WorkspaceFn workspace_fn =
      aclnn::detail::get_op_api_func<WorkspaceFn>(
          "aclnnRowColInt8QuantGetWorkspaceSize");
  static const ExecuteFn execute_fn =
      aclnn::detail::get_op_api_func<ExecuteFn>("aclnnRowColInt8Quant");
  CHECK(workspace_fn != nullptr && execute_fn != nullptr)
      << "Install the RowColInt8Quant custom OPP package before using this "
         "operator";
  Outputs outputs{empty_nd(input.sizes(), torch::kInt8, input.device()),
                  empty_nd(input.sizes(), torch::kInt8, input.device()),
                  empty_nd({input.size(0)}, torch::kFloat32, input.device()),
                  empty_nd({input.size(1)}, torch::kFloat32, input.device())};
  const PlanKey key{
      input.size(0), input.size(1), input.device().index(), stream.id()};
  auto plan = plans.find(key);
  if (plan == plans.end()) {
    auto created =
        std::make_unique<ExecutionPlan>(input, outputs, workspace_fn);
    if (plans.size() >= kMaximumPlans) {
      plans.erase(plans.begin());
    }
    plan = plans.emplace(key, std::move(created)).first;
  }
  plan->second->bind(input, outputs);
  const uint64_t workspace_bound =
      kSystemWorkspaceBytes + static_cast<uint64_t>(4) *
                                  ((input.size(0) + 63) / 64) *
                                  ((input.size(1) + 255) / 256) * 320;
  CHECK_LE(plan->second->workspace_size(), workspace_bound);
  aclrtStream raw_stream = stream.stream();
  auto workspace =
      empty_nd({static_cast<int64_t>(plan->second->workspace_size())},
               torch::kUInt8,
               input.device());
  c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                             stream);
  CHECK_EQ(execute_fn(workspace.data_ptr(),
                      plan->second->workspace_size(),
                      plan->second->executor(),
                      raw_stream),
           ACL_SUCCESS);
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}
}  // namespace xllm::kernel::npu
