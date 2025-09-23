#include <ATen/cuda/EmptyTensor.h>
#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "pytorch_extension_utils.h"

namespace torch_ext {

struct _1SM {};
struct _2SM {};

template <typename T>
struct SMTypeAdapter {};

template <>
struct SMTypeAdapter<_1SM> {
  static int const Scale = 1;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
};

template <>
struct SMTypeAdapter<_2SM> {
  static int const Scale = 2;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
};

template <int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_, int32_t CGA_M_, int32_t CGA_N_,
          int32_t CGA_K_, typename XSM_>
struct DeviceBf16GemmSigmodBias {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using ElementC = cutlass::bfloat16_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementD = cutlass::bfloat16_t;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using MmaTileShape = cute::Shape<cute::Int<CTA_M_ * SMTypeAdapter<XSM_>::Scale>,
                                   cute::Int<CTA_N_>, cute::Int<CTA_K_>>;
  using ClusterShape = cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>;
  // using ClusterShape = cute::Shape<int, int, _1>;

  using EpilogSchedule = typename SMTypeAdapter<XSM_>::EpilogueSchedule;
  using MainloopSchedule = typename SMTypeAdapter<XSM_>::MainloopSchedule;

  using CustomEVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::plus, ElementD, ElementAccumulator,
          cutlass::FloatRoundStyle::round_to_nearest>,  // bias + sigmoid(acc)
      cutlass::epilogue::fusion::Sm90RowBroadcast<0, MmaTileShape, ElementD, ElementAccumulator,
                                                  cute::Stride<cute::_0, cute::_1, int64_t>,
                                                  AlignmentD>,  // per-column bias
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::Sm90Compute<
              cutlass::epilogue::thread::Sigmoid, ElementAccumulator, ElementAccumulator,
              cutlass::FloatRoundStyle::round_to_nearest>,  // sigmoid(acc)
          cutlass::epilogue::fusion::Sm90AccFetch           // acc
          >>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutC, AlignmentD, EpilogSchedule,
      CustomEVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename GemmT>
typename GemmT::Arguments prepareBf16GemmSigmoidBiasArgs(void const* A, void const* B, void* D,
                                                         void const* bias, int32_t m, int32_t n,
                                                         int32_t k) {
  using CollectiveEpilogueT = typename GemmT::GemmKernel::CollectiveEpilogue;
  using FusionCallbacksT = typename CollectiveEpilogueT::FusionCallbacks;
  using EpilogueArgsT = typename FusionCallbacksT::Arguments;
  using ElementCompute = float;
  EpilogueArgsT epi{{static_cast<cutlass::bfloat16_t const*>(bias), cutlass::bfloat16_t(0),
                     cute::Stride<cute::_0, cute::_1, int64_t>{}},
                    {{}, {}},
                    {}};
  using StrideAT = typename GemmT::GemmKernel::StrideA;
  using StrideBT = typename GemmT::GemmKernel::StrideB;
  using StrideCT = typename GemmT::GemmKernel::StrideC;
  using StrideDT = typename GemmT::GemmKernel::StrideD;
  auto stride_A_T = cutlass::make_cute_packed_stride(StrideAT{}, {m, k, 1});
  auto stride_B_T = cutlass::make_cute_packed_stride(StrideBT{}, {n, k, 1});
  auto stride_C_T = cutlass::make_cute_packed_stride(StrideCT{}, {m, n, 1});
  auto stride_D_T = cutlass::make_cute_packed_stride(StrideDT{}, {m, n, 1});
  typename GemmT::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                      {m, n, k, 1},
                                      {static_cast<cutlass::bfloat16_t const*>(A), stride_A_T,
                                       static_cast<cutlass::bfloat16_t const*>(B), stride_B_T},
                                      {epi, static_cast<cutlass::bfloat16_t const*>(D), stride_C_T,
                                       static_cast<cutlass::bfloat16_t*>(D), stride_D_T}};
  arguments.scheduler.max_swizzle_size = 0;
  return arguments;
}

template <int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_, int32_t CGA_M_, int32_t CGA_N_,
          int32_t CGA_K_, typename XSM_>
size_t genericBf16GemmSigmoidBiasLauncher(void const* A, void const* B, void* D, void const* bias,
                                          int32_t m, int32_t n, int32_t k, void* workspace,
                                          size_t workspaceBytes, cudaStream_t stream) {
  using Bf16GemmOperator =
      typename DeviceBf16GemmSigmodBias<CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_, XSM_>::Gemm;
  Bf16GemmOperator gemm;
  auto arguments = prepareBf16GemmSigmoidBiasArgs<Bf16GemmOperator>(A, B, D, bias, m, n, k);
  auto requiredWorkspaceSize = gemm.get_workspace_size(arguments);
  if (!A && !B && !D) {
    return requiredWorkspaceSize;
  }
  if (requiredWorkspaceSize > workspaceBytes) {
    std::string errMsg("Requested workspace size insufficient. Required " +
                       std::to_string(requiredWorkspaceSize) + ", got " +
                       std::to_string(workspaceBytes));
    throw std::runtime_error(errMsg);
  }
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed in BF16 Gemm cutlass kernel can_implement(). Error: " +
                         std::string(cutlass::cutlassGetStatusString(status));
    throw std::runtime_error(errMsg);
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to initialize BF16 Gemm cutlass kernel. Error: " +
                         std::string(cutlass::cutlassGetStatusString(status));
    throw std::runtime_error(errMsg);
  }
  status = gemm.run(arguments, workspace, stream, nullptr, /*enablePDL*/ true);
  if (status != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to run BF16 Gemm cutlass kernel. Error: " +
                         std::string(cutlass::cutlassGetStatusString(status));
    throw std::runtime_error(errMsg);
  }
  return requiredWorkspaceSize;
}

at::Tensor small_gemm_fused_sigmoid_bias_impl(at::Tensor const& A, at::Tensor const& B,
                                              at::Tensor const& bias, at::Tensor out,
                                              at::Tensor workspace_buffer) {
  CHECK_INPUT_AND_TYPE(A, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(B, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(bias, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(out, at::ScalarType::BFloat16);

  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = B.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  int64_t const provided_workspace_size =
      workspace_buffer.numel() * workspace_buffer.element_size();
  size_t required_workspace_size = genericBf16GemmSigmoidBiasLauncher<64, 256, 128, 1, 1, 1, _1SM>(
      A.const_data_ptr(), B.const_data_ptr(), out.data_ptr(), bias.const_data_ptr(), M, N, K,
      nullptr, 0, stream);

  auto runKernel = [&](void* workspace) {
    genericBf16GemmSigmoidBiasLauncher<128, 64, 128, 2, 1, 1, _2SM>(
        A.const_data_ptr(), B.const_data_ptr(), out.data_ptr(), bias.const_data_ptr(), M, N, K,
        workspace, required_workspace_size, stream);
  };

  if (required_workspace_size > provided_workspace_size) {
    at::Tensor new_workspace =
        at::detail::empty_cuda({static_cast<int64_t>(required_workspace_size)},
                               at::ScalarType::Char, A.device(), std::nullopt);
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
  return out;
}

}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("small_gemm_fused_sigmoid_bias", &torch_ext::small_gemm_fused_sigmoid_bias_impl);
}
