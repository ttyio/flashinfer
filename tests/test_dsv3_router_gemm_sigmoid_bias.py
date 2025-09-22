import pytest
import torch
import tensorrt_llm  # noqa: F401
from flashinfer import gemm_fused_sigmoid_bias


def ref_pipeline(x, w, bias):
    logits = torch.matmul(x.float(), w.float())
    out = torch.sigmoid(logits) + bias.float()
    return out.to(torch.bfloat16)


def trtllm_unfused_pipeline(x, w, bias):
    logits_trt = torch.ops.trtllm.dsv3_router_gemm_op(x, w, None, torch.bfloat16)
    return torch.sigmoid(logits_trt) + bias


def flashinfer_pipeline(x, w, bias):
    return gemm_fused_sigmoid_bias(x, w, bias, backend="cuda")


def test_router_gemm_sigmoid_bias(num_tokens, num_experts, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(torch.ops, "trtllm") or not hasattr(
        torch.ops.trtllm, "dsv3_router_gemm_op"
    ):
        pytest.skip("trtllm op missing")
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w = torch.randn(num_experts, hidden_size, dtype=dtype, device=device).T
    bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

    out_trt = trtllm_unfused_pipeline(x, w, bias)
    out_ref = flashinfer_pipeline(x, w, bias)
    assert out_trt.dtype == torch.bfloat16 and out_ref.dtype == torch.bfloat16
    assert torch.allclose(out_trt, out_ref, rtol=5e-2, atol=5e-2)

    try:
        import triton
    except Exception:
        pytest.skip("triton required")

    def run_trt():
        trtllm_unfused_pipeline(x, w, bias)

    def run_ref():
        flashinfer_pipeline(x, w, bias)

    quantiles = [0.5, 0.2, 0.8]
    trt_ms, trt_min, trt_max = triton.testing.do_bench_cudagraph(
        run_trt, quantiles=quantiles
    )
    fi_ms, fi_min, fi_max = triton.testing.do_bench_cudagraph(
        run_ref, quantiles=quantiles
    )
    print(
        f"{num_tokens}x{num_experts}x{hidden_size}, trtllm {trt_ms:.4f} flashinfer {fi_ms:.4f}"
    )


if __name__ == "__main__":
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        test_router_gemm_sigmoid_bias(num_tokens, 256, 7168, torch.bfloat16)
