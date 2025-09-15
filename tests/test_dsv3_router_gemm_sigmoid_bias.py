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
    return gemm_fused_sigmoid_bias(x, w, bias)


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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

    iters = 100
    g_trt = torch.cuda.CUDAGraph()
    g_ref = torch.cuda.CUDAGraph()
    x_static = x.clone()
    w_static = w.clone()
    bias_static = bias.clone()
    out_trt_static = torch.empty_like(out_trt)
    out_ref_static = torch.empty_like(out_ref)
    torch.cuda.synchronize()
    with torch.cuda.graph(g_trt):
        out_trt_static.copy_(trtllm_unfused_pipeline(x_static, w_static, bias_static))
    with torch.cuda.graph(g_ref):
        out_ref_static.copy_(flashinfer_pipeline(x_static, w_static, bias_static))
    for _ in range(10):
        g_trt.replay()
    for _ in range(10):
        g_ref.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g_trt.replay()
    end.record()
    torch.cuda.synchronize()
    trt_ms = start.elapsed_time(end) / iters
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(iters):
        g_ref.replay()
    end2.record()
    torch.cuda.synchronize()
    ref_ms = start2.elapsed_time(end2) / iters
    print(f"trtllm_avg_ms {trt_ms:.4f}")
    print(f"flashinfer_avg_ms {ref_ms:.4f}")
    print("PASSED")


if __name__ == "__main__":
    test_router_gemm_sigmoid_bias(16, 256, 7168, torch.bfloat16)
    test_router_gemm_sigmoid_bias(8, 256, 7168, torch.bfloat16)
    test_router_gemm_sigmoid_bias(1, 256, 7168, torch.bfloat16)
