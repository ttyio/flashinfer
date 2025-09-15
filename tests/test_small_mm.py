import pytest
import torch
from flashinfer import gemm_fused_sigmoid_bias


def ref_pipeline(x, w, bias):
    logits = torch.matmul(x.float(), w.float())
    out = torch.sigmoid(logits) + bias.float()
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_router_gemm_sigmoid_bias(num_tokens, num_experts, hidden_size, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w = torch.randn(num_experts, hidden_size, dtype=dtype, device=device).T
    bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    print(f"bias shape: {bias.shape}")

    out_trt = gemm_fused_sigmoid_bias(x, w, bias)
    out_ref = ref_pipeline(x, w, bias)
    assert out_trt.dtype == torch.bfloat16 and out_ref.dtype == torch.bfloat16
    assert torch.allclose(out_trt, out_ref, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    test_router_gemm_sigmoid_bias(16, 256, 7168, torch.bfloat16)
