import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

from flashinfer import bmm_fp4, e2m1_and_ufp8sf_scale_to_float, fp4_quantize


def quant_fp4(a):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(),
        a_global_sf.cuda(),
        sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
        folding_batch_dim_to_m=False,
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_cutlass(a):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    is_column_major = a.stride(-2) == 1
    if is_column_major:
        a = a.transpose(-2, -1)
    a_fp4, a_sf = scaled_fp4_quant(a, a_global_sf)
    if is_column_major:
        a_fp4 = a_fp4.transpose(-2, -1)
        a_sf = a_sf.transpose(-2, -1)

    return a_fp4, a_sf, a_global_sf


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

kE2M1ToFloatArray = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


def e2m1_to_fp32(int4_value):
    signBit = int4_value & 0x8
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if signBit:
        float_result = -float_result
    return float_result


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    # Get upper 4 bits
    highHalfByte = (a & 0xF0) >> 4
    # Get lower 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([e2m1_to_fp32(x) for x in highHalfByte]).to(a.device)
    fL = torch.tensor([e2m1_to_fp32(x) for x in lowHalfByte]).to(a.device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    sf_m, sf_k = a_sf_swizzled.shape
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(tensor_fp4, tensor_sf, global_scale, dtype, block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
):
    _, m_k = a_fp4.shape
    n_k, _ = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, block_size=block_size
    )
    b_in_dtype = dequantize_to_dtype(
        b_fp4.transpose(-1, -2),
        b_sf,
        b_global_scale.transpose(-1, -2),
        dtype=dtype,
        block_size=block_size,
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


def save_cutlass_checkpoints(b, m, n, k):
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    (input_fp4_cutlass, input_inv_s_cutlass, global_sf_input_cutlass) = (
        quant_fp4_cutlass(input)
    )
    torch.save(
        input,
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_tensor_b{b}_m{m}_k{k}.pt",
    )
    torch.save(
        input_fp4_cutlass,
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_fp4_cutlass_b{b}_m{m}_k{k}.pt",
    )
    torch.save(
        input_inv_s_cutlass,
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_inv_s_cutlass_b{b}_m{m}_k{k}.pt",
    )
    torch.save(
        global_sf_input_cutlass,
        f"/home/scratch.vincenth_inf/trees/flashinfer/global_sf_input_cutlass_b{b}_m{m}_k{k}.pt",
    )

    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    (mat2_fp4, mat2_inv_s, global_sf_mat2) = quant_fp4_cutlass(mat2)
    torch.save(
        mat2,
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_tensor_b{b}_n{n}_k{k}.pt",
    )
    torch.save(
        mat2_fp4,
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_fp4_cutlass_b{b}_n{n}_k{k}.pt",
    )
    torch.save(
        mat2_inv_s,
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_inv_s_cutlass_b{b}_n{n}_k{k}.pt",
    )
    torch.save(
        global_sf_mat2,
        f"/home/scratch.vincenth_inf/trees/flashinfer/global_sf_mat2_cutlass_b{b}_n{n}_k{k}.pt",
    )

    out = cutlass_scaled_fp4_mm(
        input_fp4_cutlass, mat2_fp4, input_inv_s_cutlass, mat2_inv_s, alpha, dtype
    )


def load_cutlass_checkpoints(b, m, n, k):
    input = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_tensor_b{b}_m{m}_k{k}.pt"
    )
    input_fp4_cutlass = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_fp4_cutlass_b{b}_m{m}_k{k}.pt"
    )
    input_inv_s_cutlass = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/input_inv_s_cutlass_b{b}_m{m}_k{k}.pt"
    )
    global_sf_input_cutlass = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/global_sf_input_cutlass_b{b}_m{m}_k{k}.pt"
    )
    mat2 = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_tensor_b{b}_n{n}_k{k}.pt"
    )
    mat2_fp4 = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_fp4_cutlass_b{b}_n{n}_k{k}.pt"
    )
    mat2_inv_s_cutlass = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/mat2_inv_s_cutlass_b{b}_n{n}_k{k}.pt"
    )
    global_sf_mat2_cutlass = torch.load(
        f"/home/scratch.vincenth_inf/trees/flashinfer/global_sf_mat2_cutlass_b{b}_n{n}_k{k}.pt"
    )

    print(f"input_shape: {input.shape} input_dtype: {input.dtype}")
    print(
        f"input_fp4_cutlass_shape: {input_fp4_cutlass.shape} input_fp4_cutlass_dtype: {input_fp4_cutlass.dtype} input_fp4_cutlass_stride: {input_fp4_cutlass.stride()}"
    )
    print(
        f"input_inv_s_cutlass_shape: {input_inv_s_cutlass.shape} input_inv_s_cutlass_dtype: {input_inv_s_cutlass.dtype} input_inv_s_cutlass_stride: {input_inv_s_cutlass.stride()}"
    )
    print(
        f"global_sf_input_cutlass_shape: {global_sf_input_cutlass.shape} global_sf_input_cutlass_dtype: {global_sf_input_cutlass.dtype}"
    )
    print(f"mat2_shape: {mat2.shape} mat2_dtype: {mat2.dtype}")
    print(
        f"mat2_fp4_cutlass_shape: {mat2_fp4.shape} mat2_fp4_cutlass_dtype: {mat2_fp4.dtype} mat2_fp4_cutlass_stride: {mat2_fp4.stride()}"
    )
    print(
        f"mat2_inv_s_cutlass_shape: {mat2_inv_s_cutlass.shape} mat2_inv_s_cutlass_dtype: {mat2_inv_s_cutlass.dtype} mat2_inv_s_cutlass_stride: {mat2_inv_s_cutlass.stride()}"
    )
    print(
        f"global_sf_mat2_cutlass_shape: {global_sf_mat2_cutlass.shape} global_sf_mat2_cutlass_dtype: {global_sf_mat2_cutlass.dtype}"
    )
    return (
        input,
        input_fp4_cutlass,
        input_inv_s_cutlass,
        global_sf_input_cutlass,
        mat2,
        mat2_fp4,
        mat2_inv_s_cutlass,
        global_sf_mat2_cutlass,
    )


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("m", [48])
@pytest.mark.parametrize("n", [128])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16])
def test_bmm_fp4(b, m, n, k, res_dtype):

    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)

    (input_fp4, input_inv_s, global_sf_input) = quant_fp4(input)
    (mat2_fp4, mat2_inv_s, global_sf_mat2) = quant_fp4(mat2)

    ref = torch.bmm(input, mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2)

    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)
    print(
        f"input_fp4: {input_fp4.shape} input_fp4_stride: {input_fp4.stride()} mat2_fp4: {mat2_fp4.shape} mat2_fp4_stride: {mat2_fp4.stride()} input_inv_s: {input_inv_s.shape} input_inv_s_stride: {input_inv_s.stride()} mat2_inv_s: {mat2_inv_s.shape} mat2_inv_s_stride: {mat2_inv_s.stride()} global_sf_input: {global_sf_input} global_sf_mat2: {global_sf_mat2}"
    )
    bmm_fp4(input_fp4, mat2_fp4, input_inv_s, mat2_inv_s, alpha, res_dtype, res)

    torch.cuda.synchronize()
    print(f"ref: {ref}")
    print(f"res: {res}")
    assert torch.testing.assert_close(ref, res, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__])
