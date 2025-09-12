import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, tma, mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    tcgen05_scaled_mma,
    allocate_tensor_memory,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    tensor_memory_descriptor,
)

@triton.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@gluon.jit
def _test(X, W, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    w_smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=W.type.layout.swizzle_byte_width,
        element_bitwidth=W.type.layout.element_bitwidth,
        rank=2, # reduce rank of W from 3D to 2D; a single expert is processed at a time
        transposed=W.type.layout.transposed,
        fp4_padded=W.type.layout.fp4_padded,
    )
    w_smem = gl.allocate_shared_memory(W.dtype, [1, BLOCK_N, BLOCK_K // 2], w_smem_layout)
    x_smem = gl.allocate_shared_memory(X.dtype, [1, BLOCK_M, BLOCK_K], X.type.layout)

    acc_shape: gl.constexpr = [BLOCK_N, BLOCK_M]
    acc_mma_shape: gl.constexpr = get_mma_instr_shape(acc_shape, gl.float32)
    acc_mem = allocate_tensor_memory(gl.float32, acc_shape, TensorMemoryLayout((acc_mma_shape[0], acc_mma_shape[1]), col_stride=1))
    x_scales = allocate_tensor_memory(gl.uint8, [BLOCK_M, BLOCK_K // 32], TensorMemoryScalesLayout())
    w_scales = allocate_tensor_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], TensorMemoryScalesLayout())

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K // 2)
    tma.async_copy_global_to_shared(X, [0, 0], bar, x_smem.index(0))
    tma.async_copy_global_to_shared(W, [0, 0, 0], bar, w_smem.index(0))

    mbarrier.wait(bar, 0)

    tcgen05_scaled_mma(
        w_smem.index(0), w_scales, "e2m1",
        x_smem.index(0).permute((1, 0)), x_scales, "e4m3",
        acc_mem, use_acc=False,
        mbarriers=[bar],
        mbarrier_preds=[True],
    )

    mbarrier.wait(bar, 1)


def test(BLOCK_M, BLOCK_N, BLOCK_K):
    x = torch.zeros(BLOCK_M, BLOCK_K, dtype=torch.float8_e4m3fn, device="cuda")
    w = torch.zeros(1, BLOCK_N, BLOCK_K // 2, dtype=torch.uint8, device="cuda")

    x_desc = TensorDescriptor(
        x, shape=x.shape, strides=x.stride(),
        block_shape=[BLOCK_M, BLOCK_K],
        layout=gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv),
    )
    w_desc = TensorDescriptor(
        w, shape=w.shape, strides=w.stride(),
        block_shape=[1, BLOCK_N, BLOCK_K // 2],
        layout=gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=3,
            transposed=False,
            fp4_padded=True,
        ))

    k = _test[(1,)](x_desc, w_desc, BLOCK_M, BLOCK_N, BLOCK_K)
    with open(f"/tmp/{k.name}.ttgir", "w") as f:
        f.write(k.asm["ttgir"])
    with open(f"/tmp/{k.name}.ptx", "w") as f:
        f.write(k.asm["ptx"])
    print(k.name, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()

if __name__ == "__main__":
    test(BLOCK_M=16, BLOCK_N=256, BLOCK_K=128)
    test(BLOCK_M=16, BLOCK_N=128, BLOCK_K=256)
    test(BLOCK_M=32, BLOCK_N=256, BLOCK_K=128)
    test(BLOCK_M=32, BLOCK_N=128, BLOCK_K=256)
    test(BLOCK_M=16, BLOCK_N=256, BLOCK_K=256)
    test(BLOCK_M=32, BLOCK_N=256, BLOCK_K=256)
