import torch
import triton
import triton.language as tl
from triton.runtime import driver

@torch.jit.script
def naive_softmax(x):
    """使用原生 pytorch 计算 X 的逐行 softmax

    我们减去最大元素以避免溢出。Softmax 对这种位移是不变的。
    """
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret


# 其中有很多指针和数组指针的概念，需要从内存的角度去考虑代码
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # 程序起始行,这两个接口中的0都是指grid的第0维
    ## 当前block的编号
    row_start = tl.program_id(0)
    ## 一共有多少个block
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # 定位到第row_idx的起始指针。input_row_stride是以元素为单位的行步长，步长表示我们需要对指针增加多少以推进 1 行（如果转置的话，可能不是行优先存储）
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的（例如，类似于 CUDA 中的 __expf）。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape
    # 每次循环迭代的块大小是大于 `x` 列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 另一个技巧是通过增加每行分配的线程数来要求编译器使用更多的线程块 (`num_warps`)
    # 将在下一个教程中看到如何以更自然的方式自动调整此值，以免自己进行手动启发式处理。
    num_warps = 8
    # 软件流水线阶段的数量
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    # 分配输出空间
    y = torch.empty_like(x)
    # 预编译内核以获取寄存器使用情况并计算线程占用情况。
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)
    # 创建一些持久化程序。
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y

def calcu():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # `x_name`的不同可能值
        line_arg='provider',  # 其值对应图表中不同线条的参数名
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # `line_arg`的可能值
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # 线条的标签名
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # 线条样式
        ylabel="GB/s",  # y轴的标签名
        plot_name="softmax-performance",  # 图表的名称。也用作保存图表的文件名。
        args={'M': 4096},  # 不在`x_names`和`y_name`中的函数参数值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    calcu()
    benchmark.run(show_plots=True, print_data=True, save_path="./calcu_result")