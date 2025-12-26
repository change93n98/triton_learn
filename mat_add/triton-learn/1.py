"""
- basics
- test
- benchmark
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # load data from dram/vram/hbm to sram/on-chip memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None) # shape: (BLOCK_SIZE,)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    out = x + y
    
    # write data back to dram
    tl.store(output_ptr + offsets, out, mask=mask)



def add(x, y):
    # prepare output tensor
    output = torch.empty_like(x, device=DEVICE)

    # check tensors are same device 
    assert x.device == y.device == output.device == DEVICE

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)   

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

def test_add_kernel(size, atol=1e-3, rtol=1e-3):
    # create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    # run triton kernel & pytorch kernel
    z_tri = add(x, y)
    z_ref = x + y
    # compare results
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("test passed!")

# performance benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'pytorch'],
        line_names=['triton', 'pytorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='add-performance',
        args={},
    )
)
def benchmark_add(size, provider):
    # create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    quantiles = [0.5, 0.05, 0.95]
    
    # benchmark triton kernel
    if provider == 'triton':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    
    # benchmark pytorch kernel
    elif provider == 'pytorch':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    
    # compute gbs
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(min_mis), gbps(max_mis)


if __name__ == "__main__":
    test_add_kernel(size=98432)

    import sys 
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_add.run(save_path=".", print_data=True)