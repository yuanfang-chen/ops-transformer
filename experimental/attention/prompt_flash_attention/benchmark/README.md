# How to use the benchmarking/correctness file

# Installation

To install the sparse kernel go in the `ops-transformer` folder and run:

```bash
./build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=prompt_flash_attention
./build_out/cann-ops-transformer-custom_linux-"$(uname -i)".run
(cd attention/prompt_flash_attention/torch_interface && bash build.sh custom)
```

## Parameters

Using the CAPITAL_LETTERS variables at the top of the file you can ru all the things you need. Wherever there is a list, it means you can set mutliple values, and the file will run correctness/time evaluations for **all the possible combinations**.

* `B_VALS`: batch sizes to test. For block-sparsity only `B_VALS = [1]` works for now.
* `H_VALS`: number of heads
* `S_VALS`: sequence lengths
* `D_VALS`: head dimensions
* `N_REPEATS`: how many runs to do to estimate time
* `N_WARMUP`: how many warmup runs before running `N_REPEATS` calls
* `ATTENTION_MATRIX`: what kind of attention matrix to use. By chosing this value, you create the matrix you want and you also automatically set all the parameters as they are required accordingly by torch_npu.npu_prompt_flash_attention. The possible values are (only the first few ones are probably interesting for you):
    * `blocks_optimized_batched`: our optimized block-sparse kernel. There is a batch dimension, but please keep it 1 for now.
    * `blocks_optimized`: same, without batch dimension (you can use either this or the other)
    * `dense`: all tokens are attended by all tokens
    * `sparse_block`: like dense, but you can pass a sparse block mask created like in blocks_optimized: Only works for H_VALS = 1!
    * `sparse_block_all_same`: like `sparse_block`, but all heads have the same mask: this works.
    * `lower_triangular`: lower triangular matrix, classic case. It's the default torch_npu behavior
    * `band`: torch_npu band mode: diagonal band. You can set the width of the diagonal with `BAND_PRE_TOKENS` and `BAND_POST_TOKENS`
    * `custom`: you can create your own custom mask in `make_custom_mask`. Will use the default `dense` pattern for the torch_npu version
* `SPARSITY_VALS`: how many blocks to activate: randomly chosen, but each Q-block row will have the same number of blocks
* `BLOCK_SIZE_Q`, `BLOCK_SIZE_KV`: For now, only the defaulst 128x512 are supported for the `blocks_optimized*` version
* `BLOCK_MASK_SEED`: for reproducibility of random sampling. Can set to any value to change random sampling
* `USE_FRAME`: Adds a frame on top of the block-sparse mask. If you want to change the frame shape, please modify the method `generate_sparse_blocks_by_row_with_frame`. If `S` (in `S_VALS`) length is too short the mask might be dense. For now, the frame is always:
    * First 8 KV-blocks are active for all rows
    * First 29 Q-rows are completely active
    * Last Q-row is compeletely active
    * Last KV-column is all active
* `PRINT_OUTPUTS`: For manual visualization of output tensors
* `PRINT_MASK`: To visually check if the maks is as expected
* `PRINT_BLOCK_EQUALITY`: if the correctness test fails, you can check which blocks in the outputs match with the reference and which not. The block granularity to visualize is given by `PRINT_HEIGHT` and `PRINT_WIDTH`
* `RUN_REFERENCE`: Besides measuring time, also test correctness. This only is possible for small enough masks, otherwise you will get OOM in the reference run. For this reason, you can disable this for very long contexts, and enable it for short enough sequences.
* `TORCH_REFERENCE`: If `True`, correctness will be tested against the torch attention implementation. If `False`, for block-sparse attention the reference is the torch_npu dense with the corresponding attention mask (like in `sparse_block` mode). Please only use `H=1`, as torch_npu doesn't support different masks for different heads.

## Examples

### Test correctness and speed with multiple values (and shorter sequence lengths)

#### Input

```python
B_VALS = [1]
H_VALS = [16, 14]
S_VALS = [4096, 10_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2
ATTENTION_MATRIX = "blocks_optimized_batched" 

SPARSITY_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = False   # For short inputs, better to disable the frame or the mask will be dense

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = True
TORCH_REFERENCE = True  # If False, will instead run the torch_npu reference
```

#### Output:

```
(base) root@5e47d24afe58:/workspace/ops-transformer/benchmark# python benchmark.py 
============================================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
============================================================================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]  Ref_BW_[TB/sec]  Our_BW_[TB/sec]
--------------------------------------------------------------------------------------------------------------------------------------------
 16   1   4096   4096  128      0.10             yes           17638.60            1328.46            0.004            0.051
 16   1   4096   4096  128      0.20             yes           17645.56            1166.37            0.004            0.058
 16   1   4096   4096  128      0.30             yes           17642.52            1162.32            0.004            0.058
 16   1   4096   4096  128      0.40             yes           17622.96            1060.28            0.004            0.063
 16   1   4096   4096  128      0.50             yes           17628.49            1025.70            0.004            0.065
 16   1   4096   4096  128      0.60             yes           17626.99             980.80            0.004            0.068
 16   1   4096   4096  128      0.70             yes           17646.98             936.72            0.004            0.072
 16   1   4096   4096  128      0.80             yes           17669.57             955.56            0.004            0.070
 16   1   4096   4096  128      0.90             yes           17663.78             910.45            0.004            0.074
 16   1  10000  10000  128      0.10             yes          110451.61            7973.21            0.001            0.021
 16   1  10000  10000  128      0.20             yes          110456.47            7199.73            0.001            0.023
 16   1  10000  10000  128      0.30             yes          110440.53            6303.86            0.001            0.026
 16   1  10000  10000  128      0.40             yes          110435.28            5396.22            0.001            0.030
 16   1  10000  10000  128      0.50             yes          110441.05            4525.04            0.001            0.036
 16   1  10000  10000  128      0.60             yes          110446.19            3694.06            0.001            0.044
 16   1  10000  10000  128      0.70             yes          110438.79            2830.48            0.001            0.058
 16   1  10000  10000  128      0.80             yes          110458.57            1968.18            0.001            0.083
 16   1  10000  10000  128      0.90             yes          110501.12            1105.10            0.001            0.148
 24   1   4096   4096  128      0.10             yes           26154.51            1991.36            0.004            0.051
 24   1   4096   4096  128      0.20             yes           26161.87            1757.80            0.004            0.057
 24   1   4096   4096  128      0.30             yes           26115.79            1740.01            0.004            0.058
 24   1   4096   4096  128      0.40             yes           26215.35            1492.14            0.004            0.067
 24   1   4096   4096  128      0.50             yes           26202.39            1243.03            0.004            0.081
 24   1   4096   4096  128      0.60             yes           26141.22            1091.14            0.004            0.092
 24   1   4096   4096  128      0.70             yes           26161.22            1028.43            0.004            0.098
 24   1   4096   4096  128      0.80             yes           26180.69            1025.53            0.004            0.098
 24   1   4096   4096  128      0.90             yes           26216.21             971.07            0.004            0.104
```
Reference goes OOM for 10_000 sequence lenght and 24 heads (not shown).

On the left you see the inputs, then the "yes" line checks correctness, after that you see the reference runtime and our runtime. After that you see the memory bandwidth usage, which is much better in our kernel.

Note that for short sequence lengths, the speedups are negligible because overheads are higher than actual computation.


# Benchmark performance (no correctness) for very long sequences

#### Input

```python
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2
ATTENTION_MATRIX = "blocks_optimized_batched" 

SPARSITY_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = True

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False
TORCH_REFERENCE = True  # If False, will instead run the torch_npu reference
```

#### Output

```
(base) root@5e47d24afe58:/workspace/ops-transformer/benchmark# python benchmark.py 
============================================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
============================================================================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]  Ref_BW_[TB/sec]  Our_BW_[TB/sec]
--------------------------------------------------------------------------------------------------------------------------------------------
  3   1 118806 118806  128      0.10             N/A                N/A          198423.45              N/A            0.002
  3   1 118806 118806  128      0.20             N/A                N/A          175165.15              N/A            0.002
  3   1 118806 118806  128      0.30             N/A                N/A          152040.25              N/A            0.002
  3   1 118806 118806  128      0.40             N/A                N/A          128694.68              N/A            0.003
  3   1 118806 118806  128      0.50             N/A                N/A          105982.90              N/A            0.003
  3   1 118806 118806  128      0.60             N/A                N/A           85362.30              N/A            0.004
  3   1 118806 118806  128      0.70             N/A                N/A           64800.79              N/A            0.006
  3   1 118806 118806  128      0.80             N/A                N/A           44342.04              N/A            0.008
  3   1 118806 118806  128      0.90             N/A                N/A           22639.43              N/A            0.016
============================================================================================================================================
```

To run the comparison against `"dense"` mode (standard `torch_npu.npu_prompt_flash_attention`), run 

```python
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2
ATTENTION_MATRIX = "dense" 

SPARSITY_VALS = [1]     # use only one value, won't be used at all, but it determines how many runs

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = True

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False
TORCH_REFERENCE = True  # If False, will instead run the torch_npu reference
```

Output:

```
(base) root@5e47d24afe58:/workspace/ops-transformer/benchmark# python benchmark.py 
============================================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
============================================================================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]  Ref_BW_[TB/sec]  Our_BW_[TB/sec]
--------------------------------------------------------------------------------------------------------------------------------------------
  3   1 118806 118806  128      1.00             N/A                N/A          262168.70              N/A            0.001
============================================================================================================================================
```