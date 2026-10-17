# GPU compatibility notes

The GPU backend uses PyCUDA and CUDA C. This repository targets a legacy Python 2.7 environment, while CUDA 12.4 is a modern toolkit. These versions may not form a supported build combination.

Before trying to install PyCUDA:

1. Confirm `python2.7 --version` reports Python 2.7.x.
2. Confirm `nvcc --version` reports the installed CUDA toolkit.
3. Confirm `nvidia-smi` can see the GPU and driver.
4. Check PyCUDA's release/build requirements for a version compatible with your compiler and CUDA toolkit.
5. Keep this legacy environment isolated; Python 2.7 is end-of-life.

The program imports PyCUDA only when `--backend gpu` is selected. CPU mode does not require PyCUDA. If GPU mode errors, the error is surfaced and the project does not silently label CPU execution as GPU execution.

## GTX 1050 Ti

The GTX 1050 Ti is a CUDA-capable GPU, but that alone does not guarantee a compatible PyCUDA/Python 2.7/CUDA 12.4 toolchain. Verify with a real GPU-mode run before drawing performance conclusions.
