# Native CUDA FD4 control

The CUDA control starts from the CPU FullFD matrices, factors, outer cosine grid, Robin mean closure and boundary lift. It retains the same Fourier cutoffs, exact de-aliasing length, explicit advection and SBDF2 time levels. The first step uses the original CPU backward-Euler initialization before uploading the starting state. There is no subsequent stepwise host transfer.

For each nonzero mode, write the discrete curl as (Omega_kpsi= (B_0+k^2B_1)psi + a,b), where (a) is the mode-1 boundary lift. The GPU uses short stencils for (D_1 B_0) and (D_1 B_1) to form the radial curl derivative. The corresponding mean stencil includes the original Robin extension. This composition changes the order of FP64 operations. It leaves the mathematical discretization unchanged. The four inverse FFT planes are (u_r,u_	heta,D_1omega,partial_	hetaomega), followed by one forward FFT of the physical advection. The tangential mean source is formed using the same real Parseval product as FullFD. The RHS kernel includes the inverse FFT normalization and original radial factor.

The two linear paths use the same DGBTRF factors. The band path replays pivoting and triangular substitutions with one warp per angular mode. It checks that all active band diagonals fit in one warp. The dense path applies the same factorization to the identity once using FP64 DGBTRS. This requires about 19.85 MB for FD176 and trades arithmetic and storage for more radial parallelism. Neither path changes the time step or resolution. The native graph changes dispatch only.

## Correctness pilots

FD176, 80 modes, 128 steps per time unit, the outer grid and 64-bit arithmetic were used for both pilots.

| Pilot | Band | Dense |
| --- | ---: | ---: |
| t=8 largest field difference from CPU, fractional | 4.49e-8 | 2.14e-7 |
| t=8 diagnostic linear stage, median CUDA event time | 127 µs | 49 µs |
| t=8 synchronized graph march | 0.302 s | 0.296 s |

The shorter event time motivated selection of the dense route for the t=72 pilot. At seven physical checkpoints and on three scoring grids, its worst field difference from CPU was (3.57	imes10^{-6}) as a fraction. Its worst velocity and vorticity errors relative to FD320 were 0.09824% and 0.38346%. Alternating shedding remained sustained, with (St=0.1755343).

The t=72 pilot took 3.175 s in total, including host setup 0.213 s, inverse preparation 0.035 s, runtime initialization 0.621 s, upload 0.0075 s, plan setup 0.018 s, graph setup 0.0020 s, synchronized march 2.267 s and output download 0.00034 s. The long pilot overlapped other accuracy work and used event instrumentation on an eager tail. These times describe that pilot only. Formal speed comparisons disable profiling and use interleaved isolated runs.

## Reproduction

Use Python with NumPy and SciPy, and the installed Windows CUDA 11.8 libraries and NVIDIA driver described in `cuda_environment_native_notes.md`.

```powershell
python work/rfm_v7/fd_cuda_native.py --horizon 8 --linear-solve band --profile-steps 16 --output fd_band_t8.json
python work/rfm_v7/fd_cuda_native.py --horizon 8 --linear-solve dense --profile-steps 16 --output fd_dense_t8.json
python work/rfm_v7/fd_cuda_native.py --horizon 72 --linear-solve dense --reference work/rfm_v6/parallel_refinement/fd320_ref_tk.npz --output fd_dense_t72.json --save-solution fd_dense_t72.npz
```

With no reusable runtime supplied, each solve creates a fresh solver, retains and releases its own CUDA context, builds the module and FFT plans, and includes those costs in its reported wall time. Process-level DLL loading, driver code cache and OS file caches can persist across runs. The timing does not assert a cleared driver cache or cold OS startup.
