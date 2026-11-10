# FP64 multifield pseudospectral flux: local contribution candidate

This is a standalone, locally validated extraction of the project's CUDA nonlinear operator. It is **not an upstream PR**, a generic FFT library, or a new Navier–Stokes benchmark. No cuFFTDx implementation or three-way performance comparison is included yet.

## What is included

- `fused_nonlinear_fft.cu`: byte-for-byte copy of the project-local kernel at `work/rfm_v8/fused_nonlinear_fft.cu`. No source from cuFFT, cuFFTDx, CuPy, SciPy, or LAPACK is included.
- `model.py`: the corresponding project-local mixed-radix CPU model, also copied without changes.
- `verify.py`: new standalone NumPy/ctypes harness. It independently checks the kernel against NumPy FFTs and retained-mode polynomial convolution. It does not import the original solver, require research snapshots, or measure latency.
- `validation.json`: the actual GPU validation receipt, with per-case and per-output-channel errors.
- `provenance.json`: source-copy hashes and the exact validation command.

The CUDA file preserves the original history-copy and FD mean-forcing interfaces so extraction does not silently change the operator. The mean inputs are already assembled; this sample does not construct a physical flow or its radial discretization.

## Operator contract

All arrays are contiguous, field-major FP64 or complex128. A half-spectrum is `[field, radial_row, N//2+1]`. Supported CUDA lengths are **200, 225, 243**, with exactly 128 threads per block and one block per row. Modes above retained `K` must be zero and `N > 3K`. DC and the even-length Nyquist coefficient must be real. The wrapper rejects other sizes, malformed layouts, nonfinite data, or invalid padding. Arbitrary sizes supported by the CPU plan builder are **not** CUDA-supported sizes.

The Galerkin variant reads the half-spectra of three real fields, `u_r`, `u_theta`, and `omega`, and returns the half-spectra of `u_r*omega` and `u_theta*omega`. The FD variant reads four real fields and returns the transform of `u_r*d_s_omega + u_theta*d_theta_omega`; the meanings include any radial metric factors already supplied by the caller. There are five logical field transforms in either variant. Packing uses two internal inverse complex transforms and one internal forward complex transform.

**Both transform directions are unnormalized**, matching cuFFT. NumPy's inverse is multiplied by `N` in the oracle. No physical `1/N` scaling is applied here: in the original solver it belongs to the subsequent projection or right-hand side. Old flux is copied to a separate history array before overwriting; the FD mean is summed in ascending mode order, excluding mode zero, into the selected parity slot. The other mean slot is unchanged.

## Run

CPU correctness (Python and NumPy only):

```powershell
python verify.py --output cpu_validation.json
```

Optional GPU correctness (NVIDIA driver and an installed CUDA toolkit with NVRTC):

```powershell
python verify.py --gpu --cuda-path 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8' --output validation.json
```

Run these from this folder. The GPU adapter creates its own context, compiles the six original entry points, runs tiny fixed-state fixtures, and releases its allocations/module/context. It uses no cuFFT plan and does not benchmark. The tested compiler emits compute_75 PTX; other toolkits and GPUs require their own validation. A Linux driver-loading path is provided but has not been executed here.

## Actual validation, 2026-11-10

The GPU command completed all **36 cases** on an NVIDIA GeForce RTX 5060 Laptop GPU, driver API version 13020, using the locally installed CUDA 11.8 NVRTC DLL. The fixtures cover three supported lengths, two operator variants, and six input types per variant: broadband, zero, constant, retained-edge, strongly imbalanced, and either a zero velocity channel (Galerkin) or additive cancellation (FD). Nine malformed-input checks independently confirm rejection of unsupported size, aliased bandwidth, nonfinite spectrum, complex DC, nonzero padding, noncontiguous layout, wrong precision, wrong output shape, and nonfinite history.

There are seven distinct fixture labels across the two variants. `channel_relative_max[c]` is `max(abs(actual[c] - expected[c])) / max(abs(expected[c]))`, with both maxima over all radial rows and half-spectrum modes; it is not a pointwise relative error. The value is `null` when the reference channel is identically zero. `uncancelled_scale_error` divides the largest absolute error over all output channels by one common scale: for each row and flux expression, multiply the separate physical-field maxima in each product, add those nonnegative product magnitudes, multiply by `N`, and take the largest value across rows and channels. A tiny positive floor handles the all-zero case. This common scale can hide weak-channel errors and must be interpreted alongside the per-channel metrics.

- Ordinary cases: largest reported per-output-channel relative error `1.1792813e-15`.
- All cases: maximum absolute error divided by a noncancelled product scale `3.8317967e-16`.
- Imbalanced stress: largest per-output-channel relative error `4.9741945e-6` after one velocity field is reduced by `1e-10`.
- Old flux history was checked exactly; FD mode sums and parity-slot preservation were checked separately. The retained-band convolution oracle also agreed with the FFT oracle within the prescribed scaled bound.

These are **fixed-state correctness checks**, not NS trajectory qualification. They add no new speed result and do not replace the paper's physical velocity/vorticity gates. Error values from these synthetic fixtures must not be silently substituted for the earlier archived measurements.

## New engineering guard and known limitations

The new Python adapter rejects packed input pairs whose nonzero per-row spectral maximum magnitudes differ by more than `1e8`, unless `allow_stress=True` is explicitly set for diagnostics. `verify.py` confirms rejection of every imbalanced fixture, then deliberately bypasses it to measure the failure mode. The original CUDA kernel is unchanged.

This `1e-8` balance floor is a conservative **dispatch policy**, not a sufficient condition for a relative-error theorem. Exactly zero channels do not enter that nonzero scale-ratio test; they require an absolute-error check, as relative error is undefined. Near-cancelling outputs can still have arbitrarily large relative error. A production solver should route rejected inputs to an independently verified unpacked path and continue checking per-channel errors; no such automatic fallback is implemented in this extraction. Reading back device data solely to decide this policy may itself be expensive; integrating a device-side or already-available-scale decision is future work.

## What remains before an upstream proposal

NVIDIA already publishes generic fused FFT/convolution examples. A useful new contribution would demonstrate the **inverse transforms → multiple cross-field products → forward transforms** CFD pattern with identical outputs, precision, padding, and batch sizes across:

1. A batched cuFFT baseline plus the same product operation.
2. An equivalently fused cuFFTDx implementation.
3. This project-local specialized fusion, if it adds useful insight.

That comparison is not implemented or measured here. It must report compilation/planning, first use, warm execution, repeated solves, field imbalance, and cancellation. A clean build/test outside this machine and upstream coding/contribution review are also outstanding. This directory deliberately makes no new license grant or DCO certification; rights and the receiving repository's requirements must be established before publication.

Primary sources consulted:

- [NVIDIA cuFFTDx examples](https://docs.nvidia.com/cuda/cufftdx/examples.html), including fusion, cuFFT, and optional LTO callback comparisons.
- [NVIDIA cuFFT documentation](https://docs.nvidia.com/cuda/cufft/index.html), including transform conventions and plan behavior.
- [NVIDIA CUDALibrarySamples contributing guide](https://github.com/NVIDIA/CUDALibrarySamples/blob/main/CONTRIBUTING.md).
