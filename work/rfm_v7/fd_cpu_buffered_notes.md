# Buffered CPU FD4

This implementation preserves the full-annulus FD4 discretization. It applies the original narrow CSR rows in their original order, first for velocity and vorticity and then for the radial vorticity derivative. The outer streamfunction lift, the mode-zero closure, and the two wall vorticity rows use the same v6 formulas. The FFT still evaluates four inverse fields and one forward product with the original normalization and de-aliasing length. The azimuthal mean follows the same Parseval sum.

The fused C call stores radial values in FFT order. It reuses the streamfunction, vorticity, spectrum and mean work arrays. Physical products overwrite one inverse-transform plane. Returned mass and nonlinear histories have independent storage, and the implicit update still uses the original cached native band factors. The source uses no fast-math or fused multiply-add contraction.

## Reproduction

From the study root, with 64-bit MinGW GCC available on Windows:

```powershell
python work/rfm_v7/fd_cpu_buffered.py --build --long
```

Run from a fresh output copy to preserve existing evidence. The build targets AVX2, as supported by the measured Ryzen 9 9955HX. Omit `--build` to use the supplied library. The long validation reads the existing FD176 trajectory and the independently refined FD320 reference.

## Observations

Fifteen fixed-state trials span FD152/K80, FD176/K80 and FD320/K112. Random complex states use a nonzero complex boundary lift. Radial velocity, vorticity, both vorticity derivatives, mass, nonlinear flux and mean agree exactly. Retained outputs remain unchanged after later calls.

The t=72 run with FD176/K80 and 128 steps per time unit agrees exactly with the archived CPU trajectory on the three scored grids and both probe histories. The worst reference errors are 0.09815% in velocity and 0.38312% in vorticity. Alternating shedding persists at St = 0.1755343.

The three paired short-screen runs gave medians of 1.442 s and 1.034 s. The screening and long validation ran while other research workloads were active. The short screen identifies the candidate for isolated full-trajectory timing.
