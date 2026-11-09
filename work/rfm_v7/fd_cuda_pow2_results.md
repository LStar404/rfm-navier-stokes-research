# CUDA replay of the high-resolution FD4 cylinder reference

The test replayed the same $Re=100$ full-annulus FD448/K144 run with 512 SBDF2 steps per time unit to $t=72$. Both CUDA arms used FP64, a 450-point dealiased angular FFT, and the original initial and boundary conditions. A staged radial pipeline constructed vorticity on the extended grid before applying its radial derivative. The first implicit step used the original scaled CPU solver. The device retained the two state and nonlinear histories for all remaining steps. The comparison used the saved CPU solution at the original times and seven snapshots on each of three physical score grids.

| Implicit BDF solve | End to end | March | Linear data | Worst velocity field | Worst vorticity field | Worst probe | Max crossing drift | Replay gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Folded dense $(DA)^{-1}D$ | 62.801 s | 60.700 s | 231.78 MB | 3.60e−6 | 1.68e−5 | 2.13e−5 | 4.84e−5 | Fail |
| Banded solve of $DAx=Db$ | 48.986 s | 47.678 s | 10.63 MB | 1.05e−9 | 4.25e−9 | 3.94e−9 | 7.02e−9 | Pass |

All field and probe errors are fractional relative L2. Crossing drift is in flow time units. Each replay gate was $10^{-6}$ in its respective units. The banded and CPU Strouhal numbers were 0.175559593623 and 0.175559593600. Both CUDA arms captured sustained alternating shedding at all three probes. The timings are concurrent, single-run diagnostics. The end-to-end clock covers fresh solver construction, CUDA initialization, transfers, FFT plan and graph construction, marching, synchronization, download and teardown. The archived CPU trajectory remains the reference for published results. The same-discretization replay addresses a separate question from reference convergence, for which the existing 0.1% uncertainty screen failed.

At $t=24,54,72$, the actual CUDA linear kernels were applied to assembled SBDF2 right-hand sides formed from the saved flow and a held-history start. Across the eight sampled modes, the maximum componentwise backward error in the original unscaled equation was $1.18\times10^{-12}$ for the folded dense operator and $4.79\times10^{-16}$ for the banded solve. The maximum forward errors against a 65-digit oracle were $2.64\times10^{-11}$ and $2.88\times10^{-13}$. These single-step errors indicate the precision cost of the folded representation. Both long runs used identical staged radial kernels, so this comparison isolates the linear-solve route and its rounding order.

The fixed-state derivative audit compared the frozen composed coefficients and the staged CPU stencil against the same 65-digit evaluation of the original double-precision stencil coefficients. The worst outer-eight relative $D_1\omega$ error moved from $1.78\times10^{-4}$ to $1.34\times10^{-4}$ with staging. The worst whole-radius relative error moved from $2.20\times10^{-5}$ to $3.59\times10^{-5}$. At the minimum log spacing of $1.53\times10^{-5}$, the dominant error came from the outer $k=1$ lifting mode. Its composed cancellation ratio reached $9.60\times10^{12}$ at $t=72$. Staging reproduces the CPU evaluation order more closely and the audit found mixed accuracy effects. A language rewrite alone leaves this cancellation mechanism in place.

Run the following from `work/rfm_v7` with the archived CPU reference artifact. Each command requires a fresh output path.

```powershell
python fd_cuda_pow2_audit.py --output fd_cuda_pow2_rhs_gradient_audit.json
python fd_cuda_pow2_replay.py --linear dense --output fd_cuda_pow2_replay_dense.json
python fd_cuda_pow2_replay.py --linear band --output fd_cuda_pow2_replay_band.json
python fd_cuda_pow2_device_residual.py --output fd_cuda_pow2_device_residual.json
```

The saved replay JSON records carry the CPU configuration, three-grid score rows, ordinal probe crossings, both wake diagnoses and hashes of the protocol and CUDA trajectories. The CPU reference digest and all comparison tolerances are in [`protocol_fd_cuda_pow2.json`](protocol_fd_cuda_pow2.json).
