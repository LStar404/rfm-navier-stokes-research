# One-bridge SBDF2 study

The middle-resolution Galerkin experiments start at 128 steps per time unit and switch to 112 at $t=8$. One variable-step SBDF2 update carries the previous two states and nonlinear histories across the switch. The production march uses 8,192 steps, 8,191 nonlinear evaluations and 16,382 FFT batch calls. CUDA also uses two FFT calls for plan priming. The bridge adds no nonlinear evaluation or FFT call.

The [research report](outputs/RFM_NS_v7_分段步长与新进程计时_研究报告.pdf) separates implementation agreement, physical accuracy and timing. The [middle qualification](work/rfm_v7/switch_qualification_output/qualification.json) uses 21 implementation checkpoints and 19 checkpoints shared by both FD references. The [outer-scale qualification](work/rfm_v7/multiscale_switch_qualification_output/qualification.json) retains the lower-resolution failures and the separate higher-resolution results. The [FD qualification](work/rfm_v7/fd_early_multiscale_output/qualification.json) includes all three FD resolutions, and the [FD184 band follow-up](work/rfm_v7/fd184_band_followup/qualification.json) preserves the earlier dense implementation failure.

All 60 timed trajectories passed their numerical gates. The fixed 900-second observation window detected competing Python activity, so this cohort is **diagnostic only**: it supports no new isolated speed claim. The GPU schedule saved about 0.119 seconds in the march but added about 0.206-0.240 seconds before the march. It did not improve complete GPU solve time in this cohort. See the [phase decomposition](work/rfm_v7/switch_formal_timing_output/phase_diagnostics.json) and [independent audit](work/rfm_v7/switch_formal_timing_verification.json).

For performance comparisons, check `isolated_speed_claim_supported` in the [60-run summary](work/rfm_v7/switch_formal_timing_output/summary.json). The solver clock and the fresh-process return clock have separate definitions in the frozen protocol. Archived timing values from an externally disturbed or numerically failed cohort are diagnostic. Shared operator and schedule improvements do not establish an advantage of random features over Legendre.

## Replay from the repository root

Use the Windows Python, NumPy and SciPy environment described in [the original entry point](START_HERE_v7.md). The CPU-native arm uses the packaged x64 library. The CUDA arm also uses the NVIDIA driver and CUDA Toolkit/cuFFT from the documented native setup. Each command requires a new output path.

```powershell
python work/rfm_v7/replay_switch_round.py switch --method rfm37_k72 --seed 7331 --arm cpu_recurrence --check-archive --output replay/rfm7331_switched_cpu.npz
python work/rfm_v7/replay_switch_round.py switch --method leg37_k72 --arm cpu_native_contract --check-archive --output replay/leg17_switched_native.npz
python work/rfm_v7/replay_switch_round.py switch --method rfm37_k72 --seed 7331 --arm cuda_graph --check-archive --output replay/rfm7331_switched_cuda.npz
python work/rfm_v7/replay_switch_round.py fd-reference --axis time_fine --check-archive --output replay/fd_time_fine_t16.npz
```

For a shorter run, add `--horizon 9` to a Galerkin command or use the FD command below. The smoke checks saved arrays at the same physical times and carries no new $t=72$ accuracy result.

```powershell
python work/rfm_v7/replay_switch_round.py fd-reference --axis time_fine --horizon 1 --check-archive --output replay/fd_time_fine_t1.npz --json-output replay/fd_time_fine_t1.json
```

The numerical replay reads the published parameter protocols and advances from rest. The original early-reference freeze preserves the digest of a local administrative checkpoint. Its contents are outside this release. The replay entry point has no administrative dependency. The four published T6 histories and four derived restart inputs remain available for the original continuation audit.
