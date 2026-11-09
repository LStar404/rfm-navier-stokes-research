# Captured CUDA graph timing and matched scale comparison

The earlier eager tail records CUDA events on the same stream between host API submissions. A gap before the next API submission can enter those intervals. The new diagnostic captures an additional graph chunk with explicit external CUDA event-record nodes, then substitutes it for the last ordinary graph chunk. Kernel launch arguments, time-step count and FFT plans are unchanged. The driver graph is inspected to confirm the expected number of event nodes. The preceding chunks and any final incomplete chunk remain uninstrumented. [CUDA Driver API event record flags](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-driver-api/group__CUDA__EVENT.html)

The sparse scheme marks one stage per sampled step, using two timing nodes on that step. For a 24-step chunk this yields four intervals per Galerkin stage or three per FD stage. The dense scheme puts markers at all stage boundaries on every sampled step and is more intrusive. The separate marker-only graph has no solver kernels and indicates the scale of driver and Windows WDDM scheduling variation. Its gaps are reported raw. Marker durations are not subtracted from solver stage intervals. These are graph-context stage intervals including device scheduling and instrumentation. The sparse medians are sampled from different steps and should not be summed as a measured step duration.

The t=1 validation compared instrumented and ordinary graph solves for RFM37, matched Legendre37 and FD176. All saved states and both probe arrays agreed bitwise. The first dense-marker tests encountered intermittent tens-of-microseconds marker-only gaps. Sparse timing reduced the sampling burden and gave consistent medium-scale Galerkin medians in the smoke. Smoke clocks do not contribute to isolated timings.

```powershell
python work/rfm_v7/cuda_graph_profile.py --horizon 1 --scheme sparse --output graph-check.json rfm37_k72 leg37_k72 fd176_k80
```

For a physical profile choose the full horizon. Keep this diagnostic separate from the latency comparison. The JSON preserves all stage samples and their step indices, event-node counts and marker controls.

```powershell
python work/rfm_v7/cuda_graph_profile.py --horizon 72 --scheme sparse --output graph-stages.json rfm37_k72 leg37_k72 fd176_k80
```

## Matched scales

The scale comparison holds (m,K,q), the step rate, layer functions and solver optimizations fixed across RFM and Legendre within each row. The three Galerkin scales are (35,64,60,128), (37,72,66,128), and (41,80,76,152). The high Legendre row deliberately uses 41 rather than the independently qualified Legendre39. The medium FD control uses FD176, K80 and 128 steps per time unit. CPU uses the recurrence Galerkin and buffered FD paths. Both native GPU dispatch modes use the same GPU operators, with graph replay toggled. The scale experiment concerns implementation and workload size. A speed comparison across discretizations at an accuracy gate uses separate qualification and independent-reference evidence.

An uninstrumented run records complete fresh in-process wall time and marching time separately. GPU marching ends after CUDA synchronization. Host model construction, CUDA runtime initialization, GPU transfer, FFT plan and graph setup, output transfer and teardown are included in wall time. Python import and OS or driver caches are not reset. Reference and implementation checks occur after the timed solve. CPU pinning is enabled in isolated mode, while all competing processes must be stopped separately.

Every completed trajectory and row is checkpointed. The journal verifies a frozen protocol, source hashes, reference hash and CPU affinity before resuming. The small smoke validated all 21 fresh combinations and six reused solves, including their matching field and probe checks.

```powershell
python work/rfm_v7/cuda_multiscale.py --mode pilot --horizon 1 --rounds 1 --reuse-repeats 2 --directory work/rfm_v7/scale-smoke-new
python work/rfm_v7/cuda_multiscale.py --mode isolated --confirm-isolated --horizon 72 --rounds 3 --reuse-repeats 3 --directory work/rfm_v7/scale-isolated
```

The optional reuse study runs after the fresh table, for the medium configurations and identical settings. Reusing the CUDA context and compiled module still leaves the original model construction, operator uploads, plans and graph capture in every timed solve. It records context creation and release separately, checks bitwise equality against a matching fresh trajectory, and estimates the amortization threshold from matched fresh and repeated times. A single matching fresh observation is retained as such.
